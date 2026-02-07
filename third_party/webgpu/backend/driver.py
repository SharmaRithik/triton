"""
WebGPU runtime driver for Triton.

Uses the ``wgpu`` Python library (wgpu-py) as the WebGPU runtime.
wgpu-py wraps wgpu-native (the Rust WebGPU implementation) and provides
cross-platform GPU access via Vulkan, Metal, or DX12 under the hood.

Install: pip install wgpu

The driver provides:
- Device detection and initialization
- Binary loading (WGSL -> compute pipeline)
- Kernel launching (dispatch compute workgroups)
- Buffer management for kernel arguments
"""

import os
import struct
import functools
import numpy as np
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


# ---------------------------------------------------------------------------
# Type mapping: Triton type strings -> Python struct format / sizes
# ---------------------------------------------------------------------------

def ty_to_cpp(ty):
    """Map Triton type strings to C/C++ type strings (used by code generator)."""
    if ty[0] == '*':
        return "uint64_t"   # Pointer -> buffer binding index
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


# ---------------------------------------------------------------------------
# WebGPU utility class: device management + binary loading
# ---------------------------------------------------------------------------

class WebGPUUtils:
    """
    Provides WebGPU device utilities.

    Wraps wgpu-py to provide:
    - Device/adapter management
    - Compute pipeline creation from WGSL
    - Device property queries
    """

    def __init__(self):
        import wgpu
        self._wgpu = wgpu
        # Request a high-performance GPU adapter
        self._adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        if self._adapter is None:
            raise RuntimeError("No WebGPU adapter found")

        # Request a device with reasonable limits
        self._device = self._adapter.request_device_sync(
            required_limits={
                "max_storage_buffer_binding_size": 1 * 1024 * 1024 * 1024,  # 1 GB
                "max_buffer_size": 1 * 1024 * 1024 * 1024,
                "max_compute_workgroup_size_x": 256,
                "max_compute_workgroup_size_y": 256,
                "max_compute_workgroup_size_z": 64,
                "max_compute_invocations_per_workgroup": 256,
                "max_compute_workgroups_per_dimension": 65535,
                "max_bind_groups": 4,
                "max_storage_buffers_per_shader_stage": 8,
                "max_uniform_buffer_binding_size": 65536,
            }
        )
        # Cache of compiled pipelines: (wgsl_bytes, entry_point) -> pipeline
        self._pipeline_cache = {}

    @property
    def device(self):
        return self._device

    @property
    def adapter(self):
        return self._adapter

    def load_binary(self, name, kernel, shared, device_idx):
        """
        Load a compiled WGSL shader and create a compute pipeline.

        Args:
            name: Kernel entry point name
            kernel: WGSL source code as bytes
            shared: Shared memory size (unused in WebGPU)
            device_idx: Device index (unused -- single device)

        Returns:
            (module, function, n_regs, n_spills, n_max_threads)
            - module: The wgpu ShaderModule
            - function: The wgpu ComputePipeline
            - n_regs: 0 (not applicable)
            - n_spills: 0 (not applicable)
            - n_max_threads: Max workgroup size
        """
        wgsl_source = kernel.decode("utf-8") if isinstance(kernel, bytes) else kernel
        cache_key = (wgsl_source, name)

        if cache_key in self._pipeline_cache:
            shader_module, pipeline = self._pipeline_cache[cache_key]
        else:
            shader_module = self._device.create_shader_module(code=wgsl_source)
            pipeline = self._device.create_compute_pipeline(
                layout="auto",
                compute={
                    "module": shader_module,
                    "entry_point": name,
                },
            )
            self._pipeline_cache[cache_key] = (shader_module, pipeline)

        max_threads = 256  # WebGPU typical max workgroup size
        return shader_module, pipeline, 0, 0, max_threads

    def get_device_properties(self, device_idx):
        """Return device properties dict matching what Triton expects."""
        adapter_info = self._adapter.request_adapter_info()
        return {
            "max_shared_mem": 16384,   # WebGPU workgroup storage limit (typically 16KB)
            "multiprocessor_count": 1,  # Not queryable via WebGPU
            "sm_clock_rate": 1000,      # Not queryable
            "mem_clock_rate": 1000,     # Not queryable
            "mem_bus_width": 256,       # Not queryable
            "max_num_regs": 0,
            "name": getattr(adapter_info, 'device', 'WebGPU Device'),
            "vendor": getattr(adapter_info, 'vendor', 'Unknown'),
            "architecture": getattr(adapter_info, 'architecture', 'Unknown'),
        }


# ---------------------------------------------------------------------------
# Launcher: dispatches compute shaders
# ---------------------------------------------------------------------------

class WebGPULauncher:
    """
    Launches WebGPU compute shaders.

    The launcher manages:
    - Buffer creation for kernel arguments
    - Bind group setup
    - Compute pass dispatch
    - Result readback
    """

    def __init__(self, src, metadata):
        """
        Initialize the launcher from compiled kernel source and metadata.

        Args:
            src: The compilation source (ASTSource or path)
            metadata: Kernel metadata (num_warps, shared, etc.)
        """
        self.src = src
        self.metadata = metadata
        # Extract signature info
        if hasattr(src, 'signature'):
            self.signature = src.signature
        else:
            self.signature = {}

    def __call__(self, gridX, gridY, gridZ, stream, function,
                 kernel_metadata, launch_metadata,
                 launch_enter_hook, launch_exit_hook, *args):
        """
        Dispatch the compute shader.

        Args:
            gridX, gridY, gridZ: Number of workgroups in each dimension
            stream: Queue/stream handle (the wgpu device)
            function: The wgpu ComputePipeline
            kernel_metadata: Packed metadata from backend.pack_metadata()
            launch_metadata: Optional launch metadata dict
            launch_enter_hook, launch_exit_hook: Optional hooks
            *args: Kernel arguments (tensors, scalars)
        """
        if launch_enter_hook:
            launch_enter_hook(launch_metadata and launch_metadata.get())

        pipeline = function
        # In our setup, stream is the wgpu device
        # (returned by get_current_stream)

        # Separate args into buffer args (tensors) and scalar args
        buffer_args = []
        scalar_args = []

        for i, arg in enumerate(args):
            if _is_tensor(arg):
                buffer_args.append(('tensor', arg))
            elif isinstance(arg, np.ndarray):
                buffer_args.append(('numpy', arg))
            else:
                scalar_args.append(arg)

        wgpu = _get_wgpu()
        wgpu_device = _get_global_utils().device

        # Create WebGPU buffers for tensor arguments
        bindings = []
        binding_idx = 0
        created_buffers = []

        for kind, arg in buffer_args:
            if kind == 'tensor':
                # Get tensor data as numpy
                data = arg.detach().cpu().numpy()
            elif kind == 'numpy':
                data = arg
            else:
                data = np.array(arg)

            buf = wgpu_device.create_buffer_with_data(
                data=data.tobytes(),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
            )
            created_buffers.append((buf, arg, data.nbytes))
            bindings.append({
                "binding": binding_idx,
                "resource": {"buffer": buf, "offset": 0, "size": data.nbytes},
            })
            binding_idx += 1

        # Create uniform buffer for scalar parameters
        if scalar_args:
            # Pack scalars into a uniform buffer
            # All scalars are packed as 32-bit values (u32 or f32)
            uniform_data = bytearray()
            for s in scalar_args:
                if isinstance(s, float):
                    uniform_data.extend(struct.pack('f', s))
                elif isinstance(s, int):
                    uniform_data.extend(struct.pack('I', s & 0xFFFFFFFF))
                else:
                    uniform_data.extend(struct.pack('I', int(s) & 0xFFFFFFFF))

            # Pad to 16-byte alignment (WebGPU uniform buffer requirement)
            remainder = len(uniform_data) % 16
            if remainder != 0:
                uniform_data.extend(b'\x00' * (16 - remainder))

            uniform_buf = wgpu_device.create_buffer_with_data(
                data=bytes(uniform_data),
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            bindings.append({
                "binding": binding_idx,
                "resource": {"buffer": uniform_buf, "offset": 0, "size": len(uniform_data)},
            })

        # Create bind group
        bind_group_layout = pipeline.get_bind_group_layout(0)
        bind_group = wgpu_device.create_bind_group(
            layout=bind_group_layout,
            entries=bindings,
        )

        # Dispatch compute shader
        command_encoder = wgpu_device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(gridX, gridY, gridZ)
        compute_pass.end()
        wgpu_device.queue.submit([command_encoder.finish()])

        # Read back results into the original tensor buffers
        for buf, original_arg, nbytes in created_buffers:
            if _is_tensor(original_arg):
                # Read buffer back and copy to the tensor
                result_data = wgpu_device.queue.read_buffer(buf)
                result_np = np.frombuffer(
                    result_data, dtype=_torch_dtype_to_numpy(original_arg.dtype)
                )[:original_arg.numel()]
                # Copy back to the PyTorch tensor
                torch = _import_torch()
                original_arg.copy_(
                    torch.from_numpy(result_np.copy()).reshape(original_arg.shape)
                )
            elif isinstance(original_arg, np.ndarray):
                result_data = wgpu_device.queue.read_buffer(buf)
                result_np = np.frombuffer(
                    result_data, dtype=original_arg.dtype
                )[:original_arg.size]
                np.copyto(original_arg, result_np.reshape(original_arg.shape))

        if launch_exit_hook:
            launch_exit_hook(launch_metadata and launch_metadata.get())


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@functools.lru_cache()
def _get_wgpu():
    import wgpu
    return wgpu


@functools.lru_cache()
def _get_global_utils():
    return WebGPUUtils()


def _import_torch():
    import torch
    return torch


def _is_tensor(obj):
    """Check if obj is a PyTorch tensor without importing torch at module level."""
    return hasattr(obj, 'data_ptr') and hasattr(obj, 'numel')


def _torch_dtype_to_numpy(dtype):
    """Map PyTorch dtype to numpy dtype."""
    torch = _import_torch()
    mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    return mapping.get(dtype, np.float32)


# ---------------------------------------------------------------------------
# Driver class
# ---------------------------------------------------------------------------

class WebGPUDriver(DriverBase):
    """
    Triton driver for WebGPU targets.

    Activation: The driver is active when:
    1. The ``wgpu`` Python package is installed, AND
    2. A WebGPU-capable GPU adapter is available, AND
    3. TRITON_WEBGPU_ENABLED=1 is set (opt-in, since WebGPU may
       conflict with CUDA/HIP when both are present)

    The opt-in requirement prevents the WebGPU driver from accidentally
    becoming the active driver on systems with NVIDIA/AMD GPUs where
    the CUDA/HIP driver should be preferred.
    """

    def __init__(self):
        super().__init__()
        self.utils = _get_global_utils()
        self.launcher_cls = WebGPULauncher
        self._device_idx = 0

    @classmethod
    def is_active(cls):
        """
        Check if WebGPU is available and opted-in.
        """
        if os.environ.get("TRITON_WEBGPU_ENABLED", "0") != "1":
            return False
        try:
            import wgpu  # noqa: F401
            adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance"
            )
            return adapter is not None
        except Exception:
            return False

    def get_current_target(self):
        """
        Return the current WebGPU GPUTarget.

        WebGPU abstracts away specific GPU architectures, so we use
        a generic "1" arch with a 32-thread warp size (common default).
        """
        return GPUTarget("webgpu", 1, 32)

    def get_current_device(self):
        """Return the current device index (single device for WebGPU)."""
        return self._device_idx

    def get_current_stream(self, device):
        """
        Return the 'stream' handle for WebGPU.

        In WebGPU there's no explicit stream concept -- the device queue
        serializes all submissions. We return the wgpu device itself,
        which the launcher uses to create command encoders.
        """
        return self.utils.device

    def get_active_torch_device(self):
        """
        Return the active torch device.

        PyTorch does not have a WebGPU device type yet. For now, we use
        the CPU device since data is transferred to WebGPU buffers during
        kernel launch.
        """
        import torch
        return torch.device("cpu")

    def get_benchmarker(self):
        """
        Return a benchmarking function for WebGPU kernels.
        """
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        """
        Return the device interface module.

        Since PyTorch has no WebGPU backend, we return a wrapper that
        provides minimal compatibility.
        """
        return _WebGPUDeviceInterface()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)


class _WebGPUDeviceInterface:
    """
    Minimal device interface for WebGPU that provides
    the methods Triton's runtime expects from a device module.
    """

    def current_device(self):
        return 0

    def set_device(self, device):
        pass  # Single device

    @staticmethod
    def is_available():
        return WebGPUDriver.is_active()
