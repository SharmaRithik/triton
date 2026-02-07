"""
WebGPU backend compiler for Triton.

Compilation pipeline:
    ttir -> wgsl

The pipeline skips the TTGIR and LLIR stages used by NVIDIA/AMD backends,
since those stages require hardware-specific lowering passes. Instead, the
TTIR (after standard optimization passes) is converted directly to WGSL
using the wgsl_emitter module.

This approach works for element-wise and simple compute kernels. More complex
patterns (e.g., tiled matmul, shared memory) will require additional lowering
passes as the backend matures.
"""

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes

from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import functools

from .wgsl_emitter import ttir_to_wgsl


@dataclass(frozen=True)
class WebGPUOptions:
    """Compilation options for the WebGPU backend."""
    num_warps: int = 1
    num_stages: int = 1
    num_ctas: int = 1
    # WebGPU workgroup size -- threads per workgroup
    warp_size: int = 64
    enable_fp_fusion: bool = True
    debug: bool = False
    backend_name: str = 'webgpu'
    supported_fp8_dtypes: Tuple[str] = ()
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    sanitize_overflow: bool = True
    arch: str = None
    instrumentation_mode: str = ""

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in sorted(self.__dict__.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class WebGPUBackend(BaseBackend):
    """
    Triton backend targeting WebGPU via WGSL (WebGPU Shading Language).

    The compilation pipeline is:
        ttir (Triton IR) -> wgsl (WebGPU Shading Language)

    The ttir stage applies standard Triton optimization passes (inlining,
    canonicalization, CSE, etc.) that are backend-agnostic. The wgsl stage
    then converts the optimized Triton IR to a WGSL compute shader using
    a Python-based converter (wgsl_emitter).

    The generated WGSL is returned as UTF-8 bytes (the "binary" format for
    this backend), which the WebGPU driver can load into a compute pipeline.
    """

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'webgpu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "wgsl"

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({
            k: opts[k]
            for k in WebGPUOptions.__dataclass_fields__.keys()
            if k in opts and opts[k] is not None
        })

        if "enable_fp_fusion" not in args:
            from triton import knobs
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        return WebGPUOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        codegen_fns = {
            "min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1),
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def load_dialects(self, ctx):
        # WebGPU backend currently has no custom MLIR dialects.
        # The standard Triton and arith/scf/math dialects suffice.
        pass

    # -------------------------------------------------------------------
    # Compilation stages
    # -------------------------------------------------------------------

    @staticmethod
    def make_ttir(mod, metadata, options):
        """
        Lower Triton IR: inline, canonicalize, CSE, etc.

        These are the same standard passes used by NVIDIA/AMD backends.
        They optimize the IR without introducing backend-specific operations.
        """
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        # WebGPU has no hardware tensor descriptor support
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_wgsl(mod, metadata, options):
        """
        Convert optimized Triton IR to WGSL.

        This stage:
        1. Extracts the MLIR textual form of the module
        2. Uses the wgsl_emitter to parse TTIR operations and generate WGSL
        3. Sets required metadata (name, shared, etc.)
        4. Returns the WGSL as UTF-8 bytes (the "binary" format)
        """
        # Get the MLIR text representation
        ttir_text = mod.str()

        # Extract kernel name from module
        kernel_name = mod.get_entry_func_name()
        if not kernel_name:
            kernel_name = "triton_kernel"

        # Compute block size from options
        block_size = options.num_warps * options.warp_size

        # Convert TTIR -> WGSL
        wgsl_source = ttir_to_wgsl(
            ttir_text=ttir_text,
            block_size=block_size,
            kernel_name=kernel_name,
        )

        # Set metadata required by the runtime
        metadata["name"] = kernel_name
        metadata["shared"] = 0   # WebGPU doesn't expose shared memory directly
        metadata["num_warps"] = options.num_warps
        metadata["global_scratch_size"] = 0
        metadata["global_scratch_align"] = 1

        # The WGSL text is our "binary" -- return as bytes
        return wgsl_source.encode("utf-8")

    def add_stages(self, stages, options, language):
        """
        Define the compilation pipeline for WebGPU.

        Pipeline: ttir -> wgsl

        The ttir stage optimizes the Triton IR.
        The wgsl stage converts it to a WebGPU compute shader.
        """
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["wgsl"] = lambda src, metadata: self.make_wgsl(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return f'webgpu-{self.target.arch}'
