# Triton WebGPU Backend

A WebGPU backend for [Triton](https://github.com/triton-lang/triton) that compiles Triton kernels to [WGSL](https://www.w3.org/TR/WGSL/) (WebGPU Shading Language) and dispatches them via [wgpu-py](https://github.com/pygfx/wgpu-py).

## Overview

This backend enables Triton to target WebGPU-capable GPUs using the standard WebGPU API, providing cross-platform GPU compute through Vulkan, Metal, or DX12 under the hood (via wgpu-native).

### Architecture

```
Triton @jit kernel (Python)
         |
         v
    Triton AST -> TTIR (Triton IR, MLIR-based)
         |
         v  [make_ttir: standard passes — inline, CSE, canonicalize]
    Optimized TTIR
         |
         v  [make_wgsl: wgsl_emitter.py — Python-based converter]
    WGSL compute shader (UTF-8 bytes)
         |
         v  [WebGPU driver: wgpu-py runtime]
    GPU execution via Vulkan / Metal / DX12
```

### How It Differs From NVIDIA/AMD Backends

| Aspect | NVIDIA/AMD | WebGPU |
|--------|-----------|--------|
| Compilation pipeline | TTIR → TTGIR → LLIR → PTX/AMDGCN | TTIR → WGSL |
| IR lowering | C++ MLIR passes (TritonGPU dialect) | Python regex-based converter |
| Shared memory | Hardware shared memory (smem) | Not supported yet |
| Runtime | CUDA/HIP C++ drivers via pybind11 | Python `wgpu-py` library |
| Binary format | PTX/cubin or HSACO | WGSL text (UTF-8 bytes) |
| C++ module | Heavy (dialects, passes, translations) | Minimal scaffold |

## File Structure

```
third_party/webgpu/
├── README.md               # This file
├── CMakeLists.txt           # CMake build for the C++ pybind11 plugin
├── triton_webgpu.cc         # Minimal C++ module (init_triton_webgpu)
├── conftest.py              # Pytest configuration for test discovery
├── test_webgpu.py           # Unit + integration tests
└── backend/
    ├── __init__.py          # Package marker
    ├── compiler.py          # WebGPUBackend (compilation pipeline)
    ├── driver.py            # WebGPUDriver + WebGPULauncher (runtime)
    └── wgsl_emitter.py      # TTIR → WGSL converter (core logic)
```

## Prerequisites

### Required

- **Python 3.10+**
- **wgpu-py** (Python WebGPU bindings):
  ```bash
  pip install wgpu
  ```
- **numpy**:
  ```bash
  pip install numpy
  ```

### Optional (for full Triton integration)

- **PyTorch** (for `@triton.jit` tensor arguments)
- **Triton build dependencies** (LLVM, MLIR, pybind11, cmake) — only needed if building Triton from source

### GPU Requirements

A GPU with Vulkan 1.1+, Metal 2.0+, or DirectX 12 support. Most modern GPUs (2016+) qualify. The wgpu-native library auto-detects the best backend.

## Getting Started

### 1. Install Dependencies

```bash
pip install wgpu numpy
```

### 2. Run the Emitter Unit Tests (No GPU Needed)

The WGSL emitter tests verify TTIR → WGSL conversion without requiring a GPU or a full Triton build:

```bash
# From the triton root directory
cd third_party/webgpu
python -m pytest test_webgpu.py -v -k "TestWGSLEmitter"
```

Or standalone:

```bash
python test_webgpu.py
```

### 3. Run the GPU Runtime Tests (Requires wgpu + GPU)

These tests dispatch actual compute shaders via wgpu-py:

```bash
python -m pytest test_webgpu.py -v -k "TestWebGPURuntime"
```

### 4. Run All Tests

```bash
python -m pytest test_webgpu.py -v
```

### 5. Build Triton With WebGPU Backend

```bash
# From the triton root
pip install -e python/

# The setup.py is already configured to include "webgpu" in the backend list.
# CMake will build triton_webgpu.cc as part of the plugin system.
```

### 6. Use the WebGPU Backend (After Triton Build)

```python
import os
os.environ["TRITON_WEBGPU_ENABLED"] = "1"

import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

n = 1024
x = torch.randn(n)
y = torch.randn(n)
out = torch.empty(n)

add_kernel[(n + 255) // 256,](x, y, out, n, BLOCK_SIZE=256)

print(torch.allclose(out, x + y))  # True
```

## Design Decisions

### Why TTIR → WGSL Directly?

The NVIDIA/AMD backends use a multi-stage pipeline: TTIR → TTGIR (TritonGPU IR) → LLIR (LLVM IR) → PTX/AMDGCN. Each stage introduces hardware-specific abstractions (warps, shared memory layout, register allocation, etc.).

For the initial WebGPU backend, we bypass these stages because:

1. **WebGPU abstracts hardware details** — there are no warps, registers, or shared memory banks to optimize for.
2. **WGSL is high-level enough** — it maps well to TTIR's operations (element-wise, load/store, control flow).
3. **Simplicity** — a Python-based converter avoids the complexity of building custom MLIR passes, SPIR-V toolchain integration, and LLVM backend modifications.
4. **Incremental approach** — start with a working subset, then add complexity (shared memory, tiled matmul, atomics) as needed.

### Why wgpu-py?

[wgpu-py](https://github.com/pygfx/wgpu-py) wraps [wgpu-native](https://github.com/gfx-rs/wgpu-native), the Rust WebGPU implementation. Benefits:

- **Cross-platform**: Works on Linux (Vulkan), macOS (Metal), and Windows (DX12/Vulkan).
- **No browser required**: Runs natively, unlike browser-based WebGPU.
- **Python-native**: No C++ compilation needed for the runtime.
- **Well-maintained**: Active development, tracks the WebGPU spec closely.

### Why Opt-In (TRITON_WEBGPU_ENABLED)?

On systems with NVIDIA/AMD GPUs, the CUDA/HIP drivers should take priority. The WebGPU driver uses an environment variable opt-in (`TRITON_WEBGPU_ENABLED=1`) to prevent it from accidentally becoming the active driver. This is checked in `WebGPUDriver.is_active()`.

### Data Transfer Model

Since PyTorch has no native WebGPU device type, data flows through CPU:

```
PyTorch CPU Tensor → numpy → wgpu Buffer → GPU compute → wgpu Buffer → numpy → PyTorch CPU Tensor
```

This incurs host↔device transfer overhead on each kernel launch. For production use, a persistent buffer pool would reduce this cost.

## TTIR → WGSL Mapping

### Core Operation Mapping

| TTIR Operation | WGSL Equivalent |
|---------------|-----------------|
| `tt.get_program_id x` | `workgroup_id.x` |
| `tt.make_range {start=0, end=N}` | `local_invocation_id.x` |
| `tt.splat %scalar` | Scalar propagation (identity) |
| `tt.addptr %base, %offset` | Buffer index expression |
| `tt.load %ptr, %mask` | `buf[idx]` inside `if (mask)` |
| `tt.store %ptr, %val, %mask` | `buf[idx] = val` inside `if (mask)` |
| `arith.addf / addi / mulf / ...` | `+` / `*` / ... (WGSL operators) |
| `arith.cmpi slt` | `<` (comparison) |
| `arith.select` | `select(false_val, true_val, cond)` |
| `math.exp / log / sqrt / ...` | `exp() / log() / sqrt()` (WGSL builtins) |
| `math.fma` | `fma(a, b, c)` |
| `arith.sitofp` | `f32(int_val)` |
| `arith.fptosi` | `i32(float_val)` |
| `arith.bitcast` | `bitcast<type>(val)` |

### Buffer Binding Layout

```wgsl
// Pointer arguments → storage buffers (binding 0, 1, 2, ...)
@group(0) @binding(0) var<storage, read>       buf_arg0: array<f32>;
@group(0) @binding(1) var<storage, read>       buf_arg1: array<f32>;
@group(0) @binding(2) var<storage, read_write> buf_arg2: array<f32>;

// Scalar arguments → uniform buffer (last binding)
struct Params { arg3: u32, }
@group(0) @binding(3) var<uniform> params: Params;
```

- Pointer arguments become storage buffers. Read-only if only loaded from; read_write if stored to.
- Scalar arguments are packed into a single uniform buffer as 32-bit values (u32 for integers, f32 for floats).
- Binding indices are assigned in argument order.

### Type Mapping

| Triton Type | WGSL Type | Notes |
|------------|-----------|-------|
| `f32` | `f32` | Direct mapping |
| `f16` | `f16` | Requires `f16` extension |
| `bf16` | `f32` | Emulated (upcast) |
| `f64` | `f32` | Emulated (precision loss) |
| `i32` | `i32` | Direct mapping |
| `i64` | `i32` | Truncated |
| `u32` | `u32` | Direct mapping |

## Supported Kernel Patterns

### Currently Supported

- **Element-wise operations**: add, sub, mul, div, neg, abs
- **Math functions**: exp, exp2, log, log2, sqrt, rsqrt, sin, cos, tanh, ceil, floor, round, fma
- **Comparisons**: all integer and float comparisons
- **Masked load/store**: bounds checking via `if (mask) { ... }`
- **Type casts**: sitofp, fptosi, uitofp, fptoui, ext, trunc, bitcast
- **Select**: conditional value selection
- **Scalar parameters**: via uniform buffer
- **Multi-buffer kernels**: arbitrary number of input/output buffers

### Not Yet Supported

- **Shared memory / workgroup storage**: `tl.zeros`, `tl.reduce` with shared memory
- **Atomic operations**: `tl.atomic_add`, `tl.atomic_cas`, etc.
- **Cross-thread reductions**: `tl.sum`, `tl.max` (per-thread only for now)
- **Tiled matrix multiply**: `tl.dot` / matmul patterns
- **2D/3D block patterns**: Only 1D workgroups currently
- **Dynamic indexing with strides**: Complex address calculations
- **FP8 types**: Not available in WebGPU

## Troubleshooting

### `wgpu` Import Fails

```
pip install wgpu
```

On Linux, you may need Vulkan drivers:
```bash
# Ubuntu/Debian
sudo apt install mesa-vulkan-drivers

# For NVIDIA
sudo apt install nvidia-driver-XXX  # or from NVIDIA's repo
```

### No GPU Adapter Found

Check that your GPU supports Vulkan 1.1+ (or Metal on macOS):
```python
import wgpu
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
print(adapter.request_adapter_info() if adapter else "No adapter!")
```

### WebGPU Driver Not Activating

Make sure the environment variable is set:
```bash
export TRITON_WEBGPU_ENABLED=1
```

### WGSL Shader Compilation Error

If the generated WGSL fails to compile, the kernel likely uses an unsupported TTIR pattern. Check the WGSL output:

```python
from third_party.webgpu.backend.wgsl_emitter import ttir_to_wgsl
wgsl = ttir_to_wgsl(ttir_text, block_size=256, kernel_name="my_kernel")
print(wgsl)  # Inspect the generated WGSL
```

### Build Issues

The C++ module (`triton_webgpu.cc`) is minimal and only requires pybind11. If CMake fails:
1. Ensure `pybind11` is installed: `pip install pybind11`
2. Check that `TRITON_CODEGEN_BACKENDS` includes `webgpu` in the CMake output
3. The `passes.h` header is in `python/src/` which is globally included

## Development Roadmap

### Phase 1: Foundation (Current)
- [x] TTIR → WGSL emitter for element-wise kernels
- [x] wgpu-py runtime driver (buffers, pipelines, dispatch)
- [x] Triton backend registration (compiler + driver)
- [x] Unit tests for emitter
- [x] Integration tests with wgpu-py GPU dispatch
- [ ] Full end-to-end test with `@triton.jit` (requires Triton build on Linux)

### Phase 2: Robustness
- [ ] Workgroup shared memory (`var<workgroup>`)
- [ ] Cross-thread reductions via shared memory
- [ ] Atomic operations (`atomicAdd`, `atomicMax`, etc.)
- [ ] Better error messages for unsupported patterns
- [ ] Persistent buffer pool (avoid per-launch allocation)

### Phase 3: Performance
- [ ] 2D workgroup tiling for matmul
- [ ] Memory coalescing analysis
- [ ] Workgroup size auto-tuning
- [ ] Pipeline caching improvements
- [ ] Async buffer transfers

### Phase 4: Feature Parity
- [ ] `tl.dot` / matrix multiply
- [ ] Complex control flow (loops, nested ifs)
- [ ] Multi-dimensional grid dispatch
- [ ] FP16 compute shaders
- [ ] Custom MLIR passes (C++) for advanced optimizations

## Contributing

When adding new features to this backend:

1. **Add the TTIR operation handler** in `wgsl_emitter.py` → `_process_line()` and the corresponding `_handle_*` method.
2. **Add a unit test** in `test_webgpu.py` → `TestWGSLEmitter` that verifies the TTIR → WGSL conversion for the new pattern.
3. **Add a GPU test** in `test_webgpu.py` → `TestWebGPURuntime` that dispatches the generated WGSL and verifies correctness.
4. **Update this README** with the new supported pattern.

## References

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [wgpu-py Documentation](https://wgpu-py.readthedocs.io/)
- [wgpu-native (Rust)](https://github.com/gfx-rs/wgpu-native)
- [Triton Documentation](https://triton-lang.org/)
- [Triton Backend Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)
