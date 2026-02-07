/**
 * Triton WebGPU Backend - C++ / pybind11 module
 *
 * This is a minimal scaffold required by Triton's plugin system.
 * Every backend listed in TRITON_CODEGEN_BACKENDS must provide an
 * `init_triton_<name>` function (see python/src/main.cc).
 *
 * The WebGPU backend performs all compilation (TTIR -> WGSL) and
 * runtime work (wgpu-py) in Python, so this C++ module only needs
 * to expose a passes submodule (which is currently empty).
 *
 * As the backend matures, WebGPU-specific MLIR passes could be
 * added here (e.g., workgroup optimization, memory coalescing).
 */

#include "passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_webgpu(py::module &&m) {
  m.doc() = "Python bindings to the WebGPU Triton backend";

  // Every backend is expected to have a "passes" submodule.
  // For now it is empty — the TTIR -> WGSL conversion is done
  // entirely in Python (backend/wgsl_emitter.py).
  auto passes = m.def_submodule("passes");

  // Placeholder for future WebGPU-specific MLIR passes, e.g.:
  //   ADD_PASS_WRAPPER_0("add_wgsl_optimize", createWGSLOptimizePass);
}
