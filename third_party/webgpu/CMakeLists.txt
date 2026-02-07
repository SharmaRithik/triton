# Triton WebGPU Backend - CMake Configuration
#
# Builds a minimal pybind11 plugin that registers the WebGPU backend
# with Triton's C++ layer.  The actual compilation (TTIR -> WGSL)
# and runtime (wgpu-py) are implemented in Python; this C++ module
# only provides the required init_triton_webgpu entry point.

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonWebGPU
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_webgpu.cc
  )
  target_link_libraries(TritonWebGPU PRIVATE Python3::Module pybind11::headers)
endif()
