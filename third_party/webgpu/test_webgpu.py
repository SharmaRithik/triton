#!/usr/bin/env python3
"""
Tests for the Triton WebGPU backend.

Test categories:
1. WGSL Emitter unit tests -- verify TTIR -> WGSL conversion (no GPU needed)
2. WebGPU runtime tests -- verify compute shader dispatch (needs wgpu + GPU)

Run with: python -m pytest third_party/webgpu/test_webgpu.py -v
Or standalone: python third_party/webgpu/test_webgpu.py
"""

import sys
import os
import unittest
import textwrap

# Ensure the backend can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# ===========================================================================
# 1. WGSL Emitter Unit Tests (no GPU required)
# ===========================================================================

class TestWGSLEmitter(unittest.TestCase):
    """Test TTIR -> WGSL conversion without needing GPU or Triton build."""

    def test_vector_add_conversion(self):
        """Test converting a vector-add TTIR to WGSL."""
        from backend.wgsl_emitter import ttir_to_wgsl

        ttir_text = textwrap.dedent("""\
            module {
              tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
                %0 = tt.get_program_id x : i32
                %c256_i32 = arith.constant 256 : i32
                %1 = arith.muli %0, %c256_i32 : i32
                %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
                %3 = tt.splat %1 : i32 -> tensor<256xi32>
                %4 = arith.addi %3, %2 : tensor<256xi32>
                %5 = tt.splat %arg3 : i32 -> tensor<256xi32>
                %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
                %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
                %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
                %9 = tt.load %8, %6 : tensor<256x!tt.ptr<f32>>
                %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
                %11 = tt.addptr %10, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
                %12 = tt.load %11, %6 : tensor<256x!tt.ptr<f32>>
                %13 = arith.addf %9, %12 : tensor<256xf32>
                %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
                %15 = tt.addptr %14, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
                tt.store %15, %13, %6 : tensor<256x!tt.ptr<f32>>
                tt.return
              }
            }
        """)

        wgsl = ttir_to_wgsl(ttir_text, block_size=256, kernel_name="add_kernel")

        # Verify the output is valid WGSL structure
        self.assertIn("@compute @workgroup_size(256)", wgsl)
        self.assertIn("fn add_kernel(", wgsl)
        self.assertIn("buf_arg0", wgsl)  # First input buffer
        self.assertIn("buf_arg1", wgsl)  # Second input buffer
        self.assertIn("buf_arg2", wgsl)  # Output buffer
        self.assertIn("Params", wgsl)    # Scalar params struct
        self.assertIn("var<storage", wgsl)

        # Verify storage buffer declarations and access modes
        self.assertIn("@binding(0) var<storage, read>", wgsl)
        self.assertIn("@binding(1) var<storage, read>", wgsl)
        self.assertIn("@binding(2) var<storage, read_write>", wgsl)

        # Verify the store operation references the correct buffers
        self.assertIn("buf_arg2[", wgsl)

        # Verify mask-guarded store
        self.assertIn("if (", wgsl)

        print("Generated WGSL for vector_add:")
        print("=" * 60)
        print(wgsl)
        print("=" * 60)

    def test_scalar_multiply_conversion(self):
        """Test converting a scalar-multiply TTIR to WGSL."""
        from backend.wgsl_emitter import ttir_to_wgsl

        ttir_text = textwrap.dedent("""\
            module {
              tt.func public @scalar_mul(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: f32) {
                %0 = tt.get_program_id x : i32
                %c128_i32 = arith.constant 128 : i32
                %1 = arith.muli %0, %c128_i32 : i32
                %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
                %3 = tt.splat %1 : i32 -> tensor<128xi32>
                %4 = arith.addi %3, %2 : tensor<128xi32>
                %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
                %6 = tt.addptr %5, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
                %7 = tt.load %6 : tensor<128x!tt.ptr<f32>>
                %8 = tt.splat %arg2 : f32 -> tensor<128xf32>
                %9 = arith.mulf %7, %8 : tensor<128xf32>
                %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
                %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
                tt.store %11, %9 : tensor<128x!tt.ptr<f32>>
                tt.return
              }
            }
        """)

        wgsl = ttir_to_wgsl(ttir_text, block_size=128, kernel_name="scalar_mul")

        self.assertIn("@compute @workgroup_size(128)", wgsl)
        self.assertIn("fn scalar_mul(", wgsl)
        self.assertIn("buf_arg0", wgsl)
        self.assertIn("buf_arg1", wgsl)
        # Scalar arg should be in Params struct
        self.assertIn("Params", wgsl)
        self.assertIn("params.arg2", wgsl)

        print("Generated WGSL for scalar_mul:")
        print("=" * 60)
        print(wgsl)
        print("=" * 60)

    def test_kernel_name_extraction(self):
        """Test that kernel name is correctly extracted from TTIR."""
        from backend.wgsl_emitter import ttir_to_wgsl

        ttir = textwrap.dedent("""\
            module {
              tt.func public @my_custom_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %c64 = arith.constant 64 : i32
                %1 = arith.muli %0, %c64 : i32
                %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %3 = tt.splat %1 : i32 -> tensor<64xi32>
                %4 = arith.addi %3, %2 : tensor<64xi32>
                %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                %7 = tt.load %6 : tensor<64x!tt.ptr<f32>>
                %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %9 = tt.addptr %8, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                tt.store %9, %7 : tensor<64x!tt.ptr<f32>>
                tt.return
              }
            }
        """)

        wgsl = ttir_to_wgsl(ttir, block_size=64)

        self.assertIn("fn my_custom_kernel(", wgsl)
        self.assertIn("@compute @workgroup_size(64)", wgsl)

    def test_no_func_raises(self):
        """Test that missing tt.func raises an error."""
        from backend.wgsl_emitter import ttir_to_wgsl

        with self.assertRaises(ValueError):
            ttir_to_wgsl("module { }", block_size=64)

    def test_buffer_access_modes(self):
        """Test that read and read_write access modes are correct."""
        from backend.wgsl_emitter import ttir_to_wgsl

        # Two input buffers (read) and one output buffer (read_write)
        ttir = textwrap.dedent("""\
            module {
              tt.func public @test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %c64 = arith.constant 64 : i32
                %1 = arith.muli %0, %c64 : i32
                %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %3 = tt.splat %1 : i32 -> tensor<64xi32>
                %4 = arith.addi %3, %2 : tensor<64xi32>
                %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                %7 = tt.load %6 : tensor<64x!tt.ptr<f32>>
                %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %9 = tt.addptr %8, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                %10 = tt.load %9 : tensor<64x!tt.ptr<f32>>
                %11 = arith.addf %7, %10 : tensor<64xf32>
                %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %13 = tt.addptr %12, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                tt.store %13, %11 : tensor<64x!tt.ptr<f32>>
                tt.return
              }
            }
        """)

        wgsl = ttir_to_wgsl(ttir, block_size=64)

        # arg0 and arg1 are read-only, arg2 is written
        self.assertIn("@binding(0) var<storage, read>", wgsl)
        self.assertIn("@binding(1) var<storage, read>", wgsl)
        self.assertIn("@binding(2) var<storage, read_write>", wgsl)

    def test_math_unary_ops(self):
        """Test math unary operations (exp, log, sqrt, etc.)."""
        from backend.wgsl_emitter import ttir_to_wgsl

        ttir = textwrap.dedent("""\
            module {
              tt.func public @math_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %c64 = arith.constant 64 : i32
                %1 = arith.muli %0, %c64 : i32
                %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %3 = tt.splat %1 : i32 -> tensor<64xi32>
                %4 = arith.addi %3, %2 : tensor<64xi32>
                %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                %7 = tt.load %6 : tensor<64x!tt.ptr<f32>>
                %8 = math.exp %7 : tensor<64xf32>
                %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %10 = tt.addptr %9, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                tt.store %10, %8 : tensor<64x!tt.ptr<f32>>
                tt.return
              }
            }
        """)

        wgsl = ttir_to_wgsl(ttir, block_size=64, kernel_name="math_test")

        self.assertIn("exp(", wgsl)
        self.assertIn("fn math_test(", wgsl)

    def test_integer_types(self):
        """Test i32 pointer types generate correct WGSL."""
        from backend.wgsl_emitter import ttir_to_wgsl

        ttir = textwrap.dedent("""\
            module {
              tt.func public @int_kernel(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
                %0 = tt.get_program_id x : i32
                %c64 = arith.constant 64 : i32
                %1 = arith.muli %0, %c64 : i32
                %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %3 = tt.splat %1 : i32 -> tensor<64xi32>
                %4 = arith.addi %3, %2 : tensor<64xi32>
                %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
                %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
                %7 = tt.load %6 : tensor<64x!tt.ptr<i32>>
                %8 = arith.muli %7, %7 : tensor<64xi32>
                %9 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
                %10 = tt.addptr %9, %4 : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
                tt.store %10, %8 : tensor<64x!tt.ptr<i32>>
                tt.return
              }
            }
        """)

        wgsl = ttir_to_wgsl(ttir, block_size=64, kernel_name="int_kernel")

        self.assertIn("array<i32>", wgsl)
        self.assertIn("fn int_kernel(", wgsl)

    def test_select_operation(self):
        """Test arith.select generates correct WGSL select()."""
        from backend.wgsl_emitter import ttir_to_wgsl

        ttir = textwrap.dedent("""\
            module {
              tt.func public @select_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
                %0 = tt.get_program_id x : i32
                %c64 = arith.constant 64 : i32
                %1 = arith.muli %0, %c64 : i32
                %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %3 = tt.splat %1 : i32 -> tensor<64xi32>
                %4 = arith.addi %3, %2 : tensor<64xi32>
                %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                %7 = tt.load %6 : tensor<64x!tt.ptr<f32>>
                %cst = arith.constant 0.0 : f32
                %8 = tt.splat %cst : f32 -> tensor<64xf32>
                %9 = tt.splat %arg2 : i32 -> tensor<64xi32>
                %10 = arith.cmpi slt, %4, %9 : tensor<64xi32>
                %11 = arith.select %10, %7, %8 : tensor<64xf32>
                %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %13 = tt.addptr %12, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                tt.store %13, %11 : tensor<64x!tt.ptr<f32>>
                tt.return
              }
            }
        """)

        wgsl = ttir_to_wgsl(ttir, block_size=64, kernel_name="select_test")

        self.assertIn("select(", wgsl)
        print("Generated WGSL for select_test:")
        print(wgsl)

    def test_wgsl_syntactic_validity(self):
        """
        Basic syntactic sanity checks on generated WGSL.
        
        Ensures braces match, required keywords are present, etc.
        """
        from backend.wgsl_emitter import ttir_to_wgsl

        ttir = textwrap.dedent("""\
            module {
              tt.func public @syntax_check(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %c64 = arith.constant 64 : i32
                %1 = arith.muli %0, %c64 : i32
                %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %3 = tt.splat %1 : i32 -> tensor<64xi32>
                %4 = arith.addi %3, %2 : tensor<64xi32>
                %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                %7 = tt.load %6 : tensor<64x!tt.ptr<f32>>
                %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %9 = tt.addptr %8, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                tt.store %9, %7 : tensor<64x!tt.ptr<f32>>
                tt.return
              }
            }
        """)

        wgsl = ttir_to_wgsl(ttir, block_size=64, kernel_name="syntax_check")

        # Check balanced braces
        self.assertEqual(wgsl.count('{'), wgsl.count('}'),
                         "Unbalanced braces in generated WGSL")

        # Check required decorators
        self.assertIn("@compute", wgsl)
        self.assertIn("@workgroup_size", wgsl)
        self.assertIn("@group(0)", wgsl)
        self.assertIn("@binding(", wgsl)
        self.assertIn("@builtin(workgroup_id)", wgsl)
        self.assertIn("@builtin(local_invocation_id)", wgsl)

        # Check it ends with a newline (WGSL convention)
        self.assertTrue(wgsl.endswith("\n"), "WGSL should end with newline")

    def test_cast_sitofp(self):
        """Test integer-to-float cast (arith.sitofp)."""
        from backend.wgsl_emitter import ttir_to_wgsl

        ttir = textwrap.dedent("""\
            module {
              tt.func public @cast_test(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %c64 = arith.constant 64 : i32
                %1 = arith.muli %0, %c64 : i32
                %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %3 = tt.splat %1 : i32 -> tensor<64xi32>
                %4 = arith.addi %3, %2 : tensor<64xi32>
                %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
                %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
                %7 = tt.load %6 : tensor<64x!tt.ptr<i32>>
                %8 = arith.sitofp %7 : tensor<64xi32> to tensor<64xf32>
                %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %10 = tt.addptr %9, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                tt.store %10, %8 : tensor<64x!tt.ptr<f32>>
                tt.return
              }
            }
        """)

        wgsl = ttir_to_wgsl(ttir, block_size=64, kernel_name="cast_test")

        self.assertIn("f32(", wgsl)  # sitofp should produce an f32() cast


# ===========================================================================
# 2. WebGPU Runtime Tests (require wgpu + GPU)
# ===========================================================================

def _has_wgpu():
    """Check if wgpu is installed and a GPU adapter is available."""
    try:
        import wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        return adapter is not None
    except Exception:
        return False


@unittest.skipUnless(_has_wgpu(), "wgpu not available or no GPU adapter found")
class TestWebGPURuntime(unittest.TestCase):
    """Test WebGPU runtime operations using wgpu-py directly."""

    def test_simple_compute_dispatch(self):
        """Test dispatching a simple WGSL compute shader via wgpu."""
        import wgpu
        import numpy as np

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync()

        # Simple WGSL that doubles each element
        wgsl_code = """\
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output_data[idx] = input_data[idx] * 2.0;
}
"""
        # Create shader and pipeline
        shader = device.create_shader_module(code=wgsl_code)
        pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "main"},
        )

        # Create input data
        n = 256
        input_data = np.arange(n, dtype=np.float32)

        # Create buffers
        input_buf = device.create_buffer_with_data(
            data=input_data.tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        output_buf = device.create_buffer(
            size=n * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        # Bind and dispatch
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": input_buf, "offset": 0, "size": n * 4}},
                {"binding": 1, "resource": {"buffer": output_buf, "offset": 0, "size": n * 4}},
            ],
        )

        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(n // 64)
        compute_pass.end()
        device.queue.submit([encoder.finish()])

        # Read back and verify
        result_bytes = device.queue.read_buffer(output_buf)
        result = np.frombuffer(result_bytes, dtype=np.float32)

        expected = input_data * 2.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        print(f"WebGPU compute dispatch test passed! ({n} elements doubled correctly)")

    def test_load_binary_from_driver(self):
        """Test the WebGPU driver's load_binary function."""
        from backend.driver import WebGPUUtils

        utils = WebGPUUtils()

        wgsl_code = """\
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn test_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = f32(gid.x);
}
"""
        module, pipeline, n_regs, n_spills, max_threads = utils.load_binary(
            "test_kernel", wgsl_code.encode("utf-8"), 0, 0
        )

        self.assertIsNotNone(module)
        self.assertIsNotNone(pipeline)
        self.assertEqual(n_regs, 0)
        self.assertEqual(n_spills, 0)
        self.assertGreater(max_threads, 0)
        print("load_binary test passed!")

    def test_device_properties(self):
        """Test querying device properties."""
        from backend.driver import WebGPUUtils

        utils = WebGPUUtils()
        props = utils.get_device_properties(0)

        self.assertIn("max_shared_mem", props)
        self.assertIn("name", props)
        self.assertGreater(props["max_shared_mem"], 0)
        print(f"Device properties: {props}")

    def test_generated_wgsl_dispatch(self):
        """
        End-to-end test: generate WGSL from TTIR and dispatch on GPU.
        
        This is the most important test -- it verifies that the WGSL
        code generated by wgsl_emitter actually computes the correct
        result when run on real hardware via wgpu-py.
        """
        import wgpu
        import numpy as np
        from backend.wgsl_emitter import ttir_to_wgsl

        # Generate WGSL from TTIR
        ttir_text = textwrap.dedent("""\
            module {
              tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
                %0 = tt.get_program_id x : i32
                %c64_i32 = arith.constant 64 : i32
                %1 = arith.muli %0, %c64_i32 : i32
                %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %3 = tt.splat %1 : i32 -> tensor<64xi32>
                %4 = arith.addi %3, %2 : tensor<64xi32>
                %5 = tt.splat %arg3 : i32 -> tensor<64xi32>
                %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
                %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                %9 = tt.load %8, %6 : tensor<64x!tt.ptr<f32>>
                %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %11 = tt.addptr %10, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                %12 = tt.load %11, %6 : tensor<64x!tt.ptr<f32>>
                %13 = arith.addf %9, %12 : tensor<256xf32>
                %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                tt.store %15, %13, %6 : tensor<64x!tt.ptr<f32>>
                tt.return
              }
            }
        """)

        wgsl_source = ttir_to_wgsl(ttir_text, block_size=64, kernel_name="add_kernel")
        print("Generated WGSL for dispatch test:")
        print(wgsl_source)

        # Now dispatch the generated WGSL
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync()

        n = 256
        x = np.random.rand(n).astype(np.float32)
        y = np.random.rand(n).astype(np.float32)
        output = np.zeros(n, dtype=np.float32)

        shader = device.create_shader_module(code=wgsl_source)
        pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "add_kernel"},
        )

        x_buf = device.create_buffer_with_data(
            data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        y_buf = device.create_buffer_with_data(
            data=y.tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        out_buf = device.create_buffer(
            size=n * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        # Pack n_elements as uniform buffer
        import struct
        params_data = struct.pack('I', n)
        # Pad to 16 bytes
        params_data = params_data + b'\x00' * (16 - len(params_data))
        params_buf = device.create_buffer_with_data(
            data=params_data, usage=wgpu.BufferUsage.UNIFORM
        )

        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": x_buf, "offset": 0, "size": n * 4}},
                {"binding": 1, "resource": {"buffer": y_buf, "offset": 0, "size": n * 4}},
                {"binding": 2, "resource": {"buffer": out_buf, "offset": 0, "size": n * 4}},
                {"binding": 3, "resource": {"buffer": params_buf, "offset": 0, "size": 16}},
            ],
        )

        num_workgroups = (n + 63) // 64
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(num_workgroups)
        compute_pass.end()
        device.queue.submit([encoder.finish()])

        result = np.frombuffer(device.queue.read_buffer(out_buf), dtype=np.float32)
        expected = x + y

        np.testing.assert_allclose(result, expected, rtol=1e-5)
        print(f"Generated WGSL dispatch test PASSED! Correctly added {n} element vectors.")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Triton WebGPU Backend Tests")
    print("=" * 60)
    print()

    # Run tests
    unittest.main(verbosity=2)
