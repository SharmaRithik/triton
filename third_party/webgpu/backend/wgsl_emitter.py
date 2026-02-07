"""
Triton IR (TTIR) to WGSL (WebGPU Shading Language) converter.

Converts the textual form of Triton IR (after standard ttir passes) into
WGSL compute shaders that can be executed via the WebGPU API.

Supports a subset of TTIR operations sufficient for:
- Element-wise operations (add, sub, mul, div, etc.)
- Masked loads and stores
- Simple control flow (if/mask patterns)
- Basic math functions (exp, log, sqrt, etc.)
- Integer and floating-point arithmetic
- Type casts (sitofp, fptosi, uitofp, fptoui, ext, trunc)
- Reduction operations (mapped to per-thread ops for now)

The key mapping from TTIR's SIMT tensor model to WGSL's compute model:
- tt.get_program_id -> workgroup_id (which workgroup)
- tt.make_range     -> local_invocation_id (lane within workgroup)
- Tensor ops        -> scalar ops per invocation
- tt.load/store with mask -> bounds-checked buffer access

Limitations (to be addressed as the backend matures):
- No shared memory (workgroup storage) support
- No 2D block / tiled matmul patterns
- No atomic operations
- No cross-thread reductions (reduce ops are per-thread)
- bf16 / f64 / i64 emulated as f32 / f32 / i32
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set


# ---------------------------------------------------------------------------
# Type Mappings
# ---------------------------------------------------------------------------

TRITON_SCALAR_TO_WGSL = {
    "f32": "f32",
    "f16": "f16",
    "bf16": "f32",      # WebGPU lacks bf16; emulate as f32
    "f64": "f32",       # WebGPU lacks f64; emulate as f32
    "i1": "bool",
    "i8": "i32",
    "i16": "i32",
    "i32": "i32",
    "i64": "i32",       # WebGPU lacks 64-bit integers
    "u8": "u32",
    "u16": "u32",
    "u32": "u32",
    "u64": "u32",
}

# WGSL storage buffer element types (only these are valid for storage arrays)
WGSL_STORAGE_TYPES = {"f32", "i32", "u32", "f16"}

ARITH_BINARY_OPS = {
    "arith.addf": "+",
    "arith.subf": "-",
    "arith.mulf": "*",
    "arith.divf": "/",
    "arith.addi": "+",
    "arith.subi": "-",
    "arith.muli": "*",
    "arith.divsi": "/",
    "arith.divui": "/",
    "arith.remsi": "%",
    "arith.remui": "%",
    "arith.andi": "&",
    "arith.ori": "|",
    "arith.xori": "^",
    "arith.shli": "<<",
    "arith.shrsi": ">>",
    "arith.shrui": ">>",
    "arith.maxf": "max",   # handled as function
    "arith.minf": "min",   # handled as function
    "arith.maxsi": "max",
    "arith.minsi": "min",
    "arith.maxui": "max",
    "arith.minui": "min",
}

CMP_PREDICATES = {
    "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=",
    "ult": "<", "ule": "<=", "ugt": ">", "uge": ">=",
    "eq": "==", "ne": "!=",
}

CMPF_PREDICATES = {
    "oeq": "==", "ogt": ">", "oge": ">=",
    "olt": "<", "ole": "<=", "one": "!=",
    "ueq": "==", "ugt": ">", "uge": ">=",
    "ult": "<", "ule": "<=", "une": "!=",
    "ord": "==",   # ordered comparison (NaN handling not in WGSL)
    "uno": "!=",   # unordered
    "true": "true",
    "false": "false",
}

MATH_UNARY_OPS = {
    "math.exp": "exp",
    "math.exp2": "exp2",
    "math.log": "log",
    "math.log2": "log2",
    "math.sqrt": "sqrt",
    "math.rsqrt": "inverseSqrt",
    "math.absf": "abs",
    "math.absi": "abs",
    "math.ceil": "ceil",
    "math.floor": "floor",
    "math.round": "round",
    "math.sin": "sin",
    "math.cos": "cos",
    "math.tanh": "tanh",
    "math.fma": None,       # ternary — handled separately
    "arith.negf": "-",      # unary minus
}


# ---------------------------------------------------------------------------
# Internal Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Value:
    """Represents an SSA value tracked during conversion."""
    name: str
    wgsl_expr: str
    value_type: str = "scalar"     # "scalar", "ptr", "ptr_offset", "mask"
    buffer_arg: str = ""           # For ptr values: which arg buffer they came from
    wgsl_data_type: str = "f32"    # The WGSL data type of this value


@dataclass
class ArgInfo:
    """Information about a kernel argument."""
    name: str           # Clean name (e.g., "arg0")
    ssa_name: str       # SSA name (e.g., "%arg0")
    triton_type: str    # Full type string (e.g., "!tt.ptr<f32>")
    is_pointer: bool = False
    pointee_type: str = ""     # For pointers: element type (e.g., "f32")
    wgsl_elem_type: str = ""   # WGSL element type (e.g., "f32")
    binding_idx: int = 0


# ---------------------------------------------------------------------------
# Main Converter
# ---------------------------------------------------------------------------

def ttir_to_wgsl(ttir_text: str, block_size: int = 256, kernel_name: str = "") -> str:
    """
    Convert Triton IR text to a WGSL compute shader.

    Args:
        ttir_text: Textual form of the Triton IR module (from mod.str()).
        block_size: Workgroup size for the compute shader.
        kernel_name: Override for the kernel name. If empty, parsed from IR.

    Returns:
        Valid WGSL compute shader source code.

    Raises:
        ValueError: If no tt.func is found in the input.
    """
    converter = _TTIRToWGSLConverter(ttir_text, block_size, kernel_name)
    return converter.convert()


class _TTIRToWGSLConverter:
    """Converts parsed Triton IR to WGSL."""

    def __init__(self, ttir_text: str, block_size: int, kernel_name: str):
        self.ttir_text = ttir_text
        self.block_size = block_size
        self.kernel_name = kernel_name
        self.args: List[ArgInfo] = []
        self.values: Dict[str, Value] = {}
        self.body_lines: List[str] = []
        self.mask_expr: Optional[str] = None
        self.stores: List[Tuple[str, str, str]] = []  # (buf, idx_expr, val_expr)
        self.stored_buffers: Set[str] = set()  # buffer names that receive stores
        self.loaded_buffers: Set[str] = set()  # buffer names that are loaded from

    def convert(self) -> str:
        self._parse_function_signature()
        self._parse_body()
        return self._emit_wgsl()

    # -------------------------------------------------------------------
    # 1. Parse the function signature
    # -------------------------------------------------------------------

    def _parse_function_signature(self):
        sig_match = re.search(
            r'tt\.func\s+(?:public\s+)?@(\w+)\s*\(([^)]*)\)',
            self.ttir_text, re.DOTALL
        )
        if not sig_match:
            raise ValueError("No tt.func found in TTIR text")

        if not self.kernel_name:
            self.kernel_name = sig_match.group(1)

        args_text = sig_match.group(2)
        binding_idx = 0

        # Match individual arguments: %argN: TYPE  or  %argN: TYPE {attrs}
        for m in re.finditer(r'(%arg\d+)\s*:\s*([^,\)]+)', args_text):
            ssa_name = m.group(1)
            raw_type = m.group(2).strip()
            # Remove trailing attribute annotations like {tt.divisibility = ...}
            raw_type = re.sub(r'\s*\{[^}]*\}', '', raw_type).strip()

            is_ptr = '!tt.ptr' in raw_type
            pointee = ''
            if is_ptr:
                pm = re.search(r'!tt\.ptr<(\w+)>', raw_type)
                pointee = pm.group(1) if pm else 'f32'

            wgsl_elem = TRITON_SCALAR_TO_WGSL.get(pointee if is_ptr else raw_type, 'f32')
            # Ensure storage element type is valid
            if is_ptr and wgsl_elem not in WGSL_STORAGE_TYPES:
                wgsl_elem = 'f32'

            self.args.append(ArgInfo(
                name=f"arg{binding_idx}",
                ssa_name=ssa_name,
                triton_type=raw_type,
                is_pointer=is_ptr,
                pointee_type=pointee,
                wgsl_elem_type=wgsl_elem,
                binding_idx=binding_idx,
            ))
            binding_idx += 1

    # -------------------------------------------------------------------
    # 2. Parse the function body
    # -------------------------------------------------------------------

    def _parse_body(self):
        # Find function body between the first { and tt.return
        body_match = re.search(
            r'tt\.func[^{]*\{\s*(.*?)tt\.return',
            self.ttir_text, re.DOTALL
        )
        if not body_match:
            return

        body = body_match.group(1)
        for line in body.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('^'):
                continue
            self._process_line(line)

    def _process_line(self, line: str):
        """Process a single TTIR line and update our value tracking."""
        # Split into result = rest
        result = None
        rest = line

        rm = re.match(r'(%[^\s=]+)\s*=\s*(.*)', line)
        if rm:
            result = rm.group(1)
            rest = rm.group(2)

        # Dispatch to handlers — order matters for disambiguation
        if 'tt.get_program_id' in rest:
            self._handle_get_program_id(result, rest)
        elif 'arith.constant' in rest:
            self._handle_constant(result, rest)
        elif 'tt.make_range' in rest:
            self._handle_make_range(result, rest)
        elif 'tt.splat' in rest:
            self._handle_splat(result, rest)
        elif 'tt.broadcast' in rest:
            self._handle_broadcast(result, rest)
        elif 'tt.expand_dims' in rest:
            self._handle_expand_dims(result, rest)
        elif 'tt.addptr' in rest:
            self._handle_addptr(result, rest)
        elif 'tt.load' in rest:
            self._handle_load(result, rest)
        elif 'tt.store' in rest:
            self._handle_store(rest)
        elif 'arith.cmpi' in rest:
            self._handle_cmpi(result, rest)
        elif 'arith.cmpf' in rest:
            self._handle_cmpf(result, rest)
        elif 'arith.sitofp' in rest:
            self._handle_cast(result, rest, 'f32')
        elif 'arith.fptosi' in rest:
            self._handle_cast(result, rest, 'i32')
        elif 'arith.uitofp' in rest:
            self._handle_cast(result, rest, 'f32')
        elif 'arith.fptoui' in rest:
            self._handle_cast(result, rest, 'u32')
        elif 'arith.extf' in rest or 'arith.truncf' in rest:
            self._handle_cast(result, rest, 'f32')
        elif 'arith.extsi' in rest or 'arith.trunci' in rest:
            self._handle_cast(result, rest, 'i32')
        elif 'arith.extui' in rest:
            self._handle_cast(result, rest, 'u32')
        elif 'arith.index_cast' in rest:
            self._handle_cast(result, rest, 'i32')
        elif 'arith.bitcast' in rest:
            self._handle_bitcast(result, rest)
        elif 'arith.select' in rest:
            self._handle_select(result, rest)
        elif 'math.fma' in rest:
            self._handle_fma(result, rest)
        elif any(op in rest for op in MATH_UNARY_OPS if MATH_UNARY_OPS[op] is not None):
            self._handle_math_unary(result, rest)
        elif any(op in rest for op in ARITH_BINARY_OPS):
            self._handle_arith_binary(result, rest)
        elif 'tt.reduce' in rest or 'tt.scan' in rest:
            # Reductions are not supported in per-thread mode yet.
            # Pass through the input operand as-is.
            self._handle_reduce_passthrough(result, rest)

    # -------------------------------------------------------------------
    # Operation Handlers
    # -------------------------------------------------------------------

    def _handle_get_program_id(self, result, rest):
        m = re.search(r'tt\.get_program_id\s+(\w+)', rest)
        axis = m.group(1) if m else 'x'
        if result:
            self.values[result] = Value(result, f"i32(workgroup_id.{axis})")

    def _handle_constant(self, result, rest):
        m = re.search(r'arith\.constant\s+([^\s:]+)', rest)
        if m and result:
            val = m.group(1)
            # Handle different constant formats
            if val.startswith('true'):
                val = 'true'
            elif val.startswith('false'):
                val = 'false'
            elif 'dense' in val:
                # dense<...> constant — extract the value
                dm = re.search(r'dense<([^>]+)>', rest)
                if dm:
                    val = dm.group(1).strip()
                else:
                    val = '0'
            elif '.' in val and 'x' not in val:
                # Float constant — ensure it has proper format
                val = val.rstrip(':')
                try:
                    fval = float(val)
                    val = f"{fval}"
                except ValueError:
                    pass
            else:
                val = val.rstrip(':')
            self.values[result] = Value(result, val)

    def _handle_make_range(self, result, rest):
        # Handle both {start = 0 : i32, end = 256 : i32} and {end = 256 : i32, start = 0 : i32}
        start_m = re.search(r'start\s*=\s*(\d+)', rest)
        end_m = re.search(r'end\s*=\s*(\d+)', rest)
        if start_m and end_m and result:
            start, end = int(start_m.group(1)), int(end_m.group(1))
            self.block_size = end - start
            self.values[result] = Value(result, "i32(local_id.x)")

    def _handle_splat(self, result, rest):
        m = re.match(r'tt\.splat\s+(%\S+)', rest)
        if not m or not result:
            return
        operand = m.group(1)

        # Check if it's a function argument
        arg = self._get_arg_by_ssa(operand)
        if arg:
            if arg.is_pointer:
                self.values[result] = Value(result, "", "ptr", buffer_arg=arg.name)
            else:
                wgsl_type = self._scalar_arg_wgsl_type(arg)
                self.values[result] = Value(
                    result, f"params.{arg.name}", "scalar",
                    wgsl_data_type=wgsl_type
                )
        elif operand in self.values:
            v = self.values[operand]
            self.values[result] = Value(
                result, v.wgsl_expr, v.value_type,
                buffer_arg=v.buffer_arg, wgsl_data_type=v.wgsl_data_type
            )

    def _handle_broadcast(self, result, rest):
        """tt.broadcast — same value for every lane; just propagate."""
        m = re.match(r'tt\.broadcast\s+(%\S+)', rest)
        if m and result:
            operand = m.group(1)
            if operand in self.values:
                v = self.values[operand]
                self.values[result] = Value(
                    result, v.wgsl_expr, v.value_type,
                    buffer_arg=v.buffer_arg, wgsl_data_type=v.wgsl_data_type
                )
            else:
                self.values[result] = Value(result, self._resolve_expr(operand))

    def _handle_expand_dims(self, result, rest):
        """tt.expand_dims — adds a size-1 dimension; no effect for per-thread model."""
        m = re.match(r'tt\.expand_dims\s+(%\S+)', rest)
        if m and result:
            operand = m.group(1)
            if operand in self.values:
                v = self.values[operand]
                self.values[result] = Value(
                    result, v.wgsl_expr, v.value_type,
                    buffer_arg=v.buffer_arg, wgsl_data_type=v.wgsl_data_type
                )
            else:
                self.values[result] = Value(result, self._resolve_expr(operand))

    def _handle_addptr(self, result, rest):
        m = re.match(r'tt\.addptr\s+(%\S+),\s*(%\S+)', rest)
        if not m or not result:
            return
        base_name = m.group(1)
        offset_name = m.group(2)

        base_val = self.values.get(base_name)
        offset_expr = self._resolve_expr(offset_name)

        if base_val and base_val.value_type == "ptr":
            self.values[result] = Value(
                result, offset_expr, "ptr_offset",
                buffer_arg=base_val.buffer_arg
            )
        elif base_val and base_val.value_type == "ptr_offset":
            # Nested addptr — add offsets
            combined = f"({base_val.wgsl_expr} + {offset_expr})"
            self.values[result] = Value(
                result, combined, "ptr_offset",
                buffer_arg=base_val.buffer_arg
            )
        else:
            self.values[result] = Value(
                result, f"({self._resolve_expr(base_name)} + {offset_expr})"
            )

    def _handle_load(self, result, rest):
        # tt.load %ptr [, %mask] [, %other]
        parts = re.match(r'tt\.load\s+(%\S+)(?:,\s*(%\S+))?(?:,\s*(%\S+))?', rest)
        if not parts or not result:
            return

        ptr_name = parts.group(1)
        ptr_val = self.values.get(ptr_name)

        if ptr_val and ptr_val.value_type == "ptr_offset":
            buf = f"buf_{ptr_val.buffer_arg}"
            idx = ptr_val.wgsl_expr
            # Track which buffers are loaded
            self.loaded_buffers.add(buf)
            # Determine data type from the buffer's argument
            arg = self._get_arg_by_name(ptr_val.buffer_arg)
            data_type = arg.wgsl_elem_type if arg else "f32"
            self.values[result] = Value(
                result, f"{buf}[{idx}]",
                wgsl_data_type=data_type
            )
        else:
            self.values[result] = Value(result, "0.0")

    def _handle_store(self, rest):
        # tt.store %ptr, %val [, %mask]
        m = re.match(r'tt\.store\s+(%\S+),\s*(%\S+)', rest)
        if not m:
            return

        ptr_name = m.group(1)
        val_name = m.group(2).rstrip(',')

        ptr_val = self.values.get(ptr_name)
        val_expr = self._resolve_expr(val_name)

        if ptr_val and ptr_val.value_type == "ptr_offset":
            buf = f"buf_{ptr_val.buffer_arg}"
            idx = ptr_val.wgsl_expr
            self.stores.append((buf, idx, val_expr))
            self.stored_buffers.add(buf)

    def _handle_arith_binary(self, result, rest):
        for op_name, op_sym in ARITH_BINARY_OPS.items():
            if op_name not in rest:
                continue

            m = re.match(rf'{re.escape(op_name)}\s+(%\S+),\s*(%\S+)', rest)
            if m and result:
                lhs_expr = self._resolve_expr(m.group(1))
                rhs_expr = self._resolve_expr(m.group(2))

                # max/min are WGSL builtins, not infix operators
                if op_sym in ("max", "min"):
                    expr = f"{op_sym}({lhs_expr}, {rhs_expr})"
                else:
                    expr = f"({lhs_expr} {op_sym} {rhs_expr})"

                self.values[result] = Value(result, expr)
            break

    def _handle_cmpi(self, result, rest):
        m = re.match(r'arith\.cmpi\s+(\w+),\s*(%\S+),\s*(%\S+)', rest)
        if m and result:
            pred = m.group(1)
            lhs = self._resolve_expr(m.group(2))
            rhs = self._resolve_expr(m.group(3))
            op = CMP_PREDICATES.get(pred, "<")
            expr = f"({lhs} {op} {rhs})"
            self.values[result] = Value(result, expr, "mask")
            self.mask_expr = expr

    def _handle_cmpf(self, result, rest):
        m = re.match(r'arith\.cmpf\s+(\w+),\s*(%\S+),\s*(%\S+)', rest)
        if m and result:
            pred = m.group(1)
            lhs = self._resolve_expr(m.group(2))
            rhs = self._resolve_expr(m.group(3))
            op = CMPF_PREDICATES.get(pred, "<")
            if op in ("true", "false"):
                expr = op
            else:
                expr = f"({lhs} {op} {rhs})"
            self.values[result] = Value(result, expr, "mask")

    def _handle_cast(self, result, rest, target_type):
        m = re.match(r'\S+\s+(%\S+)', rest)
        if m and result:
            operand = self._resolve_expr(m.group(1))
            self.values[result] = Value(
                result, f"{target_type}({operand})",
                wgsl_data_type=target_type
            )

    def _handle_bitcast(self, result, rest):
        """Bitcast — in WGSL, use bitcast<target>(operand)."""
        m = re.match(r'arith\.bitcast\s+(%\S+)', rest)
        if m and result:
            operand = self._resolve_expr(m.group(1))
            # Determine target type from the result type annotation
            type_m = re.search(r':\s*\S+\s+to\s+(\S+)', rest)
            if type_m:
                target = type_m.group(1).strip()
                # Extract scalar type from tensor type
                scalar_m = re.search(r'<\d+x(\w+)>', target)
                if scalar_m:
                    target = scalar_m.group(1)
                wgsl_target = TRITON_SCALAR_TO_WGSL.get(target, 'f32')
            else:
                wgsl_target = 'f32'
            self.values[result] = Value(
                result, f"bitcast<{wgsl_target}>({operand})",
                wgsl_data_type=wgsl_target
            )

    def _handle_select(self, result, rest):
        m = re.match(r'arith\.select\s+(%\S+),\s*(%\S+),\s*(%\S+)', rest)
        if m and result:
            cond = self._resolve_expr(m.group(1))
            true_val = self._resolve_expr(m.group(2))
            false_val = self._resolve_expr(m.group(3))
            expr = f"select({false_val}, {true_val}, {cond})"
            self.values[result] = Value(result, expr)

    def _handle_fma(self, result, rest):
        """math.fma %a, %b, %c -> fma(a, b, c)"""
        m = re.match(r'math\.fma\s+(%\S+),\s*(%\S+),\s*(%\S+)', rest)
        if m and result:
            a = self._resolve_expr(m.group(1))
            b = self._resolve_expr(m.group(2))
            c = self._resolve_expr(m.group(3))
            expr = f"fma({a}, {b}, {c})"
            self.values[result] = Value(result, expr)

    def _handle_math_unary(self, result, rest):
        for op_name, wgsl_fn in MATH_UNARY_OPS.items():
            if wgsl_fn is None:
                continue  # Skip ternary ops handled elsewhere
            if op_name not in rest:
                continue
            m = re.match(rf'{re.escape(op_name)}\s+(%\S+)', rest)
            if m and result:
                operand = self._resolve_expr(m.group(1))
                if wgsl_fn == "-":
                    expr = f"-({operand})"
                else:
                    expr = f"{wgsl_fn}({operand})"
                self.values[result] = Value(result, expr)
            break

    def _handle_reduce_passthrough(self, result, rest):
        """
        Stub handler for tt.reduce / tt.scan.

        In a per-thread execution model, cross-thread reductions are not
        meaningful.  For now we just propagate the first operand so that
        subsequent uses don't crash the emitter.  A proper implementation
        would use workgroupBarrier() + shared memory.
        """
        m = re.match(r'tt\.\w+\s+(%\S+)', rest)
        if m and result:
            operand = m.group(1)
            self.values[result] = Value(result, self._resolve_expr(operand))

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _get_arg_by_ssa(self, ssa: str) -> Optional[ArgInfo]:
        for arg in self.args:
            if arg.ssa_name == ssa:
                return arg
        return None

    def _get_arg_by_name(self, name: str) -> Optional[ArgInfo]:
        for arg in self.args:
            if arg.name == name:
                return arg
        return None

    def _scalar_arg_wgsl_type(self, arg: ArgInfo) -> str:
        """Determine the WGSL type for a scalar kernel argument."""
        tt = arg.triton_type
        if tt in ('f32', 'f16', 'bf16', 'f64'):
            return 'f32'
        if tt in ('i1', 'i8', 'i16', 'i32', 'i64'):
            return 'u32'  # uniform buffers use u32 for integer scalars
        if tt in ('u8', 'u16', 'u32', 'u64'):
            return 'u32'
        return 'u32'  # default for unknown scalar types

    def _resolve_expr(self, ssa: str) -> str:
        """Get the WGSL expression for an SSA value or argument."""
        if ssa in self.values:
            return self.values[ssa].wgsl_expr

        arg = self._get_arg_by_ssa(ssa)
        if arg:
            if arg.is_pointer:
                return f"/*buf_{arg.name}*/"
            else:
                return f"params.{arg.name}"

        # Might be a constant embedded in the line (e.g., "true", "false")
        return ssa

    # -------------------------------------------------------------------
    # 3. Emit WGSL
    # -------------------------------------------------------------------

    def _emit_wgsl(self) -> str:
        lines: List[str] = []
        ptr_args = [a for a in self.args if a.is_pointer]
        scalar_args = [a for a in self.args if not a.is_pointer]

        lines.append("// Auto-generated by Triton WebGPU backend")
        lines.append(f"// Kernel: {self.kernel_name}")
        lines.append("")

        # ---- Storage buffer declarations ----
        for arg in ptr_args:
            buf_name = f"buf_{arg.name}"
            access = "read_write" if buf_name in self.stored_buffers else "read"
            lines.append(
                f"@group(0) @binding({arg.binding_idx}) "
                f"var<storage, {access}> {buf_name}: array<{arg.wgsl_elem_type}>;"
            )

        # ---- Uniform buffer for scalar parameters ----
        if scalar_args:
            lines.append("")
            lines.append("struct Params {")
            for arg in scalar_args:
                wgsl_type = self._scalar_arg_wgsl_type(arg)
                lines.append(f"    {arg.name}: {wgsl_type},")
            lines.append("}")
            # Binding index comes after all pointer args
            params_binding = len(ptr_args)
            lines.append(
                f"@group(0) @binding({params_binding}) "
                f"var<uniform> params: Params;"
            )

        # ---- Compute shader function ----
        lines.append("")
        lines.append(f"@compute @workgroup_size({self.block_size})")
        lines.append(f"fn {self.kernel_name}(")
        lines.append("    @builtin(workgroup_id) workgroup_id: vec3<u32>,")
        lines.append("    @builtin(local_invocation_id) local_id: vec3<u32>,")
        lines.append("    @builtin(global_invocation_id) global_id: vec3<u32>")
        lines.append(") {")

        # ---- Body ----
        if self.stores:
            if self.mask_expr:
                lines.append(f"    if {self.mask_expr} {{")
                for buf, idx, val in self.stores:
                    lines.append(f"        {buf}[{idx}] = {val};")
                lines.append("    }")
            else:
                for buf, idx, val in self.stores:
                    lines.append(f"    {buf}[{idx}] = {val};")
        else:
            lines.append("    // No store operations detected -- kernel may be incomplete")

        lines.append("}")
        lines.append("")
        return "\n".join(lines)
