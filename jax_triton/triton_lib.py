# Copyright 2022 The jax_triton Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for calling Triton kernels from JAX."""
import collections
import functools
import math
import os
import pickle

from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple, Union

import jax
import jaxlib
from jax import tree_util
from jax._src import core
from jax._src import state
from jax._src import util
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
from jax._src.typing import Array
import jax.dlpack
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.lib import xla_client as xc
import jax.numpy as jnp
from jax_triton import triton_kernel_call_lib
import numpy as np

CAN_USE_TRITON = False
try:
  import triton
  import triton.compiler as tc
  import triton.language as tl
  CAN_USE_TRITON = True
except ModuleNotFoundError:
  pass

os.environ["TRITON_CACHE_DIR"] = ""
map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


def get_triton_type(obj: Any) -> str:
  type_map = {
      jnp.dtype("bfloat16"): "bf16",
      jnp.dtype("float64"): "fp64",
      jnp.dtype("float32"): "fp32",
      jnp.dtype("float16"): "fp16",
      # Triton has 'fp8' as well which Jax doesn't support yet.

      jnp.dtype("int64"): "i64",
      jnp.dtype("int32"): "i32",
      jnp.dtype("int16"): "i16",
      jnp.dtype("int8"): "i8",

      jnp.dtype("uint64"): "u64",
      jnp.dtype("uint32"): "u32",
      jnp.dtype("uint16"): "u16",
      jnp.dtype("uint8"): "u8",

      # Triton defines a 'B' type, which is an alias for both i1 and bool.
      jnp.dtype("bool"): "B",
  }

  if isinstance(obj, (jax.core.ShapedArray, state.ShapedArrayRef)):
    return f"*{type_map[obj.dtype]}"
  if isinstance(obj, tl.constexpr):
    obj = obj.value
  if isinstance(obj, int):
    if -2**31 <= obj < 2**31:
      return "i32"
    elif 2**31 <= obj < 2**32:
      return "u32"
    elif -2**63 <= obj < 2**63:
      return "i64"
    elif 2**63 <= obj < 2**64:
      return "u64"
    else:
      raise ValueError(f"integer overflow representing {obj}")
  if isinstance(obj, float):
    return "f"
  if isinstance(obj, bool):
    return "B"
  if isinstance(obj, str):
    return "str"
  raise NotImplementedError(
      f"could not compute type name for {obj}: {type(obj)}"
  )


def get_triton_python_ir(aval):
  return get_triton_type(aval)

Metaparameters = Any
ShapeDtypeDuck = Any

Grid = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]
GridOrLambda = Union[Grid, Callable[[Any], Grid]]

triton_kernel_call_p = jax.core.Primitive('triton_kernel_call')
triton_kernel_call_p.multiple_results = True

triton_kernel_call_p.def_impl(
    functools.partial(xla.apply_primitive, triton_kernel_call_p))

@triton_kernel_call_p.def_abstract_eval
def triton_kernel_call_abstract_eval(*_, out_shapes, **__):
  return [
      core.ShapedArray(out_shape.shape, out_shape.dtype)
      for out_shape in out_shapes
  ]


def aval_to_layout(aval):
  arange = np.arange(aval.ndim, dtype="int64")[::-1].copy()
  return ir.DenseIntElementsAttr.get(arange, type=ir.IndexType.get())


def avals_to_layouts(avals):
  return ir.ArrayAttr.get([aval_to_layout(a) for a in avals])


def compile_triton_func(
    avals_in, avals_out, triton_func, num_warps, num_stages, metaparams):
  metadata = {triton_func.arg_names.index(k): v for k, v in metaparams.items()}
  all_args = [*avals_in, *avals_out]
  signature = {i: get_triton_python_ir(a) for i, a in enumerate(all_args)}
  # TODO(sharadmv,zhangqiaorjc): handle differently aligned pointers
  specialization = collections.namedtuple(
      "instance_descriptor", ["divisible_by_16", "equal_to_1"])(
          tuple(range(len(all_args))), ())
  # TODO(sharadmv): handle multiple devices, right now we assume device 0 which
  # is fine when we have multiple of the same GPU but this won't work in
  # general.
  asm, shared_mem, name = tc._compile(
      triton_func, signature=signature, device=0,
      specialization=specialization,
      constants=metadata, num_warps=num_warps,
      num_stages=num_stages, extern_libs={}, output="cubin")
  return name, asm, shared_mem


def emit_triton_kernel_call(
    ctx,
    name,
    asm_map,
    shared_mem,
    *,
    dump_binary_path: Optional[str],
    grid: GridOrLambda,
    metaparams,
    num_warps,
):
  if dump_binary_path is not None:
    binary = dict(asm=asm_map, shared_mem=shared_mem, name=name)
    with open(dump_binary_path, "wb") as fp:
      pickle.dump(binary, fp)

  # Prefer cubin to PTX, if available.
  asm = asm_map.get("cubin", asm_map["ptx"])

  if callable(grid):
    grid = grid(metaparams)
  grid = list(grid)
  grid.extend([1] * (3 - len(grid)))  # Fill with `1`s to length 3.
  grid_0, grid_1, grid_2 = grid
  arity = len(ctx.avals_in) + len(ctx.avals_out)

  return triton_kernel_call_lib.TritonExecutable(
      asm, name, arity, grid_0, grid_1, grid_2, num_warps, shared_mem
  )


def triton_kernel_call_lowering(
    ctx,
    *args,
    kernel_name,
    call_name,
    asm,
    shared_mem,
    out_shapes,
    grid: GridOrLambda,
    num_warps: int,
    dump_binary_path: Optional[str],
    input_output_aliases: Tuple[Tuple[int, int], ...],
    **metaparams,
):
  if jaxlib.version.__version_info__ < (0, 3, 22) and input_output_aliases:
    raise NotImplementedError(
        "`input_output_aliases` only supported on `jaxlib>=0.3.22")
  out_type = ir.TupleType.get_tuple([
      ir.RankedTensorType.get(out_shape.shape, mlir.dtype_to_ir_type(out_shape.dtype))
      for out_shape in out_shapes])
  i32_type = ir.IntegerType.get_signless(32)
  executable = emit_triton_kernel_call(
      ctx,
      kernel_name,
      asm.asm_map,
      shared_mem,
      dump_binary_path=dump_binary_path,
      num_warps=num_warps,
      grid=grid,
      metaparams=metaparams,
  )
  ctx.module_context.add_keepalive(executable)
  output_operand_aliases = ir.ArrayAttr.get(
      [
          mhlo.OutputOperandAlias.get(
              output_tuple_indices=[output],
              operand_index=input,
              operand_tuple_indices=[],
          )
          for input, output in input_output_aliases
      ]
  )
  out = mhlo.CustomCallOp(
      [out_type],
      args,
      call_target_name=ir.StringAttr.get(call_name),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(executable.descriptor),
      api_version=ir.IntegerAttr.get(i32_type, 1),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=avals_to_layouts(ctx.avals_in),
      result_layouts=avals_to_layouts(ctx.avals_out),
      output_operand_aliases=output_operand_aliases,
  )
  results = [mhlo.GetTupleElementOp(out, mlir.i32_attr(i)).result
             for i in range(len(out_shapes))]
  return results


mlir.register_lowering(triton_kernel_call_p, triton_kernel_call_lowering)

class Asm:
  """Hides the huge ASM dict from showing up in Jaxprs."""
  def __init__(self, asm_map):
    self.asm_map = asm_map

class ShapeDtype(Protocol):

  @property
  def shape(self) -> Tuple[int, ...]:
    ...

  @property
  def dtype(self) -> np.dtype:
    ...

def triton_call(*args: Array, kernel: triton.JITFunction,
                out_shape: Union[ShapeDtype, Sequence[ShapeDtype]],
                grid: Union[int, GridOrLambda],
                call_name: str = "triton_kernel_call",
                num_warps: int = 4, num_stages: int = 2,
                dump_binary_path: Optional[str] = None,
                input_output_aliases: Dict[int, int] = {},
                **metaparams: Any):
  """Calls a Triton kernel with `jax.Array` arguments.

  Args:
    *args: `jax.Array`-valued inputs for the Triton kernel.
    kernel: A Triton kernel (e.g. a function decorated with `triton.jit`). The
      Triton kernel cannot take in scalar-valued arguments and all static values
      should be annotated with `triton.language.constexpr`.
    out_shape: A `jax.ShapeDtypeStruct` (or something that has `.shape` and
      `.dtype` attributes) or a sequence thereof that specify the output(s) of
      the kernel. Pointers for each of the `jax.ShapeDtypeStruct`s in `out_shape`
      will be passed into `kernel` following the pointers corresponding to the
      inputs.
    grid: An integer, tuple of up to 3 integers, or a function that returns a
      tuple of up to 3 integers. When `grid` is an integer, `kernel` is
      invocated in `grid`-many parallel executions. When `grid` is a sequence of
      integers, `kernel` is launched in a `prod(grid)`-many parallel execution.
      When `grid` is a function, it is passed `**metaparams` and should return a
      tuple of up to 3 integers.
    input_output_aliases: A dictionary mapping input argument indices to output
      indices. Providing a mapping will alias the corresponding buffers.
    num_warps: The number of warps used to execute the Triton kernel.
    num_stages: The number of stages emitted by the Triton compiler.
    **metaparams: Additional keyword arguments that will be provided to a `grid`
      (if it is a function) and to the Triton kernel as `constexpr` arguments.

  ## Example usage

  First we define a simple kernel that adds two vectors.

  ```python
  import triton
  import triton.language as tl

  @triton.jit
  def add_kernel(
      x_ptr,
      y_ptr,
      output_ptr,
      block_size: tl.constexpr,
  ):
    \"\"\"Adds two vectors.\"\"\"
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < 8
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
  ```

  Then we use `triton_call` to call it from JAX.

  ```python
  import jax
  import jax.numpy as jnp
  import jax_triton as jt

  def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    block_size = 8
    return jt.triton_call(
        x,
        y,
        kernel=add_kernel,
        out_shape=out_shape,
        grid=(x.size // block_size,),
        block_size=block_size)

  x_val = jnp.arange(8)
  y_val = jnp.arange(8, 16)
  print(add(x_val, y_val))
  print(jax.jit(add)(x_val, y_val))
  ```
  """
  if not CAN_USE_TRITON:
    raise ValueError(
        "`triton_call` is only available when `triton` is installed."
    )
  xc.register_custom_call_target(
      call_name, triton_kernel_call_lib.get_custom_call(), platform="CUDA"
  )
  if isinstance(grid, int):
    grid = (grid,)
  out_shape = tree_util.tree_map(
      lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_shape)
  flat_args, in_tree = tree_util.tree_flatten(args)
  del in_tree
  # TODO(sharadmv): check in_tree is flat (no Pytrees allowed in triton_call)
  flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)
  avals_in = [core.raise_to_shaped(core.get_aval(a)) for a in flat_args]
  avals_out = [core.ShapedArray(a.shape, a.dtype) for a in flat_out_shapes]
  kernel_name, asm_map, shared_mem = compile_triton_func(
      avals_in, avals_out, kernel, num_warps, num_stages, metaparams
  )
  asm = Asm(asm_map)
  out_flat = triton_kernel_call_p.bind(
      *flat_args,
      kernel_name=kernel_name,
      call_name=call_name,
      asm=asm,
      shared_mem=shared_mem,
      out_shapes=tuple(flat_out_shapes),
      grid=grid,
      num_warps=num_warps,
      num_stages=num_stages,
      dump_binary_path=dump_binary_path,
      input_output_aliases=tuple(input_output_aliases.items()),
      **metaparams,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)


def cdiv(a, b):
  return triton.cdiv(a, b)


def strides_from_shape(shape: Tuple[int]) -> Tuple[int]:
  size = np.prod(shape)
  strides = []
  for s in shape:
    size = size // s
    strides.append(int(size))
  return tuple(strides)


def next_power_of_2(x: int) -> int:
  if x == 0:
    return 1
  return 2 ** math.ceil(math.log2(x))
