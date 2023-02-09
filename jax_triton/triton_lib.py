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
import weakref

from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple, Union

from absl import logging
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


_JAX_TO_TRITON_TYPE_MAP = {
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


def get_triton_type(obj: Any) -> str:
  if isinstance(obj, (jax.core.ShapedArray, state.ShapedArrayRef)):
    return f"*{_JAX_TO_TRITON_TYPE_MAP[obj.dtype]}"
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


Grid = Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
GridOrLambda = Union[Grid, Callable[[Dict[str, Any]], Grid]]

triton_kernel_call_p = jax.core.Primitive("triton_kernel_call")
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


# Compiled kernels are kept alive by the kernel call which, in turn, are kept
# alive by the jitted JAX function.
_COMPILED_KERNEL_CACHE = weakref.WeakValueDictionary()


def get_or_create_triton_kernel(
    fn,
    arg_dtypes,
    *,
    num_warps,
    num_stages,
    metaparams,
    dump_binary_path,
) -> triton_kernel_call_lib.TritonKernel:
  signature = dict(enumerate(arg_dtypes))

  constants = {fn.arg_names.index(k): v for k, v in metaparams.items()}
  # TODO(sharadmv,zhangqiaorjc): handle differently aligned pointers
  specialization = collections.namedtuple(
      "instance_descriptor", ["divisible_by_16", "equal_to_1"]
  )(tuple(range(len(arg_dtypes))), ())

  # Cache key should contain any parameter that can affect the compiler output.
  cache_key = (
      fn,
      tuple(arg_dtypes),
      tuple(metaparams.items()),
      specialization,
      num_warps,
      num_stages,
  )
  kernel = _COMPILED_KERNEL_CACHE.get(cache_key)

  if kernel is None:
    # TODO(sharadmv): handle multiple devices, right now we assume device 0
    # which is fine when we have multiple of the same GPU but this won't work in
    # general.
    asm, shared_mem, name = tc._compile(
        fn,
        signature=signature,
        device=0,
        specialization=specialization,
        constants=constants,
        num_warps=num_warps,
        num_stages=num_stages,
        extern_libs={},
        output="cubin",
    )

    if dump_binary_path is not None:
      with open(dump_binary_path, "wb") as fp:
        pickle.dump(dict(asm=asm, shared_mem=shared_mem, name=name), fp)

    kernel = triton_kernel_call_lib.TritonKernel(
        asm["cubin"], name, num_warps, shared_mem
    )
    _COMPILED_KERNEL_CACHE[cache_key] = kernel

  return kernel


_KERNEL_CALL_CACHE = weakref.WeakValueDictionary()


def triton_kernel_call_lowering(
    ctx,
    *array_args,
    fn,
    scalar_args,
    call_name,
    out_shapes,
    grid,
    num_warps,
    num_stages,
    dump_binary_path,
    input_output_aliases,
    **metaparams,
):
  if jaxlib.version.__version_info__ < (0, 3, 22) and input_output_aliases:
    raise NotImplementedError(
        "`input_output_aliases` only supported on `jaxlib>=0.3.22")

  out_type = ir.TupleType.get_tuple(
      [
          ir.RankedTensorType.get(
              shape.shape, mlir.dtype_to_ir_type(shape.dtype)
          )
          for shape in out_shapes
      ]
  )
  i32_type = ir.IntegerType.get_signless(32)

  arg_dtypes = list(map(get_triton_type, ctx.avals_in))
  # Buffer args are filled in at runtime.
  encoded_args = [None] * (len(array_args) + len(scalar_args) + len(out_shapes))
  for idx, dtype, v in scalar_args:
    arg_dtypes.insert(idx, dtype)
    encoded_args[idx] = triton_kernel_call_lib.encode_kernel_parameter(v, dtype)

  arg_dtypes.extend(map(get_triton_type, ctx.avals_out))

  if isinstance(fn, triton.runtime.autotuner.Autotuner):
    if any(idx not in fn.key_idx for idx, _, _ in scalar_args):
      logging.warning(
          "Auto-tuning key does not include all scalar arguments. "
          "We may perform redundant auto-tuning."
      )

    # If any metaparams have been specified explicitly, we prune any configs
    # that conflict. Note that this is more permissive than Triton's autotuner
    # implementation, which will throw an error if any keys match.
    # TODO(cjfj): Prune explicit `num_warps` / `num_stages`.
    prev_early_config_prune_fn = fn.early_config_prune

    def prune_configs(configs, named_args):
      pruned_configs = []
      for config in configs:
        if all(config.kwargs.get(k, v) == v for k, v in metaparams.items()):
          pruned_configs.append(config)
      if prev_early_config_prune_fn is not None:
        pruned_configs = prev_early_config_prune_fn(pruned_configs, named_args)
      return pruned_configs

    fn.early_config_prune = prune_configs
    named_scalar_args = {fn.arg_names[idx]: v for idx, _, v in scalar_args}
    fn.nargs = {name: named_scalar_args.get(name) for name in fn.arg_names}
    configs = fn.prune_configs(metaparams)
    fn = fn.fn
  elif isinstance(fn, triton.JITFunction):
    configs = [triton.Config({}, num_warps=num_warps, num_stages=num_stages)]
  else:
    raise ValueError("`kernel` must be a `JITFunction` or `Autotuner`.")

  # Cache auto-tuned calls with the same parameters, so the auto-tuning need
  # only be performed once.
  cache_key = (
      fn,
      tuple(configs),
      tuple(arg_dtypes),
      tuple(encoded_args),
      tuple(metaparams.items()),
  )
  kernel_call = _KERNEL_CALL_CACHE.get(cache_key)

  if kernel_call is None:
    kernel_calls = []
    for config in configs:
      config_metaparams = metaparams.copy()
      config_metaparams.update(config.kwargs)
      kernel = get_or_create_triton_kernel(
          fn,
          arg_dtypes,
          num_warps=config.num_warps,
          num_stages=config.num_stages,
          metaparams=config_metaparams,
          dump_binary_path=dump_binary_path,
      )
      grid = normalize_grid(grid, config_metaparams)
      kernel_calls.append(
          triton_kernel_call_lib.TritonKernelCall(
              kernel, grid[0], grid[1], grid[2], encoded_args
          )
      )

    if len(kernel_calls) > 1:
      kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
          f"{fn.fn.__name__} ({call_name=}) {named_scalar_args}",
          [(call, str(config)) for call, config in zip(kernel_calls, configs)],
      )
    else:
      kernel_call = kernel_calls[0]

    _KERNEL_CALL_CACHE[cache_key] = kernel_call

  ctx.module_context.add_keepalive(kernel_call)

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
      array_args,
      call_target_name=ir.StringAttr.get(call_name),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(kernel_call.descriptor),
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


class ShapeDtype(Protocol):

  @property
  def shape(self) -> Tuple[int, ...]:
    ...

  @property
  def dtype(self) -> np.dtype:
    ...


def normalize_grid(grid: GridOrLambda, metaparams) -> Tuple[int, int, int]:
  if callable(grid):
    grid = grid(metaparams)
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))


def triton_call(
    *args: Union[Array, bool, int, float],
    kernel: triton.JITFunction,
    out_shape: Union[ShapeDtype, Sequence[ShapeDtype]],
    grid: GridOrLambda,
    call_name: str = "triton_kernel_call",
    num_warps: int = 4,
    num_stages: int = 2,
    dump_binary_path: Optional[str] = None,
    input_output_aliases: Optional[Dict[int, int]] = None,
    **metaparams: Any,
) -> Any:
  """Calls a Triton kernel with `jax.Array` arguments.

  Example usage:

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

  Args:
    *args: Inputs for the Triton kernel.
    kernel: A Triton kernel (e.g. a function decorated with `triton.jit`). All
      static values should be annotated with `triton.language.constexpr`.
    out_shape: A `jax.ShapeDtypeStruct` (or something that has `.shape` and
      `.dtype` attributes) or a sequence thereof that specify the output(s) of
      the kernel. Pointers for each of the `jax.ShapeDtypeStruct`s in
      `out_shape` will be passed into `kernel` following the input parameters.
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

  Returns:
    Outputs from the Trion kernel.
  """
  if not CAN_USE_TRITON:
    raise ValueError(
        "`triton_call` is only available when `triton` is installed."
    )
  xc.register_custom_call_target(
      call_name, triton_kernel_call_lib.get_custom_call(), platform="CUDA"
  )
  out_shape = tree_util.tree_map(
      lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_shape)
  flat_args, _ = tree_util.tree_flatten(args)
  # TODO(sharadmv): check in_tree is flat (no Pytrees allowed in triton_call)
  flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)

  array_args = []
  scalar_args = []
  for i, arg in enumerate(flat_args):
    if isinstance(arg, (bool, int, float)):
      scalar_args.append((i, get_triton_type(arg), arg))
    else:
      array_args.append(arg)

  if input_output_aliases is None:
    input_output_aliases = {}

  out_flat = triton_kernel_call_p.bind(
      *array_args,
      fn=kernel,
      scalar_args=tuple(scalar_args),
      call_name=call_name,
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
