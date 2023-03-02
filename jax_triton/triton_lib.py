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
import jax.dlpack
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.lib import xla_client as xc
import jax.numpy as jnp
from jax_triton import triton_kernel_call_lib
from jax_triton import utils
import numpy as np

CAN_USE_TRITON = False
try:
  import triton
  import triton.compiler as tc
  import triton.language as tl
  from triton.runtime import autotuner
  import triton._C.libtriton.triton as _triton
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
  if isinstance(obj, (jax.core.ShapedArray, state.AbstractRef)):
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


def aval_size_bytes(aval):
  return np.dtype(aval.dtype).itemsize * aval.size


# Compiled kernels are kept alive by the kernel call which, in turn, are kept
# alive by the jitted JAX function.
_COMPILED_KERNEL_CACHE = weakref.WeakValueDictionary()


def compile_ttir(
    ttir,
    device: int = 0,
    num_warps: int = 4,
    num_stages: Optional[int] = None,
    dump: bool = False) -> Tuple[str, Dict[str, Any], int]:
  compute_capability = triton_kernel_call_lib.get_compute_capability(device)
  if num_stages is None:
    num_stages = 3 if compute_capability >= 75 else 2
  if dump:
    ttir.dump()
  try:
    ttir = tc.optimize_triton_ir(ttir)
    ttgir = tc.ttir_to_ttgir(ttir, num_warps)
    ttgir = tc.optimize_ttgir(ttgir, num_stages, compute_capability)
  except RuntimeError as e:
    ttir.dump()
    raise ValueError("TTIR->TTGIR pass failed!") from e
  extern_libs = {}
  try:
    llir = tc.ttgir_to_llir(ttgir, extern_libs, compute_capability)
  except RuntimeError as e:
    ttgir.dump()
    raise ValueError("TTIR->TTGIR pass failed!") from e
  if dump:
    ttgir.dump()
  shared_mem = _triton.get_shared_memory_size(ttgir)
  ptx = str(tc.llir_to_ptx(llir, compute_capability))
  name = tc.ptx_get_kernel_name(ptx)
  cubin = tc.ptx_to_cubin(ptx, compute_capability)
  asm = dict(ttir=ttir, ttgir=ttgir, llir=llir, ptx=ptx, cubin=cubin)
  return name, asm, shared_mem

def get_or_create_triton_kernel(
    fn,
    arg_dtypes,
    *,
    num_warps,
    num_stages,
    metaparams,
    dump: bool,
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
    device = 0
    ttir = tc.ast_to_ttir(fn, signature, specialization, constants)
    name, asm, shared_mem = compile_ttir(ttir, device=device, num_warps=num_warps,
                                         num_stages=num_stages, dump=dump)

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
    input_output_aliases,
    zeroed_outputs,
    debug,
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

  args = list(ctx.avals_in)
  arg_dtypes = list(map(get_triton_type, ctx.avals_in))
  # Buffer args are filled in at runtime.
  encoded_args = [None] * (len(array_args) + len(scalar_args) + len(out_shapes))
  for idx, dtype, v in scalar_args:
    args.insert(idx, v)
    arg_dtypes.insert(idx, dtype)
    encoded_args[idx] = triton_kernel_call_lib.encode_kernel_parameter(v, dtype)

  args.extend(ctx.avals_out)
  arg_dtypes.extend(map(get_triton_type, ctx.avals_out))
  named_args = dict(unsafe_zip(fn.arg_names, args))

  if isinstance(fn, autotuner.Autotuner):
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
        if config.pre_hook is not None:
          raise NotImplementedError("`pre_hook` is not supported")

        if all(config.kwargs.get(k, v) == v for k, v in metaparams.items()):
          pruned_configs.append(config)
      if prev_early_config_prune_fn is not None:
        pruned_configs = prev_early_config_prune_fn(pruned_configs, named_args)
      return pruned_configs

    fn.early_config_prune = prune_configs
    fn.nargs = named_args
    configs = fn.prune_configs(metaparams)
    fn = fn.fn
  else:
    configs = [triton.Config({}, num_warps=num_warps, num_stages=num_stages)]

  if isinstance(fn, autotuner.Heuristics):
    for config in configs:
      for name, heuristic in fn.values.items():
        all_args = {**named_args, **metaparams, **config.kwargs}
        config.kwargs[name] = heuristic(all_args)
    fn = fn.fn

  if not isinstance(fn, triton.JITFunction):
    raise ValueError(
        "`kernel` must be a Triton `JITFunction`, `Heuristics` or `Autotuner`."
    )

  # Cache auto-tuned calls with the same parameters, so the auto-tuning need
  # only be performed once.
  # FIXME(cjfj): Should the key include `grid` and `zeroed_outputs`?
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
      config_metaparams = {**metaparams, **config.kwargs}

      kernel = get_or_create_triton_kernel(
          fn,
          arg_dtypes,
          num_warps=config.num_warps,
          num_stages=config.num_stages,
          metaparams=config_metaparams,
          dump=debug,
      )
      grid = utils.normalize_grid(grid, config_metaparams)

      config_zeroed_outputs = zeroed_outputs
      if callable(zeroed_outputs):
        config_zeroed_outputs = config_zeroed_outputs(config_metaparams)

      zeroed_outputs_with_sizes = {
          i + len(ctx.avals_in): aval_size_bytes(ctx.avals_out[i])
          for i in config_zeroed_outputs
      }

      kernel_calls.append(
          triton_kernel_call_lib.TritonKernelCall(
              kernel,
              grid[0],
              grid[1],
              grid[2],
              encoded_args,
              zeroed_outputs_with_sizes,
          )
      )

    if len(kernel_calls) > 1:
      named_scalar_args = {fn.arg_names[i]: v for i, _, v in scalar_args}
      input_output_aliases_with_sizes = tuple(
          (input_idx, output_idx, aval_size_bytes(ctx.avals_in[input_idx]))
          for input_idx, output_idx in input_output_aliases
      )
      kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
          f"{fn.fn.__name__} ({call_name=}) {named_scalar_args}",
          [(call, str(config)) for call, config in zip(kernel_calls, configs)],
          input_output_aliases_with_sizes,
      )
    else:
      kernel_call = kernel_calls[0]

    _KERNEL_CALL_CACHE[cache_key] = kernel_call

  ctx.module_context.add_keepalive(kernel_call)

  output_operand_aliases = ir.ArrayAttr.get(
      [
          mhlo.OutputOperandAlias.get(
              output_tuple_indices=[output_idx],
              operand_index=input_idx,
              operand_tuple_indices=[],
          )
          for input_idx, output_idx in input_output_aliases
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
      operand_layouts=utils.avals_to_layouts(ctx.avals_in),
      result_layouts=utils.avals_to_layouts(ctx.avals_out),
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


def triton_call(
    *args: Union[jax.Array, bool, int, float],
    kernel: triton.JITFunction,
    out_shape: Union[ShapeDtype, Sequence[ShapeDtype]],
    grid: GridOrLambda,
    call_name: str = "triton_kernel_call",
    num_warps: int = 4,
    num_stages: int = 2,
    input_output_aliases: Optional[Dict[int, int]] = None,
    zeroed_outputs: Union[
        Sequence[int], Callable[[Dict[str, Any]], Sequence[int]]
    ] = (),
    debug: bool = False,
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
    zeroed_outputs: A sequence of indices, or a function returning a sequence of
      indices, for outputs that should be zeroed before the kernel is launched.
    num_warps: The number of warps used to execute the Triton kernel.
    num_stages: The number of stages emitted by the Triton compiler.
    debug: Prints out intermediate IRs if True for debugging purposes.
    **metaparams: Additional keyword arguments that will be provided to a `grid`
      (if it is a function) and to the Triton kernel as `constexpr` arguments.

  Returns:
    Outputs from the Triton kernel.
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
      input_output_aliases=tuple(input_output_aliases.items()),
      zeroed_outputs=zeroed_outputs,
      debug=debug,
      **metaparams,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)
