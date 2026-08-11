# Copyright 2026 The jax_triton Authors.
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

"""Module for calling Triton or Triton.Gluon kernels from JAX."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import functools
import inspect
import json
import os
import pprint
import shutil
import tempfile
import types
from typing import Any, Final, Protocol, TypeGuard, TypedDict
import warnings
import zlib

import jax
from jax import tree_util
from jax._src import config as jax_config
from jax._src import core
from jax._src import state
from jax._src import util
from jax._src.frozen_dict import FrozenDict
from jax._src.interpreters import partial_eval as pe
from jax._src.lib import version as jaxlib_version
from jax._src.state import discharge as state_discharge
import jax.extend as jex
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp
import numpy as np
import triton
import triton._C.libtriton as _triton
import triton.compiler.compiler as tc
import triton.experimental.gluon._runtime as gl_runtime
import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.tools import tensor_descriptor

try:
  from jax._src.pallas.triton import gpu_info  # pyrefly: ignore[missing-module-attribute]
except ImportError:
  gpu_info = None  # Only available in JAX 0.11.0+.

try:
  from jax._src.lib import gpu_triton as triton_kernel_call_lib
except ImportError:
  # GPU support is not available.
  triton_kernel_call_lib: Any = None


class _Stub(Any):
  ...


try:
  import triton.backends.nvidia.compiler as cb
  from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST as _TMA_DTYPE_DEVICE_TO_HOST
  from triton.backends.nvidia.driver import TMA_TF32 as _TMA_TF32
  from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor
except ImportError:
  # NVIDIA backend is not available.
  cb: Any = types.SimpleNamespace(
      CUDAOptions=_Stub, CUDABackend=_Stub, GPUTarget=_Stub
  )
  _TMA_DTYPE_DEVICE_TO_HOST = {}
  _TMA_TF32 = 0
  GluonTensorDescriptor = _Stub

try:
  import triton.backends.amd.compiler as hb
except ImportError:
  # AMD backend is not available.
  hb: Any = types.SimpleNamespace(
      HIPOptions=_Stub, HIPBackend=_Stub, GPUTarget=_Stub
  )

# TODO(slebedev): Investigate if this is necessary.
if "TRITON_CACHE_DIR" in os.environ:
  del os.environ["TRITON_CACHE_DIR"]
_JAX_TRITON_DUMP_DIR = os.environ.get("JAX_TRITON_DUMP_DIR")


class CostEstimate(TypedDict, total=False):
  flops: int
  bytes_accessed: int

CUSTOM_CALL_TARGET_NAME: Final[str] = "triton_kernel_call_ffi"


_HSACO_TMPDIR = tempfile.TemporaryDirectory(delete=True)

strict_zip = functools.partial(zip, strict=True)

# b/447434580: Exceeding this limit will cause Triton to emit a single trap
# instruction, which will cause the GPU to hang indefinitely. See
# triton/third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp;l=718
_TMEM_MAX_SIZE = 512

_JAX_TO_TRITON_TYPE_MAP = {
    jnp.dtype("bfloat16"): "bf16",
    jnp.dtype("float64"): "fp64",
    jnp.dtype("float32"): "fp32",
    jnp.dtype("float16"): "fp16",
    jnp.dtype("float8_e4m3fn"): "fp8e4nv",
    jnp.dtype("float8_e5m2"): "fp8e5",
    jnp.dtype("float8_e4m3fnuz"): "fp8e4b8",
    jnp.dtype("float8_e5m2fnuz"): "fp8e5b16",
    jnp.dtype("int64"): "i64",
    jnp.dtype("int32"): "i32",
    jnp.dtype("int16"): "i16",
    jnp.dtype("int8"): "i8",
    jnp.dtype("uint64"): "u64",
    jnp.dtype("uint32"): "u32",
    jnp.dtype("uint16"): "u16",
    jnp.dtype("uint8"): "u8",
    jnp.dtype("bool"): "i1",
}

Heuristics = triton.runtime.Heuristics
Autotuner = triton.runtime.Autotuner
TensorDescriptor = tensor_descriptor.TensorDescriptor
type JITFunction = triton.JITFunction | gl_runtime.GluonJITFunction
type StaticScalar = bool | int | float | np.float32
type Grid = int | tuple[int] | tuple[int, int] | tuple[int, int, int]
type ValueOrFn[T] = T | Callable[[Mapping[str, Any]], T]


def normalize_grid(grid: ValueOrFn[Grid], metaparams) -> tuple[int, int, int]:
  if callable(grid):
    grid = grid(metaparams)
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))  # pyrefly: ignore[bad-return]


def get_type_id(obj: Any) -> str:
  if is_tensor_descriptor(obj):
    elem = _JAX_TO_TRITON_TYPE_MAP[obj.base.dtype]
    block = ",".join(str(b) for b in obj.block_shape)
    if getattr(obj, "layout", None) is not None:
      return f"tensordesc<{elem}[{block}],{repr(obj.layout)}>"
    return f"tensordesc<{elem}[{block}]>"
  if isinstance(obj, (jax.core.ShapedArray, state.AbstractRef)):
    return f"*{_JAX_TO_TRITON_TYPE_MAP[obj.dtype]}"
  if isinstance(obj, (tl.constexpr, gl.constexpr)):
    obj = obj.value
  if isinstance(obj, bool):  # True == isinstance(True, int) !!!
    return "B"
  if isinstance(obj, int):
    if -(2**31) <= obj < 2**31:
      return "i32"
    elif 2**31 <= obj < 2**32:
      return "u32"
    elif -(2**63) <= obj < 2**63:
      return "i64"
    elif 2**63 <= obj < 2**64:
      return "u64"
    else:
      raise ValueError(f"integer overflow representing {obj}")
  if isinstance(obj, float):
    fi = np.finfo(np.float32)
    abs_obj = abs(obj)
    if (
        np.isinf(obj)
        or np.isnan(obj)
        or abs_obj == 0.0
        or fi.tiny <= abs_obj <= fi.max
    ):
      return "fp32"
    else:
      return "fp64"
  if isinstance(obj, np.float32):
    warnings.warn(
        "Passing np.float32 scalars to triton_call is deprecated, use"
        " float instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return "fp32"
  if isinstance(obj, str):
    return "str"
  raise NotImplementedError(
      f"could not compute type name for {obj}: {type(obj)}"
  )


def to_python_type(arg: Any) -> Any:
  """Typecasts a scalar to a native Python type."""
  # Note that typecasting ints/floats to their respective types also converts JAX's
  # subclasses like TypedInt and TypedFloat, which choke nanobind's type caster in
  # strict mode.
  if isinstance(arg, (bool, np.bool_)):
    arg = bool(arg)
  elif isinstance(arg, (int, np.integer)):
    arg = int(arg)
  elif isinstance(arg, (float, np.floating)):
    arg = float(arg)
  # else return as-is and let it possibly, but not necessarily, fail (constexprs and
  # strings pass through, and the rest isn't expected here, so saving cycles on that)
  return arg


def _tensor_descriptor_flatten(desc):
  children = (desc.base,)
  aux = (
      tuple(desc.shape),
      tuple(desc.strides),
      tuple(desc.block_shape),
      getattr(desc, "layout", None),
      desc.padding,
      desc.round_f32_to_tf32,
  )
  return children, aux


def _tensor_descriptor_unflatten(cls, aux, children):
  (base,) = children
  shape, strides, block_shape, layout, padding, round_f32_to_tf32 = aux
  # ``object.__new__`` bypasses ``__post_init__`` validation.
  obj = object.__new__(cls)
  obj.base = base
  obj.shape = list(shape)
  obj.strides = list(strides)
  obj.block_shape = list(block_shape)
  obj.padding = padding
  obj.round_f32_to_tf32 = round_f32_to_tf32
  obj.layout = layout
  return obj


tree_util.register_pytree_node(
    TensorDescriptor,
    _tensor_descriptor_flatten,
    functools.partial(_tensor_descriptor_unflatten, TensorDescriptor),
)
tree_util.register_pytree_node(
    GluonTensorDescriptor,
    _tensor_descriptor_flatten,
    functools.partial(_tensor_descriptor_unflatten, GluonTensorDescriptor),
)


def is_tensor_descriptor(
    x: object,
) -> TypeGuard[TensorDescriptor | GluonTensorDescriptor]:
  return isinstance(x, (TensorDescriptor, GluonTensorDescriptor))


triton_kernel_call_p = jex.core.Primitive("triton_kernel_call")
triton_kernel_call_p.multiple_results = True


def _is_ref(x: Any) -> bool:
  try:
    return isinstance(core.typeof(x), state.AbstractRef)
  except TypeError:
    return False


def _triton_kernel_call_impl(*args, **params):
  if any(_is_ref(a) for a in args):
    # Ensure the jit is enabled to trigger the discharge.
    with jax_config.disable_jit(False):
      return jax.jit(functools.partial(triton_kernel_call_p.bind, **params))(
          *args
      )
  return xla.apply_primitive(triton_kernel_call_p, *args, **params)


triton_kernel_call_p.def_impl(_triton_kernel_call_impl)


@triton_kernel_call_p.def_effectful_abstract_eval
def triton_kernel_call_abstract_eval(*in_avals, out_shapes, **__):
  # We emit a read and write effect for each ref input, as there is no easy
  # way to tell whether a ref is read from/written to in the kernel.
  effects = {
      effect(i)
      for i, aval in enumerate(in_avals)
      if isinstance(aval, state.AbstractRef)
      for effect in (state.ReadEffect, state.WriteEffect)
  }
  return [core.ShapedArray(s.shape, s.dtype) for s in out_shapes], effects


def _triton_kernel_call_dce_rule(
    used_outs: list[bool], eqn: core.JaxprEqn
) -> tuple[list[bool], core.JaxprEqn | None]:
  return [True] * len(eqn.invars), eqn


pe.dce_rules[triton_kernel_call_p] = _triton_kernel_call_dce_rule


def _triton_kernel_call_discharge_impl(
    in_avals,
    out_avals,
    args,
    in_tree,
    out_shapes,
    input_output_aliases,
    **params,
):
  if input_output_aliases:
    raise NotImplementedError(
        "input_output_aliases cannot be combined with ref arguments"
    )

  ref_indices = []
  arr_idx = op_idx = -1
  for _, leaf in tree_util.tree_leaves_with_path(
      in_tree.unflatten(in_avals),
      is_leaf=lambda a: isinstance(a, (_StaticArg, _OutputPlaceholder)),
  ):
    if isinstance(leaf, _OutputPlaceholder):
      continue
    op_idx += 1
    if isinstance(leaf, _StaticArg):
      continue
    arr_idx += 1
    if isinstance(leaf, state.AbstractRef):
      ref_indices.append((arr_idx, op_idx))

  # Turn each ref into an input-output alias: append an extra output and alias
  # it to the corresponding operand.
  num_out_orig = len(out_shapes)
  new_out_shapes = list(out_shapes)
  new_aliases: dict[int, int] = {}
  for out_idx, (arr_idx, op_idx) in enumerate(ref_indices):
    new_out_shapes.append(
        jax.ShapeDtypeStruct.like(in_avals[arr_idx].inner_aval)
    )
    new_aliases[op_idx] = num_out_orig + out_idx
  res = triton_kernel_call_p.bind(
      *args,
      in_tree=in_tree,
      out_shapes=tuple(new_out_shapes),
      input_output_aliases=FrozenDict(new_aliases),
      **params,
  )
  ans, updated_refs = util.split_list(res, [num_out_orig])
  new_invals: list[Any] = [None] * len(in_avals)
  for out_idx, (arr_idx, _) in enumerate(ref_indices):
    new_invals[arr_idx] = updated_refs[out_idx]
  return new_invals, ans


# TODO(slebedev): Remove once the minimum JAX version we support is 0.12.0.
if hasattr(state_discharge, "DischargeContext"):
  def _triton_kernel_call_discharge_rule(
      ctx: state_discharge.DischargeContext, *args, **params,
  ):
    return _triton_kernel_call_discharge_impl(
        ctx.in_avals, ctx.out_avals, args, **params
    )
else:
  def _triton_kernel_call_discharge_rule(in_avals, out_avals, *args, **params):
    return _triton_kernel_call_discharge_impl(
        in_avals, out_avals, args, **params
    )

state_discharge.register_discharge_rule(triton_kernel_call_p)(
    _triton_kernel_call_discharge_rule
)


def aval_size_bytes(aval):
  return np.dtype(aval.dtype).itemsize * aval.size


def make_cuda_target(
    compute_capability: int | None, num_ctas: int
) -> cb.GPUTarget:
  # TODO(sharadmv): handle multiple devices, right now we assume device 0
  # which is fine when we have multiple of the same GPU but this won't work in
  # general. See also how Triton did this in JITFunction's
  # `self.device_caches = defaultdict(self.create_binder)` -- it spawns a new set of
  # precomputes for each new device with `x,y,.. = self.device_caches[device]` using the
  # create_binder() factory function.
  device = 0
  if compute_capability is None:
    try:
      compute_capability = triton_kernel_call_lib.get_compute_capability(device)
    except RuntimeError:
      if gpu_info is None:
        raise
      # TODO(slebedev): Consider *only* using ``gpu_info`` here.
      compute_capability = gpu_info.get_gpu_info().compute_capability
  if num_ctas > 1 and compute_capability < 90:
    raise ValueError("num_ctas > 1 unsupported before Hopper.")
  return cb.GPUTarget("cuda", compute_capability, 32)


_IS_HIPBackend_PATCHED = False
def _patch_hip_backend():
  """
  This defuses a bomb planted into Triton's AMD-specific compilation path by
  https://github.com/triton-lang/triton/commit/37ff43c5efd6e1b84c00a599ba070a501181e832#diff-33c9a103282c05c9d9d213b94450ae7481b6db8c3c6d810f54f175b4735a3c72
  In short: there's an unconditional and totally unnecessary "import torch" directive crashing
  the code when torch isn't installed.

  Remove the patch once triton wheel package version is pinned to >= triton version with the fix.
  """
  global _IS_HIPBackend_PATCHED
  if _IS_HIPBackend_PATCHED:
    return
  _IS_HIPBackend_PATCHED = True

  if not hasattr(hb.HIPBackend, "is_within_2gb"):
    return
  try:
    hb.HIPBackend.is_within_2gb(1)
    # if we're here, either the torch is installed, or the code was fixed
  except ImportError:
    # redefining poisoned implementation. At this point, it's super unlikely a user
    # would update python package discovery paths before the real call to is_within_2gb() to make
    # `import torch` succeed, so we could assume there's just no torch in the redefinition.
    def fixed_is_within_2gb(arg):
      MAX_INT_32 = 2**31 - 1
      if hasattr(arg, "ptr_range"):
        return arg.ptr_range() <= MAX_INT_32
      return False

    hb.HIPBackend.is_within_2gb = fixed_is_within_2gb


def make_hip_target(
    compute_capability: int | None, num_ctas: int
) -> hb.GPUTarget:
  del compute_capability, num_ctas
  # TODO(Arech): remove _patch_hip_backend() once Triton releases a fix
  _patch_hip_backend()
  device = 0
  arch = triton_kernel_call_lib.get_arch_details(device)
  arch = arch.split(":")[0]
  return hb.GPUTarget("hip", arch, 64)


@dataclasses.dataclass
class CompilationResult:
  binary: str
  name: str
  shared_mem_bytes: int
  ttgir: str
  llir: str
  global_scratch_size: int | None = None
  global_scratch_align: int | None = None
  tensordesc_meta: Sequence[Any] | None = None


def compile_ttir_inplace(
    ttir,
    backend: cb.CUDABackend | hb.HIPBackend,
    options: cb.CUDAOptions | hb.HIPOptions,
    gpu_target: tc.GPUTarget,
) -> CompilationResult:
  if isinstance(backend, cb.CUDABackend):
    return compile_ttir_to_ptx_inplace(ttir, backend, options, gpu_target.arch)
  else:
    assert isinstance(backend, hb.HIPBackend)
    return compile_ttir_to_hsaco_inplace(ttir, backend, options)


def compile_ttir_to_ptx_inplace(
    ttir,
    cuda_backend: cb.CUDABackend,
    cuda_options: cb.CUDAOptions,
    compute_capability,
) -> CompilationResult:
  if cuda_options.debug:
    print(ttir)
  try:
    metadata = {}
    opt_ttir = cuda_backend.make_ttir(
        ttir, metadata, cuda_options, compute_capability
    )
    ttgir = cuda_backend.make_ttgir(
        opt_ttir,
        metadata,
        cuda_options,
        compute_capability,
    )
  except RuntimeError as e:
    ttir.dump()
    raise ValueError("TTIR->TTGIR pass failed!") from e
  if cuda_options.debug:
    print(ttgir)
  try:
    llir = cuda_backend.make_llir(
        ttgir,
        metadata,
        cuda_options,
        compute_capability,
    )
  except RuntimeError as e:
    ttgir.dump()
    raise ValueError("TTGIR->LLIR pass failed!") from e
  if metadata["tmem_size"] > _TMEM_MAX_SIZE:
    raise ValueError(
        f"TMEM size {metadata['tmem_size']} exceeds limit {_TMEM_MAX_SIZE}."
    )
  shared_mem_bytes = metadata["shared"]
  if cuda_options.debug:
    print(llir)
  ptx = cuda_backend.make_ptx(
      llir,
      metadata,
      cuda_options,
      compute_capability,
  )
  if cuda_options.debug:
    print(ptx)
  name = metadata["name"]
  global_scratch_size = metadata.get("global_scratch_size")
  if global_scratch_size == 0:
    global_scratch_size = None
  global_scratch_align = metadata.get("global_scratch_align")
  if global_scratch_align == 1:
    global_scratch_align = None
  return CompilationResult(
      binary=ptx,
      name=name,
      shared_mem_bytes=shared_mem_bytes,
      ttgir=str(ttgir),
      llir=str(llir),
      global_scratch_size=global_scratch_size,
      global_scratch_align=global_scratch_align,
      tensordesc_meta=metadata.get("tensordesc_meta"),
  )


def compile_ttir_to_hsaco_inplace(
    ttir,
    hip_backend: hb.HIPBackend,
    hip_options: hb.HIPOptions,
) -> CompilationResult:
  if hip_options.debug:
    print(ttir)
  try:
    metadata = {}
    opt_ttir = hip_backend.make_ttir(ttir, metadata, hip_options)
    ttgir = hip_backend.make_ttgir(opt_ttir, metadata, hip_options)
  except RuntimeError as e:
    ttir.dump()
    raise ValueError("TTIR->TTGIR pass failed!") from e
  if hip_options.debug:
    print(ttgir)
  try:
    llir = hip_backend.make_llir(ttgir, metadata, hip_options)
  except RuntimeError as e:
    ttgir.dump()
    raise ValueError("TTGIR->LLIR pass failed!") from e
  shared_mem_bytes = metadata["shared"]
  if hip_options.debug:
    print(llir)

  amdgcn = hip_backend.make_amdgcn(llir, metadata, hip_options)
  hsaco = hip_backend.make_hsaco(amdgcn, metadata, hip_options)

  name = metadata["name"]
  # Instead of passing hsaco which are "bytes", we first write
  # to a file and then pass the "string" path. This is needed because
  # nanobind doesn't automatically convert between bytes and string.
  # https://github.com/wjakob/nanobind/discussions/137
  fd, hsaco_path = tempfile.mkstemp(dir=_HSACO_TMPDIR.name)
  with os.fdopen(fd, "wb") as f:
    f.write(hsaco)
  return CompilationResult(
      binary=hsaco_path,
      name=name,
      shared_mem_bytes=shared_mem_bytes,
      ttgir=str(ttgir),
      llir=str(llir),
      global_scratch_size=metadata.get("global_scratch_size"),
      global_scratch_align=metadata.get("global_scratch_align"),
  )


@dataclasses.dataclass(frozen=True)
class KernelSpecialization:
  """Kernel specialization for a specific set of argument types and values.

  Attributes:
    signature: Argument names to dtype strings (e.g. ``"*fp32"``). Constants
      are mapped to ``"constexpr"``.
    specialization: Per-argument ``(type_string, specialization_key)`` tuples,
      used in the compilation cache key.
    attrs: Per-argument attributes (e.g. divisibility hints) keyed by argument
      index tuples.
    constants: Constant argument names to their values.
  """

  signature: dict[str, str]
  specialization: list[Any]
  attrs: dict[tuple[int, ...], Any]
  constants: dict[str, Any]

  @classmethod
  def build(
      cls,
      arg_names: list[str],
      args: list[Any],
      arg_dtypes: list[str],
      in_tree: tree_util.PyTreeDef,
      objpaths: list[tuple[int, ...]],
      metaparams: Mapping[str, Any],
      backend: tc.BaseBackend,
  ) -> KernelSpecialization:
    # Build the signature dict, restoring nested structure from ``in_tree``.
    signature = dict(strict_zip(arg_names, in_tree.unflatten(arg_dtypes)))

    # TODO(sharadmv,zhangqiaorjc): handle differently aligned pointers
    # We assume that all arrays are aligned to 16 bytes, and Triton may use this
    # assumption, unless array args are included in the `do_not_specialize` list.
    static_indices = {
        i
        for i, a in enumerate(args)
        if not (isinstance(a, core.AbstractValue) or is_tensor_descriptor(a))
    }
    alignments = [0 if i in static_indices else 16 for i in range(len(args))]
    specialize_impl = _triton.native_specialize_impl  # pyrefly: ignore[missing-attribute]
    is_const = False
    do_specialize = True
    specialization = [
        specialize_impl(
            backend,
            types.SimpleNamespace(
                data_ptr=lambda a=alignment: a,
                dtype=_JAX_TO_TRITON_TYPE_MAP[arg.base.dtype]
                if is_tensor_descriptor(arg)
                else arg_dtype.removeprefix("*"),
            ),
            is_const,
            do_specialize,
            alignment > 0,
        )
        for arg, arg_dtype, alignment in strict_zip(
            args, arg_dtypes, alignments
        )
    ]

    attrs: dict[tuple[int, ...], Any] = {
        objpaths[i]: backend.parse_attr(attr)
        for i, (_, attr) in enumerate(specialization)
    }

    constants = dict(metaparams)
    constants.update({
        arg_names[objpaths[i][0]]: 1
        for i in static_indices
        if args[i] == 1 and len(objpaths[i]) == 1
    })
    for constant in constants:
      signature[constant] = "constexpr"
    return cls(signature, specialization, attrs, constants)


def _dump_kernel_artifacts(
    cache_key: tuple[Any, ...],
    options: Any,
    ttir: str,
    compilation_result: CompilationResult,
    platform: str,
) -> None:
  base = f"{_JAX_TRITON_DUMP_DIR}/{abs(hash(cache_key))}"
  os.makedirs(base)
  kernel_name = compilation_result.name
  with open(f"{base}/config", "w") as f:
    pprint.pprint(cache_key, stream=f)
    pprint.pprint(options, stream=f)
  with open(f"{base}/{kernel_name}.ttir", "w") as f:
    f.write(ttir)
  if platform == "rocm":
    shutil.copy2(
        compilation_result.binary,
        f"{base}/{kernel_name}.hsaco",
    )
  else:
    with open(f"{base}/{kernel_name}.ptx", "w") as f:
      f.write(compilation_result.binary)
  with open(f"{base}/{kernel_name}.ttgir", "w") as f:
    f.write(compilation_result.ttgir)
  with open(f"{base}/{kernel_name}.llir", "w") as f:
    f.write(compilation_result.llir)
  with open(f"{base}/{kernel_name}.compile_info", "w") as f:
    f.write(
        f"{kernel_name}: shared_mem_bytes:"
        f" {compilation_result.shared_mem_bytes}\n"
    )


class TritonFunction:
  """A unified wrapper around a Triton kernel.

  The wrapper is responsible for abstracting away low-level Triton API access,
  kernel compilation and caching.
  """
  autotuner: Autotuner | None = None
  heuristics: Heuristics | None = None
  fn: JITFunction

  def __init__(self, fn: Autotuner | Heuristics | JITFunction):
    if isinstance(fn, Autotuner):
      self.autotuner = fn
      fn = fn.fn
    if isinstance(fn, Heuristics):
      self.heuristics = fn
      fn = fn.fn

    self.fn = fn

    # TODO(cjfj): Convert to LRU cache?
    if not hasattr(fn, "_jT_kernel_cache"):
      fn._jT_kernel_cache = {}  # pyrefly: ignore[missing-attribute]

  @property
  def name(self) -> str:
    """Name of the underlying kernel function."""
    return self.fn.fn.__name__

  def make_configs(
      self,
      backend_options: Mapping[str, Any],
      metaparams: Mapping[str, Any],
      named_args: Mapping[str, Any],
  ) -> list[triton.Config]:
    """Returns the list of Triton configs.

    Autotuner configs that conflict with user-provided metaparams are pruned at
    lowering time. Unlike Triton, which errors when a config key also appears
    in metaparams, we allow it as long as the values match.
    """
    if self.autotuner is not None:
      prev_early_config_prune_fn = self.autotuner.early_config_prune

      def prune_configs(configs, named_args, **conf_kwargs):
        pruned_configs = []
        for config in configs:
          if config.pre_hook is not None:
            raise NotImplementedError("`pre_hook` is not supported")
          if all(config.kwargs.get(k, v) == v for k, v in metaparams.items()):
            pruned_configs.append(config)
        if prev_early_config_prune_fn is not None:
          pruned_configs = prev_early_config_prune_fn(
              pruned_configs, named_args
          )
        return pruned_configs

      self.autotuner.early_config_prune = prune_configs
      self.autotuner.nargs = named_args  # pyrefly: ignore[bad-assignment]
      configs = self.autotuner.prune_configs(metaparams)  # pyrefly: ignore[bad-argument-type]
    else:
      configs = [
          triton.Config(
              {},
              num_warps=backend_options["num_warps"],
              num_stages=backend_options["num_stages"],
              num_ctas=backend_options["num_ctas"],
          )
      ]

    if self.heuristics is not None:
      for i, config in enumerate(configs):
        kwargs = config.kwargs.copy()
        for name, heuristic in self.heuristics.values.items():
          kwargs[name] = heuristic({**named_args, **metaparams, **kwargs})
        updated_config = copy.copy(config)
        updated_config.kwargs = kwargs
        configs[i] = updated_config

    return configs

  @property
  def params(self) -> list[triton.runtime.jit.KernelParam]:
    return self.fn.params

  @functools.cached_property
  def param_names(self) -> list[str]:
    """Kernel parameter names in declaration order."""
    # JITFunction::arg_names is deprecated, per the deprecation notice.
    return (
        self.fn.arg_names
        if hasattr(self.fn, "arg_names")
        else [p.name for p in self.params]
    )

  @functools.cached_property
  def constexpr_param_names(self) -> frozenset[str]:
    """Names of parameters annotated with ``constexpr``."""
    return frozenset(p.name for p in self.params if p.is_constexpr)

  @functools.cached_property
  def non_constexpr_param_names(self) -> list[str]:
    """Names of parameters not annotated with ``constexpr``."""
    return [p.name for p in self.params if not p.is_constexpr]

  @functools.cached_property
  def param_defaults(self) -> dict[str, Any]:
    """Declared default values for kernel parameters."""
    return {p.name: p.default for p in self.params if p.has_default}

  @property
  def signature(self) -> inspect.Signature:
    return self.fn.signature

  @property
  def _kernel_cache(self) -> dict[tuple[Any, ...], Any]:
    return self.fn._jT_kernel_cache  # pyrefly: ignore[missing-attribute]

  def _make_cache_key(
      self,
      spec: KernelSpecialization,
      gpu_target: tc.GPUTarget,
      backend_options: Mapping[str, Any],
  ) -> tuple[Any, ...]:
    """Builds the cache key from parameters that affect the compiler output."""
    return (
        self.fn,
        tuple(spec.signature.items()),
        tuple(spec.specialization),
        tuple(spec.constants.items()),
        gpu_target,
        FrozenDict(backend_options),
    )

  def _compile_kernel(
      self,
      spec: KernelSpecialization,
      gpu_target: tc.GPUTarget,
      backend: cb.CUDABackend | hb.HIPBackend,
      backend_options: Mapping[str, Any],
  ) -> tuple[triton_kernel_call_lib.TritonKernel, str, Any, CompilationResult]:
    """Compiles a Triton kernel from a specialization.

    Args:
      spec: The kernel specialization.
      gpu_target: The GPU target to compile for.
      backend: The Triton backend (CUDA or HIP).
      backend_options: Backend-specific compiler options.

    Returns:
      A ``(kernel, ttir, options, compilation_result)`` tuple.
    """
    fn = self.fn
    if len(self.signature.parameters) != len(spec.signature):
      raise TypeError(
          f"Number of parameters in the kernel '{fn}' signature"
          f" ({len(self.signature.parameters)}: {self.signature})"
          " does not match reconstructed signature"
          f" ({len(spec.signature)}: {spec.signature}). If the"
          " kernel was working on an older version of jax-triton"
          " and its triton_call() launcher uses"
          " `input_output_aliases` argument, note that implicit"
          " output arguments are no longer required for aliased"
          " arguments."
      )

    options = backend.parse_options(backend_options)  # pyrefly: ignore[bad-argument-type]

    context = _triton.ir.context()  # pyrefly: ignore[missing-attribute]
    _triton.ir.load_dialects(context)  # pyrefly: ignore[missing-attribute]
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)

    if isinstance(fn, gl_runtime.GluonJITFunction):
      ast_source_cls = gl_runtime.GluonASTSource
    else:
      ast_source_cls = tc.ASTSource
    ast_source = ast_source_cls(fn, spec.signature, spec.constants, spec.attrs)
    module = ast_source.make_ir(
        gpu_target,
        options,
        codegen_fns,
        backend.get_module_map(),
        context,
    )
    ttir = str(module)

    compilation_result = compile_ttir_inplace(
        module, backend, options, gpu_target
    )

    num_warps = backend_options["num_warps"]
    num_ctas = backend_options["num_ctas"]
    kernel_args = [
        compilation_result.name,
        num_warps,
        num_ctas,
        compilation_result.shared_mem_bytes,
        compilation_result.binary,
        ttir,
        gpu_target.arch if isinstance(gpu_target.arch, int) else 0,
    ]
    if jaxlib_version > (0, 11, 0):
      if compilation_result.global_scratch_size is not None:
        kernel_args += [
            compilation_result.global_scratch_size,
            compilation_result.global_scratch_align,
        ]
    elif compilation_result.global_scratch_size is not None:
      raise NotImplementedError(
          "The kernel requires an on-device global scratch buffer, which is "
          " only supported by jaxlib >0.11.0. Please upgrade."
      )
    kernel = triton_kernel_call_lib.TritonKernel(*kernel_args)
    return kernel, ttir, options, compilation_result

  def get_or_create_kernel(
      self,
      make_target_func,
      platform,
      args,
      arg_dtypes,
      *,
      in_tree: tree_util.PyTreeDef,
      objpaths: list[tuple[int, ...]],
      compute_capability,
      backend_options: Mapping[str, Any],
      metaparams,
  ) -> tuple[triton_kernel_call_lib.TritonKernel, Any, Any]:
    gpu_target = make_target_func(
        compute_capability, backend_options["num_ctas"]
    )
    backend = triton.compiler.make_backend(gpu_target)
    assert isinstance(backend, (cb.CUDABackend, hb.HIPBackend))

    spec = KernelSpecialization.build(
        self.non_constexpr_param_names,
        args,
        arg_dtypes,
        in_tree,
        objpaths,
        metaparams,
        backend,
    )

    cache_key = self._make_cache_key(spec, gpu_target, backend_options)
    cached = self._kernel_cache.get(cache_key)

    if cached is None:
      kernel, ttir, options, compilation_result = self._compile_kernel(
          spec, gpu_target, backend, backend_options
      )
      cached = (kernel, compilation_result.tensordesc_meta)
      self._kernel_cache[cache_key] = cached

      if _JAX_TRITON_DUMP_DIR:
        _dump_kernel_artifacts(
            cache_key,
            options,
            ttir,
            compilation_result,
            platform,
        )

    kernel, tensordesc_meta = cached
    return kernel, spec.attrs, tensordesc_meta


def _missing_gpu_support_error() -> Exception:
  return RuntimeError(
      "jax-triton requires JAX to be installed with GPU support. See "
      "https://jax.readthedocs.io/en/latest/installation.html."
  )


def triton_kernel_call_lowering(
    make_target_func,
    ctx,
    *array_args: mlir.Value,
    fn,
    name,
    in_tree: tree_util.PyTreeDef,
    out_shapes,
    grid,
    compute_capability,
    backend_options: FrozenDict[str, Any],
    input_output_aliases: FrozenDict[int, int],
    zeroed_outputs,
    serialized_metadata,
    metaparams: FrozenDict[str, Any],
    has_side_effect: bool = False,
    cost_estimate: CostEstimate | None = None,
):
  if triton_kernel_call_lib is None:
    raise _missing_gpu_support_error()

  kernel_call_name = name

  flat_with_path, full_tree = tree_util.tree_flatten_with_path(
      in_tree.unflatten(ctx.avals_in),
      is_leaf=lambda a: (
          isinstance(a, (_StaticArg, _OutputPlaceholder))
          or is_tensor_descriptor(a)
      ),
  )

  args: list[Any] = []
  objpaths: list[tuple[int, ...]] = []
  static_indices: set[int] = set()
  output_flat_idx: dict[int, int] = {}
  operand_flat_idx: list[int] = []
  for flat_idx, (path, leaf) in enumerate(flat_with_path):
    objpaths.append(tuple(k.idx for k in path))
    if isinstance(leaf, _OutputPlaceholder):
      args.append(ctx.avals_out[leaf.ordinal])
      output_flat_idx[leaf.ordinal] = flat_idx
      continue
    operand_flat_idx.append(flat_idx)
    if isinstance(leaf, _StaticArg):
      args.append(leaf.value)
      static_indices.add(flat_idx)
    else:
      args.append(leaf)  # operand array aval
  arg_dtypes = list(map(get_type_id, args))

  triton_fn = TritonFunction(fn)
  named_args = dict(
      strict_zip(
          triton_fn.non_constexpr_param_names, full_tree.unflatten(args)
      ),
      **metaparams,
  )
  configs = triton_fn.make_configs(backend_options, metaparams, named_args)

  io_aliases: dict[int, int] = {}
  for operand_idx, out_ordinal in input_output_aliases.items():
    output_flat_idx[out_ordinal] = operand_flat_idx[operand_idx]
    # Position of this operand among the traced (non-static) array args.
    array_idx = sum(
        fi not in static_indices for fi in operand_flat_idx[:operand_idx]
    )
    io_aliases[array_idx] = out_ordinal
  input_output_aliases = FrozenDict(io_aliases)

  equal_to_1 = {
      i for i in static_indices if args[i] == 1 and len(objpaths[i]) == 1
  }

  kernel_calls = []
  for config in configs:
    config_metaparams = {**metaparams, **config.kwargs}
    config_grid = normalize_grid(grid, config_metaparams)

    config_zeroed_outputs = zeroed_outputs
    if callable(zeroed_outputs):
      config_zeroed_outputs = config_zeroed_outputs(config_metaparams)

    # zeroed_params_with_sizes is a dict output_arg_idx -> aval_size_bytes
    # config_zeroed_outputs contains output ordinal indices
    zeroed_params_with_sizes = {
        output_flat_idx[i]: aval_size_bytes(ctx.avals_out[i])
        for i in sorted(config_zeroed_outputs)
    }

    config_backend_options = {
        **backend_options,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": config.num_ctas,
    }

    kernel, specialization_attr, tensordesc_meta = (
        triton_fn.get_or_create_kernel(
            make_target_func,
            ctx.module_context.platforms[0],
            args,
            arg_dtypes,
            in_tree=full_tree,
            objpaths=objpaths,
            compute_capability=compute_capability,
            backend_options=config_backend_options,
            metaparams=config_metaparams,
        )
    )

    kernel_params = []
    desc_idx = 0
    for i, (arg, dtype) in enumerate(strict_zip(args, arg_dtypes)):
      if is_tensor_descriptor(arg):
        assert tensordesc_meta and desc_idx < len(tensordesc_meta)
        meta = tensordesc_meta[desc_idx]
        desc_idx += 1
        if meta.get("is_im2col", False):
          raise NotImplementedError(
              "im2col TMA descriptors are not supported by jax-triton"
          )
        elem_type = _TMA_TF32 if arg.round_f32_to_tf32 else meta["elem_type"]
        shape = arg.shape
        expanded_shape = list(shape)
        if meta["fp4_padded"]:
          expanded_shape[-1] *= 2
        kernel_params.append(
            triton_kernel_call_lib.create_tma_descriptor_parameter(
                elem_type=_TMA_DTYPE_DEVICE_TO_HOST[elem_type],
                elem_size_bytes=meta["elem_size"],
                swizzle=meta["swizzle"],
                shape=[int(s) for s in expanded_shape],
                strides=[int(s) for s in arg.strides],
                block_shape=[int(b) for b in meta["block_size"]],
                oob_fill=1 if arg.padding == "nan" else 0,
            )
        )
        for dim in shape:
          kernel_params.append(
              triton_kernel_call_lib.create_scalar_parameter(int(dim), "i32")
          )
        for stride in arg.strides:
          kernel_params.append(
              triton_kernel_call_lib.create_scalar_parameter(int(stride), "i64")
          )
      elif isinstance(arg, core.ShapedArray):
        arg_attrs = specialization_attr[objpaths[i]]
        kernel_params.append(
            triton_kernel_call_lib.create_array_parameter(
                zeroed_params_with_sizes.get(i, 0),
                16 if (["tt.divisibility", 16] in arg_attrs) else 0,
            )
        )
      elif i not in equal_to_1:
        # Convert TypedInt/TypedFloat subclasses to plain Python types,
        # as nanobind's strict-mode integer caster rejects subclasses.
        arg = to_python_type(arg)
        kernel_params.append(
            triton_kernel_call_lib.create_scalar_parameter(arg, dtype)
        )

    kernel_calls.append(
        triton_kernel_call_lib.TritonKernelCall(
            kernel,
            config_grid[0],
            config_grid[1],
            config_grid[2],
            kernel_params,
        )
    )

  if len(kernel_calls) > 1:
    named_static_args = {}
    for i in static_indices:
      arg_name = triton_fn.non_constexpr_param_names[objpaths[i][0]]
      path_str = "".join(f"[{idx}]" for idx in objpaths[i][1:])
      named_static_args[f"{arg_name}{path_str}"] = args[i]
    input_output_aliases_with_sizes = tuple(
        (input_idx, output_idx, aval_size_bytes(ctx.avals_in[input_idx]))
        for input_idx, output_idx in input_output_aliases.items()
    )
    kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
        f"{kernel_call_name} ({triton_fn.name}) {named_static_args}",
        [
            (call, str(config))
            for call, config in strict_zip(kernel_calls, configs)
        ],
        input_output_aliases_with_sizes,
    )
  else:
    kernel_call = kernel_calls[0]

  call_proto = kernel_call.to_proto(kernel_call_name, serialized_metadata)

  rule = jax.ffi.ffi_lowering(
      CUSTOM_CALL_TARGET_NAME,
      api_version=4,
      operand_output_aliases=input_output_aliases,
      has_side_effect=has_side_effect,
  )
  kwargs: dict[str, Any] = {"opaque": zlib.compress(call_proto)}
  if cost_estimate is not None:
    kwargs["cost_estimate_json"] = json.dumps(dict(cost_estimate))
  return rule(ctx, *array_args, **kwargs)


mlir.register_lowering(
    triton_kernel_call_p,
    functools.partial(triton_kernel_call_lowering, make_cuda_target),
    platform="cuda",
)

mlir.register_lowering(
    triton_kernel_call_p,
    functools.partial(triton_kernel_call_lowering, make_hip_target),
    platform="rocm",
)


def triton_kernel_call_raise_on_jvp(*args, **kwargs):
  del args, kwargs  # unused
  raise NotImplementedError(
      "jax_triton.triton_call does not support automatic differentiation. Use "
      "jax.custom_jvp or jax.custom_vjp to implement a custom automatic "
      "differentiation rule for your kernel."
  )


ad.primitive_jvps[triton_kernel_call_p] = triton_kernel_call_raise_on_jvp


def triton_kernel_call_raise_on_vmap(*args, **kwargs):
  del args, kwargs  # unused
  raise NotImplementedError(
      "jax_triton.triton_call does not support batching with jax.vmap. Use "
      "jax.custom_batching.custom_vmap to implement a custom batching rule for "
      "your kernel."
  )


batching.primitive_batchers[triton_kernel_call_p] = (
    triton_kernel_call_raise_on_vmap
)


class ShapeDtype(Protocol):

  @property
  def shape(self) -> tuple[int, ...]:
    ...

  @property
  def dtype(self) -> np.dtype:
    ...


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _StaticArg:
  """Wraps a static kernel argument so it is baked into the ``PyTreeDef``."""

  value: Any = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def maybe_wrap(cls, x: Any) -> _StaticArg | Any:
    # TODO(slebedev): Drop np.ndarray once all callers are migrated.
    if isinstance(x, (jax.Array, np.ndarray)) or _is_ref(x):
      return x
    return cls(x)


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _OutputPlaceholder:
  """Placeholder for a kernel output, baked into the ``PyTreeDef``."""

  ordinal: int = dataclasses.field(metadata=dict(static=True))


def triton_call(
    *args: jax.Array | jax.Ref | StaticScalar,
    kernel: Autotuner | Heuristics | JITFunction,
    out_type: ShapeDtype | Sequence[ShapeDtype] | None = None,
    grid: ValueOrFn[Grid],
    name: str = "",
    num_warps: int | None = None,
    num_stages: int | None = None,
    # TODO(giorgioa): Add support for dimensions tuple.
    num_ctas: int | None = None,
    compute_capability: int | None = None,
    backend_options: Mapping[str, Any] | None = None,
    input_output_aliases: dict[int, int] | None = None,
    zeroed_outputs: ValueOrFn[Sequence[int]] = (),
    debug: bool = False,
    serialized_metadata: bytes = b"",
    cost_estimate: CostEstimate | None = None,
    has_side_effect: bool = False,
    **kwargs: Any,
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
      block_size: tl.constexpr = 128,
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
    block_size = 8
    return jt.triton_call(
        x,
        y,
        kernel=add_kernel,
        out_type=jax.typeof(x),
        grid=(x.size // block_size,),
        block_size=block_size)

  x_val = jnp.arange(8)
  y_val = jnp.arange(8, 16)
  print(add(x_val, y_val))
  print(jax.jit(add)(x_val, y_val))
  ```

  Args:
    *args: Positional operands for the Triton kernel. Array and ``Ref``
      arguments are passed as runtime buffers in their positional order before
      any ``out_type`` pointers. ``Ref`` arguments, created via ``jax.new_ref``
      are read-write buffers that the kernel mutates in-place. Unlike
      ``input_output_aliases``, they should not be included in ``out_type``.
      Non-array scalars are baked in as static (specialization) values, and any
      argument bound to a ``constexpr`` parameter becomes a metaparam.
    kernel: A Triton kernel (e.g. a function decorated with `triton.jit`). All
      static values should be annotated with `triton.language.constexpr` or
      `triton.experimental.gluon.language.constexpr`.
    out_type: An object with ``shape`` and ``dtype`` attributes or a sequence of
      such objects. Pointers for each of the elements of ``out_type`` will be
      passed into ``kernel`` following the inputs.
    grid: An integer, tuple of up to 3 integers, or a function that returns a
      tuple of up to 3 integers. When `grid` is an integer, `kernel` is invoked
      in `grid`-many parallel executions. When `grid` is a sequence of integers,
      `kernel` is launched in a `prod(grid)`-many parallel executions. When
      `grid` is a function, it is passed `**metaparams` and should return a
      tuple of up to 3 integers.
    name: A name for the kernel call.
    compute_capability: The GPU compute capability to compile for.
    input_output_aliases: Deprecated. A dictionary mapping input argument
      indices to output indices. Providing a mapping will alias the
      corresponding buffers. Input indices refer to the flattened
      non-``constexpr`` operands in kernel parameter declaration order (whether
      passed positionally or as keyword arguments). If operands contain nested
      tuples, the indices correspond to the flattened leaves. Output indices
      correspond to the flattened ``out_type``.
    zeroed_outputs: Deprecated. A sequence of indices into the flattened
      ``out_type``, or a function returning such a sequence, for outputs that
      should be zeroed before the kernel is launched. Note that this also
      supports zeroing input-output (i.e. aliased through
      ``input_output_aliases``) arguments that should be treated as outputs in
      this argument.
    num_warps: The number of warps used to execute the Triton kernel.
    num_stages: The number of stages emitted by the Triton compiler.
    num_ctas: The size of thread blocks per cluster to be used on GPUs with
      compute capabilities >= 9.0. It must be less or equal to 8.
    debug: Prints out intermediate IRs if True for debugging purposes.
    backend_options: A mapping of backend-specific compiler options. The
      available options depend on the Triton backend. The ``num_warps``,
      ``num_stages``, ``num_ctas`` and ``debug`` are merged into this mapping.
      It is an error to specify the same option in both.
    serialized_metadata: Arbitrary metadata that will be added into the
      serialized kernel call.
    cost_estimate: An estimate of the number of floating point operations
      ("flops") and memory bytes accessed ("bytes_accessed") by this kernel
      invocation. This is used by profiling tools to compute the performance
      metrics of this custom call (e.g. FLOPs/s and bandwidth).
    has_side_effect: Whether the Triton kernel has side effects.
    **kwargs: Keyword arguments for the Triton kernel. A keyword that names a
      non-``constexpr`` kernel parameter is treated as an operand and is subject
      to the same scalar-static/runtime buffer separation as positional
      ``*args``. All other keywords -- ``constexpr`` parameters and names that
      are not kernel parameters -- are treated as metaparams. Missing constexpr
      arguments are filled from the kernel's declared defaults. Metaparams are
      also provided to ``grid`` and ``zeroed_outputs`` when either is a
      function. A misspelled operand name silently becomes a metaparam rather
      than raising an error.

  Returns:
    Outputs from the Triton kernel.
  """
  if backend_options is None:
    backend_options = {}
  explicit_options = {
      "num_warps": num_warps,
      "num_stages": num_stages,
      "num_ctas": num_ctas,
      "debug": debug,
  }
  del num_ctas, num_stages, num_warps, debug
  for k, v in list(explicit_options.items()):
    if v is None:
      del explicit_options[k]

  if conflicts := explicit_options.keys() & backend_options.keys():
    raise ValueError(
        f"Cannot specify {conflicts} both as explicit arguments and in"
        " ``backend_options``"
    )
  backend_options = {**backend_options, **explicit_options}
  backend_options.setdefault("num_warps", 4)
  backend_options.setdefault("num_stages", 3)
  backend_options.setdefault("num_ctas", 1)
  for k, v in backend_options.items():
    try:
      hash(v)
    except TypeError:
      raise TypeError(
          f"backend_options[{k!r}] must be hashable, got {v!r}"
      ) from None

  if "out_shape" in kwargs:
    if out_type is not None:
      raise TypeError(
          "Cannot specify both out_type= and the deprecated out_shape="
      )
    warnings.warn(
        "out_shape= is deprecated in favor of out_type=",
        DeprecationWarning,
        stacklevel=2,
    )
    out_type = kwargs.pop("out_shape")
  elif out_type is None:
    raise TypeError(
        "Either out_type= or the deprecated out_shape= must be provided"
    )

  out_type = tree_util.tree_map(
      lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_type
  )

  triton_fn = TritonFunction(kernel)
  constexpr_names = triton_fn.constexpr_param_names
  param_names = frozenset(triton_fn.param_names)

  # Keywords naming a kernel parameter are bound to it; the rest are metaparams.
  metaparams: dict[str, Any] = {
      name: triton_fn.param_defaults[name]
      for name in constexpr_names
      if name in triton_fn.param_defaults
  }
  kernel_kwargs: dict[str, Any] = {}
  for kw_name, value in kwargs.items():
    if kw_name in param_names:
      kernel_kwargs[kw_name] = value
    else:
      metaparams[kw_name] = value

  bound = triton_fn.signature.bind_partial(*args, **kernel_kwargs)

  operands: dict[str, Any] = {}
  has_refs = False
  for p_name, value in bound.arguments.items():
    if p_name in constexpr_names:
      metaparams[p_name] = value
      continue
    leaves = tree_util.tree_leaves(
        value, is_leaf=lambda a: isinstance(a, state.TransformedRef)
    )
    if any(isinstance(leaf, state.TransformedRef) for leaf in leaves):
      raise NotImplementedError(
          "TransformedRefs are not supported as triton_call arguments"
      )
    has_refs = has_refs or any(map(_is_ref, leaves))
    operands[p_name] = tree_util.tree_map(_StaticArg.maybe_wrap, value)

  flat_out_shapes, out_tree = tree_util.tree_flatten(out_type)

  if input_output_aliases is None:
    input_output_aliases = {}
  elif input_output_aliases:
    warnings.warn(
        "input_output_aliases is deprecated. Pass inputs you want to alias as"
        " Ref arguments instead",
        DeprecationWarning,
        stacklevel=2,
    )

    if has_refs:
      raise ValueError(
          "input_output_aliases cannot be combined with Ref arguments. Refs"
          " express input/output aliasing implicitly"
      )

  if zeroed_outputs if callable(zeroed_outputs) else len(zeroed_outputs):
    warnings.warn(
        "zeroed_outputs is deprecated. Zero the buffer explicitly and pass"
        " it as a Ref argument instead",
        DeprecationWarning,
        stacklevel=2,
    )

  # Aliased outputs reuse their input buffer; unaliased outputs are matched
  # to the kernel parameters not supplied by the caller.
  aliased_out_ordinals = frozenset(input_output_aliases.values())
  if len(aliased_out_ordinals) != len(input_output_aliases):
    raise ValueError("input_output_aliases must be a bijection")
  n_missing_operands = len(triton_fn.non_constexpr_param_names) - len(operands)
  n_unaliased_outputs = len(flat_out_shapes) - len(aliased_out_ordinals)
  if n_missing_operands != n_unaliased_outputs:
    raise ValueError(
        f"out_type has {n_unaliased_outputs} unaliased outputs, but"
        f" {n_missing_operands} kernel parameters are missing an argument"
    )
  out_ordinals = iter(
      o for o in range(len(flat_out_shapes)) if o not in aliased_out_ordinals
  )
  array_args, in_tree = tree_util.tree_flatten([
      operands[n] if n in operands else _OutputPlaceholder(next(out_ordinals))
      for n in triton_fn.non_constexpr_param_names
  ])

  out_flat = triton_kernel_call_p.bind(
      *array_args,
      fn=kernel,
      name=name,
      in_tree=in_tree,
      out_shapes=tuple(flat_out_shapes),
      grid=grid,
      compute_capability=compute_capability,
      backend_options=FrozenDict(backend_options),
      input_output_aliases=FrozenDict(input_output_aliases),
      zeroed_outputs=zeroed_outputs,
      serialized_metadata=serialized_metadata,
      cost_estimate=FrozenDict(cost_estimate) if cost_estimate else None,
      has_side_effect=has_side_effect,
      metaparams=FrozenDict(metaparams),
  )
  return tree_util.tree_unflatten(out_tree, out_flat)
