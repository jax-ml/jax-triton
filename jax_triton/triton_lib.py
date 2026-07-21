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
from functools import cached_property
import inspect
import os
import pprint
import shutil
import tempfile
import types
from typing import Any, Protocol, TypeVar
import zlib

import jax
from jax import tree_util
from jax._src import core
from jax._src import state
from jax._src import util
from jax._src.frozen_dict import FrozenDict
from jax._src.interpreters import partial_eval as pe
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

try:
  from jax._src.pallas.triton import gpu_info  # pyrefly: ignore[missing-module-attribute]
except ImportError:
  gpu_info = None  # Only available in JAX 0.11.0+.

try:
  from jax._src.lib import gpu_triton as triton_kernel_call_lib
except ImportError:
  # GPU support is not available.
  triton_kernel_call_lib: Any = None


class _Stub:
  ...


try:
  import triton.backends.nvidia.compiler as cb
except ImportError:
  # NVIDIA backend is not available.
  cb: Any = types.SimpleNamespace(
      CUDAOptions=_Stub, CUDABackend=_Stub, GPUTarget=_Stub
  )

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

_HSACO_TMPDIR = tempfile.TemporaryDirectory(delete=True)

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

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
JITFunction = triton.JITFunction | gl_runtime.GluonJITFunction


StaticScalar = bool | int | float | np.float32

Grid = int | tuple[int] | tuple[int, int] | tuple[int, int, int]

_T = TypeVar("_T")
ValueOrFn = _T | Callable[[Mapping[str, Any]], _T]


def normalize_grid(grid: ValueOrFn[Grid], metaparams) -> tuple[int, int, int]:
  if callable(grid):
    grid = grid(metaparams)
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))  # pyrefly: ignore[bad-return]


def get_type_id(obj: Any) -> str:
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
    return "fp64"
  if isinstance(obj, np.float32):
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


triton_kernel_call_p = jex.core.Primitive("triton_kernel_call")
triton_kernel_call_p.multiple_results = True
triton_kernel_call_p.def_impl(
    functools.partial(xla.apply_primitive, triton_kernel_call_p)
)


@triton_kernel_call_p.def_abstract_eval
def triton_kernel_call_abstract_eval(*_, out_shapes, **__):
  return [
      core.ShapedArray(out_shape.shape, out_shape.dtype)
      for out_shape in out_shapes
  ]


def _triton_kernel_call_dce_rule(
    used_outs: list[bool], eqn: core.JaxprEqn
) -> tuple[list[bool], core.JaxprEqn | None]:
  return [True] * len(eqn.invars), eqn


pe.dce_rules[triton_kernel_call_p] = _triton_kernel_call_dce_rule


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
  return CompilationResult(
      binary=ptx,
      name=name,
      shared_mem_bytes=shared_mem_bytes,
      ttgir=str(ttgir),
      llir=str(llir),
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
      arg_dtypes: list[str],
      in_tree: tree_util.PyTreeDef,
      objpaths: list[tuple[int, ...]],
      scalar_args: tuple[tuple[int, str, Any], ...],
      metaparams: Mapping[str, Any],
      backend: tc.BaseBackend,
  ) -> KernelSpecialization:
    # Build the signature dict, restoring nested structure from ``in_tree``.
    n = in_tree.num_leaves
    nested_dtypes = in_tree.unflatten(arg_dtypes[:n])
    values = list(nested_dtypes) + arg_dtypes[n:]
    # ``arg_names`` includes metaparams, so we slice them out here.
    signature = dict(zip(arg_names[:len(values)], values))

    # TODO(sharadmv,zhangqiaorjc): handle differently aligned pointers
    # We assume that all arrays are aligned to 16 bytes, and Triton may use this
    # assumption, unless array args are include in the `do_not_specialize` list.
    alignments = [16] * len(arg_dtypes)
    for i, _, _ in scalar_args:
      alignments[i] = 0
    specialize_impl = _triton.native_specialize_impl
    is_const = False
    do_specialize = True
    specialization = [
        specialize_impl(
            backend,
            types.SimpleNamespace(
                data_ptr=lambda a=alignment: a,
                dtype=arg_dtype.removeprefix("*"),
            ),
            is_const,
            do_specialize,
            alignment > 0,
        )
        for arg_dtype, alignment in zip(arg_dtypes, alignments)
    ]

    attrs: dict[tuple[int, ...], Any] = {
        objpaths[i]: backend.parse_attr(attr)
        for i, (_, attr) in enumerate(specialization)
    }

    constants = dict(metaparams)
    constants.update({
        arg_names[objpaths[i][0]]: 1
        for i, _, v in scalar_args
        if v == 1 and len(objpaths[i]) == 1
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

  @property
  def name(self) -> str:
    """Name of the underlying kernel function."""
    return self.fn.fn.__name__

  @cached_property
  def arg_names(self) -> list[str]:
    """Returns a list of the kernel parameter names in the order they are declared in
    the kernel's signature."""
    # JITFunction::arg_names is deprecated, per the deprecation notice
    return (
      self.fn.arg_names
      if hasattr(self.fn, "arg_names")
      else [p.name for p in self.fn.params]
    )

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

  @cached_property
  def constexpr_param_names(self) -> frozenset[str]:
    """Names of parameters annotated with ``tl.constexpr`` / ``gl.constexpr``."""
    return frozenset(p.name for p in self.params if p.is_constexpr)

  @cached_property
  def param_defaults(self) -> dict[str, Any]:
    """Declared default values for kernel parameters."""
    return {p.name: p.default for p in self.params if p.has_default}

  @property
  def signature(self) -> inspect.Signature:
    return self.fn.signature

  @property
  def compiled_kernels_cache_size(self) -> int:
    return len(self.fn._jT_kernel_cache) if hasattr(self.fn, "_jT_kernel_cache") else 0

  def get_or_create_triton_kernel(
      self,
      make_target_func,
      platform,
      arg_dtypes,
      scalar_args,
      *,
      in_tree: tree_util.PyTreeDef,
      objpaths: list[tuple[int, ...]],
      compute_capability,
      backend_options: Mapping[str, Any],
      metaparams,
  ) -> tuple[triton_kernel_call_lib.TritonKernel, Any]:
    fn = self.fn
    num_warps = backend_options["num_warps"]
    num_ctas = backend_options["num_ctas"]

    gpu_target = make_target_func(compute_capability, num_ctas)
    backend = triton.compiler.make_backend(gpu_target)
    assert isinstance(backend, (cb.CUDABackend, hb.HIPBackend))

    spec = KernelSpecialization.build(
        self.arg_names,
        arg_dtypes,
        in_tree,
        objpaths,
        scalar_args,
        metaparams,
        backend,
    )

    # Cache key should contain any parameter that can affect the compiler output.
    cache_key = (
        fn,
        tuple(spec.signature.items()),
        tuple(spec.specialization),
        tuple(spec.constants.items()),
        gpu_target,
        tuple(sorted(backend_options.items())),
    )
    if not hasattr(self.fn, "_jT_kernel_cache"):
      # TODO(cjfj): Convert to LRU cache?
      self.fn._jT_kernel_cache = {}  # pyrefly: ignore[missing-attribute]
    kernel = self.fn._jT_kernel_cache.get(cache_key)  # pyrefly: ignore[missing-attribute]

    if kernel is None:
      if len(self.signature.parameters) != len(spec.signature):
        raise TypeError(
            f"Number of parameters in the kernel '{fn}' signature"
            f" ({len(self.signature.parameters)}: {self.signature}) does not"
            f" match reconstructed signature ({len(spec.signature)}:"
            f" {spec.signature}). If the kernel was working on an older version"
            " of jax-triton and its triton_call() launcher uses"
            " `input_output_aliases` argument, note that implicit output"
            " arguments are no longer required for aliased arguments."
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
      ast_source = ast_source_cls(
          fn, spec.signature, spec.constants, spec.attrs
      )
      module = ast_source.make_ir(
          gpu_target, options, codegen_fns, backend.get_module_map(), context
      )
      ttir = str(module)

      compilation_result = compile_ttir_inplace(
          module, backend, options, gpu_target
      )

      if _JAX_TRITON_DUMP_DIR:
        _dump_kernel_artifacts(
            cache_key, options, ttir, compilation_result, platform
        )

      kernel = triton_kernel_call_lib.TritonKernel(
          compilation_result.name,
          num_warps,
          num_ctas,
          compilation_result.shared_mem_bytes,
          compilation_result.binary,
          ttir,
          gpu_target.arch if isinstance(gpu_target.arch, int) else 0,
      )

      self.fn._jT_kernel_cache[cache_key] = kernel  # pyrefly: ignore[missing-attribute]

    return kernel, spec.attrs


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
    scalar_args: tuple[tuple[int, str, Any], ...],
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
):
  if triton_kernel_call_lib is None:
    raise _missing_gpu_support_error()

  kernel_call_name = name
  args = list(ctx.avals_in)
  arg_dtypes = list(map(get_type_id, ctx.avals_in))
  for idx, dtype, v in scalar_args:
    args.insert(idx, v)
    arg_dtypes.insert(idx, dtype)
  # Extract only the output avals not referenced in the input_output_aliases mapping.
  strictly_out_avals = [
    aval
    for i, aval in enumerate(ctx.avals_out)
    if i not in input_output_aliases.values()
  ]
  args.extend(strictly_out_avals)
  arg_dtypes.extend(map(get_type_id, strictly_out_avals))

  triton_fn = TritonFunction(fn)

  # Fill in missing constexpr defaults before metaparams are used.
  metaparams: dict[str, Any] = dict(metaparams)
  for name in triton_fn.constexpr_param_names:
    if name not in metaparams and name in triton_fn.param_defaults:
      metaparams[name] = triton_fn.param_defaults[name]

  named_args = dict(unsafe_zip(triton_fn.arg_names, args))
  configs = triton_fn.make_configs(backend_options, metaparams, named_args)

  # output2input maps output index to the original user-facing input index,
  # which matches the position in the reconstructed args list.
  output2input = {v: k for k, v in input_output_aliases.items()}
  if len(output2input) != len(input_output_aliases):
    raise ValueError("input_output_aliases must be a bijection")

  # Translate input_output_aliases from user-facing, indexing the original
  # ``*args``, to indices into ``array_args``.
  input_output_aliases = FrozenDict({
      k - sum(i < k for i, _, _ in scalar_args): v
      for k, v in input_output_aliases.items()
  })

  outputs_offset = len(ctx.avals_in) + len(scalar_args)

  # Map flat indices to Triton ObjPath tuples.
  nested = in_tree.unflatten(range(in_tree.num_leaves))
  objpaths = [
      tuple(k.idx for k in kp)
      for kp, _ in tree_util.tree_leaves_with_path(nested)
  ]
  # Outputs are appended as flat params after the input tree.
  num_top = len(nested)
  for j in range(len(arg_dtypes) - in_tree.num_leaves):
    objpaths.append((num_top + j,))

  equal_to_1 = {
      i for i, _, v in scalar_args if v == 1 and len(objpaths[i]) == 1
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
        output2input.get(i, i + outputs_offset): aval_size_bytes(
            ctx.avals_out[i]
        )
        for i in sorted(config_zeroed_outputs)
    }

    config_backend_options = {
        **backend_options,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": config.num_ctas,
    }

    kernel, specialization_attr = triton_fn.get_or_create_triton_kernel(
        make_target_func,
        ctx.module_context.platforms[0],
        arg_dtypes,
        scalar_args,
        in_tree=in_tree,
        objpaths=objpaths,
        compute_capability=compute_capability,
        backend_options=config_backend_options,
        metaparams=config_metaparams,
    )

    kernel_params = []
    for i, (arg, dtype) in enumerate(zip(args, arg_dtypes)):
      if isinstance(arg, core.ShapedArray):
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
    named_scalar_args = {triton_fn.arg_names[i]: v for i, _, v in scalar_args}
    input_output_aliases_with_sizes = tuple(
        (input_idx, output_idx, aval_size_bytes(ctx.avals_in[input_idx]))
        for input_idx, output_idx in input_output_aliases.items()
    )
    kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
        f"{kernel_call_name} ({triton_fn.name}) {named_scalar_args}",
        [(call, str(config)) for call, config in zip(kernel_calls, configs)],
        input_output_aliases_with_sizes,
    )
  else:
    kernel_call = kernel_calls[0]

  call_proto = kernel_call.to_proto(kernel_call_name, serialized_metadata)

  rule = jax.ffi.ffi_lowering(
      "triton_kernel_call_ffi",
      api_version=4,
      operand_output_aliases=input_output_aliases,
      has_side_effect=has_side_effect,
  )
  return rule(ctx, *array_args, opaque=zlib.compress(call_proto))


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


def triton_call(
    *args: jax.Array | StaticScalar,
    kernel: Autotuner | Heuristics | JITFunction,
    out_shape: ShapeDtype | Sequence[ShapeDtype],
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
    has_side_effect: bool = False,
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
      static values should be annotated with `triton.language.constexpr` or
      `triton.experimental.gluon.language.constexpr`.
    out_shape: An object with ``shape`` and ``dtype`` attributes or a sequence
      of such objects. Pointers for each of the elements of ``out_shape`` will
      be passed into ``kernel`` following the inputs.
    grid: An integer, tuple of up to 3 integers, or a function that returns a
      tuple of up to 3 integers. When `grid` is an integer, `kernel` is
      invocated in `grid`-many parallel executions. When `grid` is a sequence of
      integers, `kernel` is launched in a `prod(grid)`-many parallel execution.
      When `grid` is a function, it is passed `**metaparams` and should return a
      tuple of up to 3 integers.
    name: A name for the kernel call.
    compute_capability: The GPU compute capability to compile for.
    input_output_aliases: A dictionary mapping input argument indices to output
      indices. Providing a mapping will alias the corresponding buffers. If
      ``*args`` contains nested tuples, the input indices correspond to the
      flattened arguments. Similarly, the output indices correspond to the
      flattened ``out_shape``.
    zeroed_outputs: A sequence of indices into the flattened ``out_shape``, or a
      function returning such a sequence, for outputs that should be zeroed
      before the kernel is launched. Note that this also supports zeroing
      input-output (i.e. aliased through ``input_output_aliases``) arguments
      that should be treated as outputs in this argument.
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
    has_side_effect: Whether the Triton kernel has side effects.
    **metaparams: ``constexpr`` arguments for the Triton kernel. Missing
      constexpr arguments are filled from the kernel's declared defaults.
      Also provided to ``grid`` and ``zeroed_outputs`` when either is a
      function.

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

  out_shape = tree_util.tree_map(
      lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_shape
  )
  flat_args, in_tree = tree_util.tree_flatten(args)
  flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)

  array_args = []
  scalar_args = []
  for i, arg in enumerate(flat_args):
    if isinstance(arg, (bool, int, float)):
      scalar_args.append((i, get_type_id(arg), arg))
    elif isinstance(arg, np.float32):
      scalar_args.append((i, get_type_id(arg), float(arg)))
    else:
      array_args.append(arg)

  if input_output_aliases is None:
    input_output_aliases = {}

  out_flat = triton_kernel_call_p.bind(
      *array_args,
      fn=kernel,
      scalar_args=tuple(scalar_args),
      name=name,
      in_tree=in_tree,
      out_shapes=tuple(flat_out_shapes),
      grid=grid,
      compute_capability=compute_capability,
      backend_options=FrozenDict(backend_options),
      input_output_aliases=FrozenDict(input_output_aliases),
      zeroed_outputs=zeroed_outputs,
      serialized_metadata=serialized_metadata,
      has_side_effect=has_side_effect,
      metaparams=FrozenDict(metaparams),
  )
  return tree_util.tree_unflatten(out_tree, out_flat)
