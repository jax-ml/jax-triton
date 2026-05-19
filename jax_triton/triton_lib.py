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

from collections.abc import Callable, Sequence
import copy
import dataclasses
import functools
from functools import cached_property
import inspect
import itertools
import os
import pprint
import tempfile
from typing import Any, Protocol, Self, Union
import zlib

import jax
from jax import tree_util
from jax._src import core
from jax._src import util
from jax._src.lib import gpu_triton as triton_kernel_call_lib
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
import triton.runtime.autotuner as autotuner
import triton.runtime.jit as triton_runtime_jit

try:
  import triton.backends.nvidia.compiler as cb
except ImportError:
  cb = None  # NVIDIA backend is not available.

try:
  import triton.backends.amd.compiler as hb
except ImportError:
  hb = None  # AMD backend is not available.


if "TRITON_CACHE_DIR" in os.environ:
  del os.environ["TRITON_CACHE_DIR"]

_JAX_TRITON_DUMP_DIR = os.environ.get("JAX_TRITON_DUMP_DIR")
map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

Grid = Union[int, tuple[int], tuple[int, int], tuple[int, int, int]]
GridOrLambda = Union[Grid, Callable[[dict[str, Any]], Grid]]

# A coordinate of an element in the kernel parameter coordinate space. The first int
# indexes the kernel parameter by its declaration order in the signature; the following
# ints (if any) index into nested tuples within the actual argument. This is therefore a
# kernel-call attribute, since it depends on actual argument values, unlike kernel
# parameters themselves.
# Triton calls a coordinate a path, so we follow that convention.
CanonicalKernelArgPath = tuple[int, ...]
KernelArgPath = tuple[int | str, *tuple[int, ...]]
InOutSpec = int | str | KernelArgPath | tuple[int | str | KernelArgPath, ...]

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


def normalize_grid(grid: GridOrLambda, metaparams) -> tuple[int, int, int]:
  if callable(grid):
    grid = grid(metaparams)
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))


# Type handling is slightly messy here. Triton uses exact dtypes for arrays, but for
# scalars (whether constexpr or runtime) it accepts native Python objects only. The
# actual type of a kernel parameter is determined solely from the parameter type
# annotation. To feed the specialization engine correctly, we must typecast all scalars
# to Python types. Caveats:
# 1. We must not typecast constexprs, as they are a separate type that influences
# specialization.
# 2. Triton's runtime specialization for scalars unconditionally treats all floats as
# fp32. It is unclear whether this is a bug — the relevant code does not even have a
# string to describe fp64. Triton's kernel launcher likely does not support doubles,
# but JAX's `create_scalar_parameter()` supports them perfectly well. However, Triton
# still produces binaries that expect floats only, so there is no point in handling
# this differently.
#
# Fortunately, both Triton and JAX use the same (likely LLVM/MLIR-based) type
# identifiers, so the function below serves both goals: Triton's specialization and
# JAX's launcher.
def get_type_id(obj: Any) -> str:
  """Returns a Triton/JAX type identifier for the given object. Intended for use with
  runtime values only. Constexprs (strings are also constexprs in Triton) don't need
  type info."""
  if hasattr(obj, "dtype"):
    return _JAX_TO_TRITON_TYPE_MAP[obj.dtype]

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
    return "fp32"  # Triton unconditionally treats all floats as fp32, see above
  raise NotImplementedError(f"could not compute type name for {obj}: {type(obj)}")


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


def avals_to_layouts(avals):
  return [list(reversed(range(aval.ndim))) for aval in avals]


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


def aval_size_bytes(aval):
  return np.dtype(aval.dtype).itemsize * aval.size


def make_gpu_target_cuda(device, compute_capability):
  return cb.GPUTarget("cuda", compute_capability, 32)


_IS_HIPBackend_PATCHED = False
def _patch_hip_backend():
  """
  This fixes unconditional and totally unnecessary "import torch" in Triton's
  AMD-specific compilation path added in
  https://github.com/triton-lang/triton/commit/37ff43c5efd6e1b84c00a599ba070a501181e832#diff-33c9a103282c05c9d9d213b94450ae7481b6db8c3c6d810f54f175b4735a3c72

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


def make_gpu_target_hip(device, compute_capability):
  # TODO(Arech): remove _patch_hip_backend() once Triton releases a fix
  _patch_hip_backend()

  arch = triton_kernel_call_lib.get_arch_details(device)
  arch = arch.split(":")[0]
  return hb.GPUTarget("hip", arch, 64)


@dataclasses.dataclass
class CompilationResult:
  binary: str
  name: str
  shared_mem_bytes: int
  ttgir: str | None
  llir: str | None


def compile_ttir_inplace(
    ttir,
    backend: [cb.CUDABackend | hb.HIPBackend],
    options: [cb.CUDAOptions | hb.HIPOptions],
    compute_capability,
    platform,
):
  if platform == "cuda":
    return compile_ttir_to_ptx_inplace(
        ttir,
        backend,
        options,
        compute_capability,
    )

  elif platform == "rocm":
    return compile_ttir_to_hsaco_inplace(
        ttir,
        backend,
        options,
        compute_capability,
    )
  else:
    raise ValueError("Unsupported device.")


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
  ttgir = str(ttgir) if _JAX_TRITON_DUMP_DIR else None
  llir = str(llir) if _JAX_TRITON_DUMP_DIR else None
  return CompilationResult(
      binary=ptx,
      name=name,
      shared_mem_bytes=shared_mem_bytes,
      ttgir=ttgir,
      llir=llir,
  )


def compile_ttir_to_hsaco_inplace(
    ttir,
    hip_backend: hb.HIPBackend,
    hip_options: hb.HIPOptions,
    compute_capability,
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
  ttgir = str(ttgir) if _JAX_TRITON_DUMP_DIR else None
  llir = str(llir) if _JAX_TRITON_DUMP_DIR else None
  # Instead of passing hsaco which are "bytes", we first write
  # to a file and then pass the "string" path. This is needed because
  # nanobind doesn't automatically convert between bytes and string.
  # https://github.com/wjakob/nanobind/discussions/137
  fd, hsaco_path = tempfile.mkstemp()
  with os.fdopen(fd, "wb") as f:
    f.write(hsaco)
  return CompilationResult(
      binary=hsaco_path,
      name=name,
      shared_mem_bytes=shared_mem_bytes,
      ttgir=ttgir,
      llir=llir,
  )


def make_backend(
  make_gpu_target_func, compute_capability: int | None, num_ctas: int
) -> tuple[tc.BaseBackend, tc.GPUTarget, int]:
  """Resolves compute_capability and creates Triton's Backend and GPUTarget objects."""

  # TODO(sharadmv): handle multiple devices, right now we assume device 0
  # which is fine when we have multiple of the same GPU but this won't work in
  # general. See also how Triton did this in JITFunction's
  # `self.device_caches = defaultdict(self.create_binder)` -- it spawns a new set of
  # precomputes for each new device with `x,y,.. = self.device_caches[device]` using the
  # create_binder() factory function.
  device = 0
  if compute_capability is None:
    compute_capability = triton_kernel_call_lib.get_compute_capability(device)
  if num_ctas > 1 and compute_capability < 90:
    raise ValueError("num_ctas > 1 unsupported before Hopper.")

  gpu_target = make_gpu_target_func(device, compute_capability)
  backend = triton.compiler.make_backend(gpu_target)

  return backend, gpu_target, compute_capability


_BACKEND_OPTIONS_FIELD_NAMES = {
  "cuda": frozenset(cb.CUDAOptions.__dataclass_fields__.keys()),
  "hip": frozenset(hb.HIPOptions.__dataclass_fields__.keys()),
}


# nb: the class name is purposely distinct from Triton's JITFunction to simplify writing
# comments and docstrings, and make them unambiguous without additional context.
class JTJITFunction:
  """A wrapper around Triton's JITFunction/GluonJITFunction object to isolate the rest
  of the code from Triton's internals and provide a unified interface to the bits it needs.

  Additionally, it provides a persistence layer to ensure that certain data doesn't
  have to be re-created on each kernel launch. A user may assume that even when they
  create a new JTJITFunction object wrapping a previously used JITFunction object,
  the persistent data is reused.

  Since we don't instantiate JTJITFunction objects the way JITFunction objects are
  instantiated and JTJITFunction objects have short lives, we have to store the data in
  the very JITFunction object itself. For this we use custom attributes on the
  JITFunction object, prefixed with `_jT_`. The capitalized letter `T` breaks
  conventions to reduce the possibility of clashing with anything else. When we're ready
  to remove `triton_call()` in favor of the corresponding method of JTJITFunction to
  launch a kernel similarly to how the upstream Triton does it, we'll be able to remove
  this since JTJITFunction object will have a proper lifetime.
  """

  def __init__(
    self,
    fn: autotuner.Heuristics
    | autotuner.Autotuner
    | triton.JITFunction
    | gl_runtime.GluonJITFunction
    | Self,
  ):
    # peel off several potential wrapper layers to get to the JITFunction object
    if isinstance(fn, JTJITFunction):
      fn = fn.fn
    if isinstance(fn, autotuner.Autotuner):
      fn = fn.fn
    if isinstance(fn, autotuner.Heuristics):
      fn = fn.fn

    if not isinstance(fn, (triton.JITFunction, gl_runtime.GluonJITFunction)):
      raise TypeError(
        "`kernel` must be a Triton `JITFunction`, `GluonJITFunction`, `Heuristics`, or `Autotuner` object."
      )
    self.fn = fn

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

  @cached_property
  def arg_name_to_index(self) -> dict[str, int]:
    """Returns a dictionary mapping the kernel parameter names to their indices in the
    kernel's signature."""
    return {name: index for index, name in enumerate(self.arg_names)}

  @property
  def params(self) -> list[triton.runtime.jit.KernelParam]:
    return self.fn.params

  @property
  def is_gluon(self) -> bool:
    return isinstance(self.fn, gl_runtime.GluonJITFunction)

  @property
  def signature(self) -> inspect.Signature:
    return self.fn.signature

  def _get_cached_kernel(
    self,
    compute_capability: int,
    specialization: list[tuple[str, Any]],
    kwargs: dict[str, Any],
  ) -> tuple[str, triton_kernel_call_lib.TritonKernel | None]:
    if "_compute_capability" in kwargs:
      raise ValueError("'_compute_capability' key is reserved in the options/kwargs!")
    kwargs["_compute_capability"] = compute_capability

    if not hasattr(self.fn, "_jT_kernel_cache_key"):
      self.fn._jT_kernel_cache_key = {}

    # `triton_runtime_jit.compute_cache_key()` walks through kernel dependencies and
    # produces a hash with absolutely all kernel compilation dependencies accounted for.
    key = triton_runtime_jit.compute_cache_key(
      self.fn._jT_kernel_cache_key, specialization, kwargs
    )
    del kwargs["_compute_capability"]

    if not hasattr(self.fn, "_jT_kernel_cache"):
      self.fn._jT_kernel_cache = {}

    return key, self.fn._jT_kernel_cache.get(key, None)

  def _cache_kernel(
    self,
    key: str,
    kernel: triton_kernel_call_lib.TritonKernel,
    asm: dict | None = None,
  ):
    assert isinstance(self.fn._jT_kernel_cache, dict)
    self.fn._jT_kernel_cache[key] = kernel
    if asm is not None:
      if not hasattr(self.fn, "_jT_asm"):
        self.fn._jT_asm = {}
      self.fn._jT_asm[key] = asm

  @property
  def asm(self) -> dict[str, dict] | None:
    """Returns a dictionary of assembly files for the kernel, if they were saved during
    compilation."""
    return self.fn._jT_asm if hasattr(self.fn, "_jT_asm") else None

  @property
  def compiled_kernels_cache_size(self) -> int:
    return len(self.fn._jT_kernel_cache) if hasattr(self.fn, "_jT_kernel_cache") else 0

  def _make_signature_constexprs(
    self,
    named_args: dict[str, Any],
    sigvals: list[str],
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = dict(zip(self.arg_names, sigvals))

    constexprs = triton_runtime_jit.find_paths_if(
      sigvals, lambda _, val: val == "constexpr"
    )
    constexprs = {
      path: triton_runtime_jit.get_iterable_path(list(named_args.values()), path)
      for path in constexprs
    }
    return signature, constexprs

  @staticmethod
  def _make_attrs_nonconstexprs_sigvals(
    backend, specialization: list[tuple[str, Any]], named_args: dict[str, Any]
  ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    # We must always build attributes, as they are needed to launch the kernel
    attrvals = [x[1] for x in specialization]
    attrs = triton_runtime_jit.find_paths_if(attrvals, lambda _, x: isinstance(x, str))
    attrs = {
      k: backend.parse_attr(triton_runtime_jit.get_iterable_path(attrvals, k))
      for k in attrs
    }
    # attrs keys are tuples of integers representing a path into the (possibly nested)
    # kernel argument list as seen by the Triton compiler (i.e. including all
    # constexprs and whatnot — everything as declared in the signature)

    sigvals = [x[0] for x in specialization]
    non_constexprs = triton_runtime_jit.find_paths_if(
      sigvals, lambda _, val: val != "constexpr"
    )
    non_constexprs = {
      path: triton_runtime_jit.get_iterable_path(list(named_args.values()), path)
      for path in non_constexprs
    }
    return attrs, non_constexprs, sigvals

  def _get_binder(
    self, backend
  ) -> Callable[
    [Any, ..., Any], tuple[dict[str, Any], list[tuple[str, Any]], dict[str, Any]]
  ]:
    if not hasattr(self.fn, "_jT_binder"):
      self.fn._jT_binder = {}

    # In the upstream, a binder is per-`torch._C._cuda_getDevice()`. Here the closest
    # equivalent for this should be backend.hash() which is generally a GPU arch string.
    backend_hash = backend.hash()
    binder = self.fn._jT_binder.get(backend_hash, None)
    if binder is None:
      # create_function_from_signature() is a clever optimization in Triton that makes
      # a so-called binder function for the kernel, which simultaneously:
      # 1. assigns default values to all unspecified kernel arguments,
      # 2. assembles a complete dict of parameter_name->value mapping for all arguments,
      # 3. runs the correct specialization pipeline for all arguments, respecting all
      #   user annotations,
      # 4. finds key-value pairs in kwargs that don't match any kernel signature param.
      self.fn._jT_binder[backend_hash] = binder = (
        triton_runtime_jit.create_function_from_signature(
          self.fn.signature, self.fn.params, backend
        )
      )
    return binder

  def get_or_create_triton_kernel(
    self,
    make_gpu_target_func,
    platform: str,
    args: list[Any],
    *,
    compute_capability: int | None,
    kwargs: dict[str, Any],
  ) -> tuple[triton_kernel_call_lib.TritonKernel, dict[str, Any], dict[str, Any]]:
    num_warps = kwargs["num_warps"]
    num_ctas = kwargs["num_ctas"]
    assert all(
      isinstance(v, int) for v in (num_warps, kwargs["num_stages"], num_ctas)
    )  # internal sanity check

    store_asm = False
    if "_store_asm" in kwargs:
      store_asm = kwargs["_store_asm"]
      del kwargs["_store_asm"]  # this is JT internal, no need to leave it further

    backend, gpu_target, compute_capability = make_backend(
      make_gpu_target_func, compute_capability, num_ctas
    )

    named_args, specialization, other_kwargs = self._get_binder(backend)(
      *args, **kwargs
    )
    # named_args is a complete dict of kernel param names -> actual values.
    # other_kwargs is kwargs with keys matching kernel parameters removed.
    # specialization is a list[tuple[str, Any]], one entry per kernel parameter, that
    # captures two things about each argument at call time: data type reflection, and
    # specialization value (an attribute that may trigger a separately compiled kernel
    # variant). Specialization serves 3 goals: (1) it affects the kernel caching key,
    # (2) discover additional vars to be turned into constexprs by the compiler,
    # (3) is a source for the `attrs` spec.

    attrs, non_constexprs, sigvals = self._make_attrs_nonconstexprs_sigvals(
      backend, specialization, named_args
    )

    key, kernel = self._get_cached_kernel(
      compute_capability, specialization, other_kwargs
    )

    if kernel is None:
      # First, check that the kernel signature and the reconstructed signature have the
      # same number of parameters. A mismatch can occur due to differences in
      # `triton_call(input_output_aliases=)` handling between jax-triton versions.
      if len(self.params) != len(named_args):
        raise TypeError(
          f"Number of parameters in the kernel '{self.fn}' signature "
          f"({len(self.params)}: {self.signature}) "
          f"does not match reconstructed signature ({len(named_args)}: {list(named_args.keys())}). "
          "If the kernel was working on an older version of jax-triton and its "
          "triton_call() launcher uses `input_output_aliases` argument, note that "
          "implicit output arguments are no longer required for aliased arguments."
        )

      signature, constexprs = self._make_signature_constexprs(named_args, sigvals)

      backend_fields = _BACKEND_OPTIONS_FIELD_NAMES[gpu_target.backend]
      unrecognized = kwargs.keys() - named_args.keys() - backend_fields
      if len(unrecognized) > 0:
        raise ValueError(
          f"Unknown backend options: '{ {k: kwargs[k] for k in unrecognized} }' "
          f"were found in kwargs! Known options are: {backend_fields}."
        )

      backend_options = backend.parse_options(kwargs)

      if _JAX_TRITON_DUMP_DIR:
        kernel_hash = abs(hash(key))
        os.makedirs(f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}")
        with open(f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/config", "w") as f:
          pprint.pprint(key, stream=f)
          pprint.pprint(backend_options, stream=f)

      context = _triton.ir.context()
      _triton.ir.load_dialects(context)
      backend.load_dialects(context)
      codegen_fns = backend.get_codegen_implementation(backend_options)

      real_ASTSource = gl_runtime.GluonASTSource if self.is_gluon else tc.ASTSource
      module = real_ASTSource(
        self.fn, constexprs=constexprs, signature=signature, attrs=attrs
      ).make_ir(
        gpu_target, backend_options, codegen_fns, backend.get_module_map(), context
      )

      ttir = str(module)

      # triton_kernel_call_lib.TritonKernel() stores ttir internally, but does not
      # expose access to it, so we store it manually elsewhere for testing purposes.

      compilation_result = compile_ttir_inplace(
        module, backend, backend_options, compute_capability, platform
      )

      kernel_name = compilation_result.name
      if _JAX_TRITON_DUMP_DIR:
        with open(f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.ttir", "w") as f:
          f.write(ttir)
        with open(f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.ptx", "w") as f:
          f.write(compilation_result.binary)
        with open(
          f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.ttgir", "w"
        ) as f:
          f.write(compilation_result.ttgir)
        with open(f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.llir", "w") as f:
          f.write(compilation_result.llir)
        with open(
          f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.compile_info",
          "w",
        ) as f:
          f.write(
            f"{kernel_name}: shared_mem_bytes: {compilation_result.shared_mem_bytes}\n"
          )

      kernel = triton_kernel_call_lib.TritonKernel(
        kernel_name,
        num_warps,
        num_ctas,
        compilation_result.shared_mem_bytes,
        compilation_result.binary,
        ttir,
        compute_capability,
      )

      asm = dict(ttir=ttir) if store_asm else None
      self._cache_kernel(key, kernel, asm)

    return kernel, attrs, non_constexprs


def make_autotuner_configs(
  fn: autotuner.Autotuner,
  kwargs: dict[str, Any],
  named_args: dict[str, Any],
) -> list[triton.Config]:
  """Make and prune redundant autotuner configs based on user-provided kwargs.

  If any kwargs have been specified explicitly, we prune any configs that conflict.
  The pruning serves a specific need in jax-triton's architecture: unlike native Triton
  where autotuning happens dynamically at kernel launch, jax-triton must decide at
  lowering/tracing time which configs to compile. If the user has already fixed certain
  metaparameters (e.g., num_warps=4, BLOCK_SIZE=128), there's no point compiling or
  benchmarking autotuner configs that specify different values for those same
  parameters. The pruning eliminates those contradictory configs, reducing compilation
  and benchmarking work.

  Note that our implementation is more permissive than Triton's autotuner
  implementation, which will throw an error if any keys match.
  """
  prev_early_config_prune_fn = fn.early_config_prune

  def prune_configs(configs, named_args, **conf_kwargs):
    pruned_configs = []
    for config in configs:
      if config.pre_hook is not None:
        raise NotImplementedError("`pre_hook` is not supported")

      # Keep the config IFF for every user-provided kwargs(k->v), the config
      # either doesn't specify k at all, or specifies the same value v. This ensures
      # the config is coherent with explicit user choices.
      if all(config.kwargs.get(k, v) == v for k, v in kwargs.items()):
        pruned_configs.append(config)
    if prev_early_config_prune_fn is not None:
      pruned_configs = prev_early_config_prune_fn(pruned_configs, named_args)
    return pruned_configs

  fn.early_config_prune = prune_configs
  fn.nargs = named_args
  configs = fn.prune_configs(kwargs)
  return configs


def apply_heuristics(
  fn: autotuner.Heuristics,
  configs: list[triton.Config],
  orig_kwargs: dict[str, Any],
  named_args: dict[str, Any],
) -> list[triton.Config]:
  """Applies heuristics to the configs and returns the updated configs."""
  updated_configs = []
  for config in configs:
    kwargs = config.kwargs.copy()
    for name, heuristic in fn.values.items():
      kwargs[name] = heuristic({**named_args, **orig_kwargs, **kwargs})
    updated_config = copy.copy(config)
    updated_config.kwargs = kwargs
    updated_configs.append(updated_config)
  return updated_configs


def make_kernel_params(
  ctx,
  outputs_offset: int,
  kwargs: dict[str, Any],
  grid,
  zeroed_outputs,
  configs: list[triton.Config],
  operand_output_aliases: dict[int, int],
) -> list[dict[str, Any]]:
  """Make kernel call parameters for each config."""
  kernel_params = []
  for config in configs:
    # propagating backend-related config params
    config_metaparams = {
      **kwargs,
      **config.kwargs,
      "num_warps": config.num_warps,
      "num_stages": config.num_stages,
      "num_ctas": config.num_ctas,
      "maxnreg": config.maxnreg,
    }
    if config.maxnreg is None:
      del config_metaparams["maxnreg"]

    config_grid = normalize_grid(grid, config_metaparams)

    config_zeroed_outputs = (
      zeroed_outputs(config_metaparams) if callable(zeroed_outputs) else zeroed_outputs
    )

    # zeroed_params_with_sizes is a dict output_arg_idx -> aval_size_bytes
    # config_zeroed_outputs contains output ordinal indices
    output2input = {v: k for k, v in operand_output_aliases.items()}
    zeroed_params_with_sizes = {
      output2input[i] if i in output2input else i + outputs_offset: aval_size_bytes(
        ctx.avals_out[i]
      )
      for i in sorted(config_zeroed_outputs)
    }

    kernel_params.append(
      dict(
        metaparams=config_metaparams,
        grid=config_grid,
        zeroed_params_with_sizes=zeroed_params_with_sizes,
      )
    )
  return kernel_params


def convert_zeroed_outputs(
  jtfu: JTJITFunction,
  kwargs: dict[str, Any],
  zeroed_outputs: Sequence[InOutSpec] | Callable[[dict[str, Any]], Sequence[InOutSpec]],
) -> tuple[int, ...]:
  idx2name = jtfu.arg_names
  assert zeroed_outputs

  def _unwrap_compound(arg: Any) -> tuple[int, ...]:
    if not isinstance(arg, tuple):
      raise ValueError(
        f"Element of zeroed_outputs must reference an array or a tuple of arrays. Found {arg} of type {type(arg)}"
      )
    flat, _ = tree_util.tree_flatten(arg)
    assert all(isinstance(elm, JTArray) for elm in flat), (
      "All elements of a compound must be arrays"
    )
    return tuple(elm.index for elm in flat)

  def _unwrap(elm) -> tuple[int, ...]:
    """For a single element of zeroed_outputs returns raw indices of all output arrays
    it corresponds to"""
    is_int = isinstance(elm, int)
    if isinstance(elm, (int, str)):
      arg_name = idx2name[elm] if is_int else elm
      arg = kwargs[arg_name]
      # we could perhaps have a check for it being output argument too by modifying JTArray
      if isinstance(arg, JTArray):
        return (arg.index,)
      return _unwrap_compound(arg)
    elif isinstance(elm, (tuple, list)):
      if not elm:
        raise ValueError("Empty tuple or list is not a valid element of zeroed_outputs")
      top_elm = elm[0]
      if not isinstance(top_elm, (int, str)):
        raise ValueError(
          "First element of a tuple or list in zeroed_outputs must be an int or a str"
        )
      arg_name = idx2name[top_elm] if isinstance(top_elm, int) else top_elm
      arg = (
        kwargs[arg_name]
        if len(elm) == 1
        else triton_runtime_jit.get_iterable_path(kwargs[arg_name], elm[1:])
      )
      if isinstance(arg, JTArray):
        return (arg.index,)
      return _unwrap_compound(arg)
    else:
      raise ValueError(f"Invalid element of zeroed_outputs: {elm}")

  if callable(zeroed_outputs):
    orig_zeroed = zeroed_outputs
    zeroed_outputs = lambda kw: tuple(
      itertools.chain.from_iterable(map(_unwrap, orig_zeroed(kw)))
    )
  else:
    zeroed_outputs = tuple(itertools.chain.from_iterable(map(_unwrap, zeroed_outputs)))

  return zeroed_outputs


def add_output_args(
  jtfu: JTJITFunction,
  ctx,
  out_info: tuple[int, tree_util.PyTreeDef],
  args: list[Any],
  kwargs: dict[str, Any],
) -> dict[str, Any]:
  """Add output arguments to the argument list/dictionary."""
  num_pure_outputs, pure_out_tree = out_info
  assert num_pure_outputs <= len(ctx.avals_out)

  outputs = tree_util.tree_unflatten(
    pure_out_tree,
    (JTArray.from_avals_by_idx(ctx.avals_out, i) for i in range(num_pure_outputs)),
  )
  # outputs is a dict mapping out-spec->possibly nested structure of 1 or many JTArrays

  idx2name = jtfu.arg_names
  for out_spec, arrays in outputs.items():
    assert isinstance(out_spec, tuple)
    top_idx = out_spec[0]
    assert isinstance(top_idx, int)
    arg_name = idx2name[top_idx]
    if top_idx < len(args):
      raise ValueError(f"Output argument idx {top_idx}=>{arg_name} mustn't be in args")
    if len(out_spec) == 1:
      if arg_name in kwargs:
        raise ValueError(f"Output argument {arg_name} mustn't be in kwargs")
      kwargs[arg_name] = arrays
    else:
      # So this is a path into a compound, and this is a bit tricky, as we could only
      # expect tuples as compounds and tuples are immutable. We'd have to rebuild the
      # whole nested tuple structure to modify just a single element. For similar
      # purposes (see triton._utils.set_iterable_path()) Triton uses
      # triton.language.core.tuple class which is a list under the hood.
      raise ValueError(
        f"Output argument spec for {arg_name} is a nested {out_spec} and we don't currently support that"
      )
  return kwargs


def triton_kernel_call_lowering(
  make_gpu_target_func,
  ctx,
  *abstract_args,
  fn,
  kernel_call_name,
  out_shapes,
  grid,
  compute_capability,
  operand_output_aliases,
  zeroed_outputs,
  serialized_metadata,
  args_kwargs,
  out_info,
):
  operand_output_aliases = dict[int, int](operand_output_aliases)

  jtfu = JTJITFunction(fn)

  args, kwargs, nonabstracted = deserialize_args_kwargs(ctx, abstract_args, args_kwargs)

  # extending with outputs
  kwargs = add_output_args(jtfu, ctx, out_info, args, kwargs)
  if zeroed_outputs:
    zeroed_outputs = convert_zeroed_outputs(jtfu, kwargs, zeroed_outputs)

  if not isinstance(fn, (triton.JITFunction, gl_runtime.GluonJITFunction)):
    # Note: `fn.arg_names` below is not `JITFunction::arg_names`: Autotuner and
    # Heuristics classes have their own field and it is NOT deprecated.
    named_args = dict(unsafe_zip(fn.arg_names, args))
    # named_args are needed only for Autotuner/Heuristics handling.
    # Note that named_args is built from positional args only.
    # This is consistent with how the upstream handles it in Autotuner/Heuristics.
    # These are name-value pairs for positional arguments only, not `bound_args`
    # constructed from the kernel's signature.

  if isinstance(fn, autotuner.Autotuner):
    # Autotuner must be unpacked before Heuristics to ensure we reach the actual
    # kernel `fn`.
    configs = make_autotuner_configs(fn, kwargs, named_args)
    fn = fn.fn
  else:
    ttcfg_args = dict(
      num_warps=kwargs["num_warps"],
      num_stages=kwargs["num_stages"],
      num_ctas=kwargs["num_ctas"],
    )
    if "maxnreg" in kwargs:
      ttcfg_args["maxnreg"] = kwargs["maxnreg"]
    configs = [triton.Config({}, **ttcfg_args)]

  if isinstance(fn, autotuner.Heuristics):
    configs = apply_heuristics(fn, configs, kwargs, named_args)
    fn = fn.fn

  kernel_params = make_kernel_params(
    ctx,
    len(abstract_args),
    kwargs,
    grid,
    zeroed_outputs,
    configs,
    operand_output_aliases,
  )

  kernel_calls = []
  for params in kernel_params:
    kernel, specialization_attr, non_constexprs = jtfu.get_or_create_triton_kernel(
      make_gpu_target_func,
      ctx.module_context.platforms[0],
      args,
      compute_capability=compute_capability,
      kwargs=params["metaparams"],
    )

    call_params = []
    zeroed_params_with_sizes = params["zeroed_params_with_sizes"]
    array_idx = 0

    for path, arg in non_constexprs.items():
      if isinstance(arg, JTArray):
        arg_attrs = specialization_attr[path]
        call_params.append(
          triton_kernel_call_lib.create_array_parameter(
            zeroed_params_with_sizes.get(array_idx, 0),
            16 if (["tt.divisibility", 16] in arg_attrs) else 0,
            # TODO shouldn't we take "tt.divisibility" attr value instead from arg_attrs
            # if it's there?
          )
        )
        array_idx += 1
      else:
        dtype = get_type_id(arg)
        call_params.append(triton_kernel_call_lib.create_scalar_parameter(arg, dtype))

    kernel_calls.append(
        triton_kernel_call_lib.TritonKernelCall(
            kernel,
            params["grid"][0],
            params["grid"][1],
            params["grid"][2],
            call_params,
        )
    )

  if len(kernel_calls) > 1:
    input_output_aliases_with_sizes = tuple(
        (input_idx, output_idx, aval_size_bytes(ctx.avals_in[input_idx]))
        for input_idx, output_idx in operand_output_aliases.items()
    )
    kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
        f"{kernel_call_name} ({fn.fn.__name__}) {nonabstracted}",
        [(call, str(config)) for call, config in zip(kernel_calls, configs)],
        input_output_aliases_with_sizes,
    )
  else:
    kernel_call = kernel_calls[0]

  call_proto = kernel_call.to_proto(kernel_call_name, serialized_metadata)

  # Note: `operand_output_aliases` uses raw positional indices into the operands seen by
  # MLIR custom_call(). Scalars in JAX-Triton are embedded into
  # `backend_config`/`call_proto` and are not passed as MLIR operands; therefore,
  # indices in `operand_output_aliases` count only arrays among all inputs.

  # TODO(phawkins): remove forward_compat after 2026-05-04
  if jax.__version_info__ < (0, 10, 1) or ctx.is_forward_compat():
    rule = jax.ffi.ffi_lowering(
        "triton_kernel_call",
        api_version=2,
        backend_config=zlib.compress(call_proto),
        operand_output_aliases=operand_output_aliases,
    )
    return rule(ctx, *abstract_args)
  else:
    rule = jax.ffi.ffi_lowering(
        "triton_kernel_call_ffi",
        api_version=4,
        operand_output_aliases=operand_output_aliases,
    )
    return rule(ctx, *abstract_args, opaque=zlib.compress(call_proto))


mlir.register_lowering(
    triton_kernel_call_p,
    functools.partial(triton_kernel_call_lowering, make_gpu_target_cuda),
    platform="cuda",
)

mlir.register_lowering(
    triton_kernel_call_p,
    functools.partial(triton_kernel_call_lowering, make_gpu_target_hip),
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
  def shape(self) -> tuple[int, ...]: ...

  @property
  def dtype(self) -> np.dtype: ...


class JTArray:
  _default_alignment = 16

  @classmethod
  def from_avals_by_idx(cls, avals: list, idx: int):
    """Instantiate a JTArray from a list of avals and an index into the list."""
    return cls(avals[idx], idx)

  def __init__(self, aval, idx: int):
    self.dtype = get_type_id(aval)
    self.index = idx  # index into a corresponding avals array.
    # TODO(sharadmv,zhangqiaorjc): handle differently aligned pointers
    self.data_ptr = lambda: JTArray._default_alignment


def serialize_args_kwargs(jtfu, args, kwargs):
  """Prepares args and kwargs for passing through JAX's primitive system by separating
  things that must be traced from things that must be passed as-is.

  Returns a sequence of traced values, a mapping of array id -> flat array index to
  construct the operand_output_aliases mapping, and a tuple of hashable reconstruction
  info. Assumes the caller adheres to Triton's rules about argument types, so no
  special checks for hashability are done for performance reasons.
  """
  assert isinstance(jtfu, JTJITFunction), "jtfu must be a JTJITFunction object"
  leaves, treedef = tree_util.tree_flatten((args, kwargs))
  is_array = tuple(isinstance(v, jax.Array) for v in leaves)
  dyn_leaves = [v for v, b in zip(leaves, is_array) if b]
  static_leaves = tuple(to_python_type(v) for v, b in zip(leaves, is_array) if not b)
  array_id2idx = {id(v): i for i, v in enumerate(dyn_leaves)}
  return dyn_leaves, array_id2idx, (treedef, is_array, static_leaves)


def deserialize_args_kwargs(ctx, abstract_args, args_kwargs_meta):
  """Reconstructs args and kwargs from the serialized form. Abstract arguments are
  replaced by stubs."""
  treedef, is_array, static_leaves = args_kwargs_meta
  abs_iter = (
    JTArray.from_avals_by_idx(ctx.avals_in, i) for i in range(len(abstract_args))
  )
  static_iter = iter(static_leaves)
  leaves = [next(abs_iter) if b else next(static_iter) for b in is_array]
  args, kwargs = tree_util.tree_unflatten(treedef, leaves)
  return args, kwargs, static_leaves


ArrayId = int
OutShapeId = int


def _in_spec_to_ShapeDtypeStruct(
  elm: InOutSpec,
  args: list[Any],
  kwargs: dict[str, Any],
  name2idx: dict[str, int],
  idx2name: list[str],
  aliases: dict[ArrayId, OutShapeId],  # mutable
  shapes: list[jax.ShapeDtypeStruct],  # mutable
):
  """For an `in-spec` element `elm`, traverses `args`/`kwargs` to find all arrays
  addressed by it and updates a list of ShapeDtypeStruct objects describing those arrays
  and a dict mapping array identifiers to ShapeDtypeStruct identifiers.
  """
  # This code runs on each kernel launch, so thorough validation is expensive; we skip
  # most checks and let invalid specs fail naturally. We might check some things later.
  orig_name = None
  if isinstance(elm, int):
    path = (elm,)
  elif isinstance(elm, str):
    orig_name = elm
    path = (name2idx[elm],)  # user is responsible for correct indexing
  elif isinstance(elm, (tuple, list)):
    param_idx = elm[0]  # first element is an index into signature parameter list
    if isinstance(param_idx, str):
      orig_name = param_idx
      path = (name2idx[param_idx], *elm[1:])
    else:
      path = elm
  else:
    raise ValueError(f"Invalid in_spec element: {elm}")
  # checking if the path refers to a terminal array, or to a compound
  param_idx = path[0]
  if param_idx < len(args):  # must be in args by construction
    arg = triton_runtime_jit.get_iterable_path(args, path)
  else:  # must be in kwargs by construction
    arg = triton_runtime_jit.get_iterable_path(
      kwargs[idx2name[param_idx] if orig_name is None else orig_name], path[1:]
    )

  def _unpack_arg(elm: Any, shapes:list[jax.ShapeDtypeStruct]):
    if isinstance(elm, jax.Array):
      shape = jax.ShapeDtypeStruct(elm.shape, elm.dtype)
      aid = id(elm)
      if aid in aliases:
        raise ValueError(f"Array {elm} found under path {path} can't be aliased twice")
      aliases[aid] = id(shape)
      shapes.append(shape)
    else:
      if isinstance(elm, tuple):
        new_shapes = []
        for e in elm:
          _unpack_arg(e, new_shapes)
        if new_shapes:
          shapes.append(tuple(new_shapes))
      # else: just ignore any other arguments

  if isinstance(arg, jax.Array):
    _unpack_arg(arg, shapes)
  elif not isinstance(arg, tuple):
    raise ValueError(
      f"Aliased element {arg} at path {path} is not a jax.Array or a tuple of them"
    )
  else:
    for elm in arg:
      _unpack_arg(elm, shapes)


def make_aliased_shapes(
  jtfu: JTJITFunction,
  input_output_aliases: dict[int, int] | Sequence[InOutSpec],
  pure_out_shapes: dict[CanonicalKernelArgPath, Sequence[jax.ShapeDtypeStruct]],
  orig_aliased_shapes: Sequence[jax.ShapeDtypeStruct],
  args: list[Any],
  kwargs: dict[str, Any],
) -> tuple[tuple[jax.ShapeDtypeStruct], dict[ArrayId, OutShapeId]]:
  """Uses `input_output_aliases` to produce a tuple of aliased shapes to append to
  `out_shapes` for the Primitive's .bind() call, and a dict mapping input array
  identifiers to output array identifiers. After serialization, we'll be able to
  create `operand_output_aliases` dictionary based on raw array indices using this
  mapping."""
  assert isinstance(jtfu, JTJITFunction), "jtfu must be a JTJITFunction object"
  # building aliased_shapes from in_spec
  is_dict = isinstance(input_output_aliases, dict)
  in_spec = list(input_output_aliases.keys()) if is_dict else input_output_aliases
  if isinstance(in_spec, (int, str)) or (
    isinstance(in_spec, tuple) and in_spec and not isinstance(in_spec[0], (tuple, list))
  ):
    in_spec = [in_spec]

  name2idx = jtfu.arg_name_to_index
  idx2name = jtfu.arg_names
  aliases = {}

  def _make_aliased(elm: Any, shapes: list[jax.ShapeDtypeStruct]) -> None:
    inner_shapes = []
    if isinstance(elm, list) or (
      isinstance(elm, tuple) and isinstance(elm[0], (tuple, list))
    ):
      tuple(_make_aliased(e, inner_shapes) for e in elm)
      assert len(inner_shapes) == len(elm)
      shapes.append(tuple(inner_shapes))
    else:
      _in_spec_to_ShapeDtypeStruct(
        elm, args, kwargs, name2idx, idx2name, aliases, inner_shapes
      )
      inner_shapes = inner_shapes[0] if len(inner_shapes) == 1 else tuple(inner_shapes)
      shapes.append(inner_shapes)

  aliased_shapes = []
  tuple(_make_aliased(e, aliased_shapes) for e in in_spec)

  # TODO: perhaps remove the whole `if is_dict:` below to spare cycles?
  # if it's a dict, validate that out_shapes match aliased_shapes
  if is_dict:  # and kwargs["debug"]:
    # getting a starting index of aliased shapes in the original out_shape for reading
    # a dict values
    aliased_idx = len(pure_out_shapes)
    for oidx in input_output_aliases.values():
      oidx -= aliased_idx
      if oidx < 0 or oidx >= len(aliased_shapes):
        raise ValueError(f"Output index {oidx} is out of range for aliased shapes")
      if aliased_shapes[oidx] != orig_aliased_shapes[oidx]:
        raise ValueError(
          f"Output shape {aliased_shapes[oidx]} at index {oidx} doesn't match to the shape "
          f"in `out_shape[{oidx + aliased_idx}]` {orig_aliased_shapes[oidx]}"
        )
  return tuple(aliased_shapes), aliases


def _split_out_values(
  out_values: Sequence,
  out_names: None | InOutSpec,
  input_output_aliases: dict[int, int] | Sequence[InOutSpec],
) -> tuple[tuple, Sequence]:
  """Splits `out_values` into a tuple of purely output values and a tuple of aliased
  values."""
  # If we know `out_names`, the first len(out_names) elements in `out_values`
  # must describe pure outputs. Else if there's a dict form of `input_output_aliases`,
  # then len(input_output_aliases) is the number of aliased buffers/shapes. Otherwise,
  # there are no aliased elements.
  if out_names is not None:
    num_purely_output = len(out_names)
    num_aliased = len(out_values) - num_purely_output
    if num_aliased < 0 or (
      isinstance(input_output_aliases, dict)
      and num_aliased != len(input_output_aliases)
    ):
      raise ValueError(
        "Content of `out_shape` doesn't match to `out_names` and `input_output_aliases` specification"
      )
  elif isinstance(input_output_aliases, dict):
    num_aliased = len(input_output_aliases)
    num_purely_output = len(out_values) - num_aliased
    if num_purely_output < 0:
      raise ValueError(
        "There was specified more aliased buffers than there are output arguments"
      )
  else:
    num_aliased = 0
    num_purely_output = len(out_values)

  if num_aliased > 0:
    aliased_shapes = out_values[num_purely_output:]
    out_values = out_values[:num_purely_output]
  else:
    aliased_shapes = []
  return out_values, aliased_shapes


def canonicalize_out_shape(
  jtfu: JTJITFunction,
  out_shape: ShapeDtype | Sequence[ShapeDtype] | dict[InOutSpec, Sequence[ShapeDtype]],
  out_names: None | InOutSpec,
  args: list[Any],  # positional args passed to the launcher
  input_output_aliases: dict[int, int] | Sequence[InOutSpec],  # original aliases
) -> tuple[
  dict[CanonicalKernelArgPath, Sequence[jax.ShapeDtypeStruct]],
  Sequence[jax.ShapeDtypeStruct],
]:
  """Converts different forms of out_shape and out_names to a single dict mapping from
  output kernel argument paths to shapes/dtypes of the output arguments for output-only
  arguments (which have corresponding parameters in the kernel's signature), and a
  sequence of shapes/dtypes of aliased arguments (which reuse input parameters and
  therefore have no separate output parameters in the kernel's signature).
  """
  assert isinstance(jtfu, JTJITFunction), "jtfu must be a JTJITFunction object"
  if isinstance(out_names, (int, str)):
    out_names = (out_names,)
  # a basic validation that will be elided by the JIT compiler, so it's fine
  if out_names is not None and not isinstance(out_names, (tuple, list)):
    raise ValueError(
      "out_names must be None, a string, an int, or a tuple/list of specific format"
    )

  if isinstance(out_shape, dict):
    # The dictionary form doesn't have info on aliased buffers
    if out_names is None:
      out_names = tuple(out_shape.keys())
      out_values = tuple(out_shape.values())  # must materialize
    else:
      # TODO: remove the check to speedup things/check only lengths, or add cache?
      if frozenset(out_names) != out_shape.keys():
        raise ValueError("out_names and out_shape must have the same keys")
      out_values = tuple(out_shape[name] for name in out_names)  # reorder

    aliased_shapes = []
  else:
    if isinstance(out_shape, list):
      out_shape = tuple(out_shape)
    out_values = out_shape if isinstance(out_shape, tuple) else (out_shape,)
    # out_values here might contain shapes for aliased elements, we need to split them
    out_values, aliased_shapes = _split_out_values(
      out_values, out_names, input_output_aliases
    )
    # TODO we could perhaps even drop `aliased_shapes` entirely, as we reconstruct them
    # from the signature
    if out_names is None:
      # building out_names from the assumption that `args` contain all input arguments
      arg_names = jtfu.arg_names
      # out_values may contain compounds, hence we must consider non-flattened shapes
      out_names = tuple(arg_names[i + len(args)] for i in range(len(out_values)))
    if len(out_names) != len(out_values):  # this checks only top level specs
      raise ValueError("out_shape specification mismatches out_names")
  del out_shape  # safer to hide from scope as it's no longer needed

  def _to_ShapeDtypeStruct(a: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(a.shape, a.dtype)

  if aliased_shapes:
    aliased_shapes = tree_util.tree_map(_to_ShapeDtypeStruct, aliased_shapes)
  out_values = tree_util.tree_map(_to_ShapeDtypeStruct, out_values)

  # now canonicalizing out_names using the out_values.
  pure_out_shapes: dict[CanonicalKernelArgPath, Sequence[jax.ShapeDtypeStruct]] = {}
  name2idx = jtfu.arg_name_to_index
  for i, outn in enumerate(out_names):
    ospec = _canonicalize_out_spec_element(outn, name2idx)
    pure_out_shapes[ospec] = out_values[i]
  # While the ordering of pure_out_shapes here reflects the ordering of out_names, it
  # does not mean that ordering will be used for passing arguments to the kernel or
  # producing outputs. Two things affect that:
  # First, `tree_util.tree_flatten()` is used to serialize compound data structures
  # including output argument info, and JAX assumes that flattening also sorts
  # dictionary keys (otherwise it would break the entire Primitive call caching system).
  # Keys here are `tuple[int, ...]` where ints are kernel signature parameter indices,
  # so they will be sorted accordingly. Using `OrderedDict()` here could bypass this,
  # but it would not help: later we inject output arguments into `kwargs` as Triton
  # variables, which also nullifies any imposed ordering in favor of kernel signature
  # order. During kernel launch, we also iterate over kernel arguments in signature
  # order. Hence, effectively, there is only one ordering: the kernel signature's.

  return pure_out_shapes, aliased_shapes


def _canonicalize_out_spec_element(
  elm: InOutSpec,
  name2idx: dict[str, int],
) -> CanonicalKernelArgPath:
  if isinstance(elm, int):
    path = (elm,)
  elif isinstance(elm, str):
    path = (name2idx[elm],)  # user is responsible for correct indexing
  elif isinstance(elm, tuple):
    prim_idx = elm[0]
    if isinstance(prim_idx, str):
      path = (name2idx[prim_idx], *elm[1:])
    else:
      if not isinstance(prim_idx, int):
        raise ValueError(f"Invalid first element of this out-spec: {elm}")
      path = elm
  else:
    raise ValueError(f"Invalid out-spec element: {elm}")
  # No need to go deeper. A Triton variable will be created for whatever shape the
  # corresponding element of out_shape has.
  return path


def triton_call(
    *args: jax.Array | bool | int | float | np.float32,
    kernel: (
        triton.JITFunction
        | gl_runtime.GluonJITFunction
        | triton.runtime.Heuristics
        | triton.runtime.Autotuner
    ),
    grid: GridOrLambda,
    out_shape: ShapeDtype | Sequence[ShapeDtype] | dict[str, ShapeDtype] = (),
    out_names: None | str | tuple[str, ...] = None,
    name: str = "",
    num_warps: int | None = None,
    num_stages: int | None = None,
    num_ctas: int = 1,  # TODO(giorgioa): Add support for dimensions tuple.
    compute_capability: int | None = None,
    enable_fp_fusion: bool = True,
    input_output_aliases: dict[int, int] | None = None,
    zeroed_outputs: (
        Sequence[InOutSpec] | Callable[[dict[str, Any]], Sequence[InOutSpec]]
    ) = (),
    debug: bool = False,
    serialized_metadata: bytes = b"",
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
    block_size = 8
    return jt.triton_call(
        x,
        y,
        kernel=add_kernel,
        out_shape=x,
        grid=(x.size // block_size,),
        block_size=block_size)

  x_val = jnp.arange(8)
  y_val = jnp.arange(8, 16)
  print(add(x_val, y_val))
  print(jax.jit(add)(x_val, y_val))
  ```

  Kernel output and aliasing specification introduce necessary complexity on top of
  Triton to interoperate with JAX. To address this, a concept of "Kernel's signature
  coordinate system" is used — it is a hashable way to address an individual array at
  any position in the kernel's (*args, **kwargs), regardless of nesting depth. A
  "coordinate" refers to either a single array or all arrays nested within the addressed
  tuple. It can be:
    - an int: indexes a kernel signature parameter.
    - a string: parameter name in the kernel's signature.
    - a tuple whose first element is a string (identifies a parameter name in the
      kernel's signature) or an int (indexes a kernel signature parameter). All other
      elements of the tuple must be ints indexing into the compound argument
      corresponding to the parameter at index 0.
      For example: a tuple `("Ptrs", 0, 1)` refers to the `Ptrs` parameter of the
      kernel, and expects the corresponding argument to be a tuple whose 0th element is
      another tuple, and whose 1st element is an array or a tuple of arrays/tuples.

  Args:
    *args: Positional inputs for the Triton kernel. In the kernel signature, purely
      output parameters should go after the last input parameter to be passed as
      a positional argument to `triton_call()`.

    kernel: Triton (e.g. a function decorated with `triton.jit`) or a Gluon kernel.
      All static values should be annotated with `triton.language.constexpr` or
      `triton.experimental.gluon.language.constexpr`. All other Triton annotations are
      also supported.

    out_shape: Description of shapes and dtypes of output parameters of the kernel.
      Can be one of the following:

      (1) a single ShapeDtype-like object (something that has `.shape` and
      `.dtype` attributes) describing a single output parameter of the kernel; or an
      ordered, potentially nested, sequence of ShapeDtype-like objects corresponding to
      one or more output parameters, where each root-level element of the sequence
      describes one kernel output parameter in its signature (the nesting defines the
      form of the individual argument; for example, `out_shape=((a,b),c)` defines two
      output arguments: the first is a tuple of arrays with the shapes and dtypes of
      arrays a and b, and the second is an array with the shape and dtype of array c).
      Ordering of elements in the sequence must correspond to the ordering of the output
      parameters in the kernel's signature.

      (2) a dictionary mapping kernel output parameter names or indices to
      one ShapeDtype-like object or an ordered, potentially nested, sequence of such
      objects. The nested structure of dictionary values defines the structure of the
      output argument. For example, `out_shape={"a": (x,y), "b": z}` defines that the
      kernel has two output parameters: the parameter named "a" is a tuple of 2 arrays
      borrowing their shapes and dtypes from arrays x and y, and "b" is a single
      `z`-like array.
      If `out_names` is specified, the `out_shape` must have the same keys. Ordering
      of the kernel signature parameters takes precedence over the ordering of the
      dictionary keys.
      If `input_output_aliases` is needed and a dictionary form of `out_shape` is used,
      `input_output_aliases` must use a non-dictionary form, enumerating input arrays
      only.

      Unlike Triton itself, JAX/XLA must manage memory buffers and organize data copying
      between host and device memory for all array parameters, so `out_shape` provides
      the necessary information.
      Due to the JAX custom call API, there is currently a restriction on the ordering
      of input/output parameters in the kernel signature: all purely output parameters
      must reside after the last non-constexpr input parameter in the flattened kernel
      signature parameter list. Interleaving input and output parameters in the
      signature is not supported.

      If `out_shape` uses a dictionary form, or if `out_names` is provided, there are
      no other restrictions. In other cases, all input arguments that are not
      specialized as constexprs must be passed strictly as positional `args`. `out_names`
      are inferred by assuming output parameters follow the positional arguments passed
      as `args` to `triton_call()`.

      Note that regardless of which form of `out_shape` is used, arguments
      corresponding to strictly output parameters must never be passed to
      `triton_call()` as part of `args` or `kwargs` — they are added implicitly by the
      launcher. If an argument is intended to be an input-output argument, set
      `input_output_aliases` accordingly.

    out_names: A sequence of output parameter names.
      Another way to specify the kernel's output parameter names. Useful in conjunction
      with the `@kernel` decorator.
      If `out_shape` is a dictionary, its keys must be consistent with `out_names`.

    grid: An integer, tuple of up to 3 integers, or a function that returns a
      tuple of up to 3 integers. When `grid` is an integer, `kernel` is
      invoked in `grid`-many parallel executions. When `grid` is a sequence of
      integers, `kernel` is launched in a `prod(grid)`-many parallel execution.
      When `grid` is a function, it is passed `kwargs` and should return a tuple of up
      to 3 integers.

    input_output_aliases: Tells XLA that it shouldn't allocate fresh output buffers for
      these arguments, but instead alias the input buffers (requiring the backend to
      arrange not only `host->device` data copying before kernel launch, but
      also `device->host` data copying after kernel execution). Can take two forms:

      (1 — deprecated): A dictionary mapping an input array coordinate in the kernel's
      signature coordinate space to the `out_shape` sequence coordinate space.
      Important note: if the kernel also has purely output arguments, they must go first
      in the `out_shape` sequence, and aliased buffers must go last in the sequence.
      Interleaving output with in/out argument shapes in `out_shape` is not supported.
      The user is responsible for adhering to this requirement.

      (2 — recommended): An ordered, potentially nested, sequence of input array
      coordinates in the kernel's signature coordinate space. When this form is used,
      there should be no corresponding output arguments in `out_shape`. Shapes and
      dtypes are inferred from referenced input arguments automatically (JAX/XLA does
      not support typecasting of aliased buffers anyway, so `out_shape` entries for
      aliased buffers are always redundant). It can specifically be:
      - a single int, a string, or a tuple having an int or a string as its first
        element: describes an input array in the kernel's signature coordinate space.
      - a list: ordered, possibly nested, sequence of the above. The nested structure of
        the sequence describes the structure of the corresponding output argument,
        containing aliased buffers as returned by `triton_call()`. For example,
        `input_output_aliases=["out_ptr",[(2,0), "out_ptr2"]]` describes three aliased
        (input-output) buffers: the first is an input array passed to the `out_ptr`
        kernel parameter, the second is an array at index 0 of a tuple passed to the
        third positional parameter, and the third is an input array passed for
        `out_ptr2`. These three buffers (assuming all are single arrays) are returned
        from `triton_call()` as a tuple of two elements: [0] is a new array wrapping
        the same buffer used for the `out_ptr` kernel parameter, and [1] is a tuple of
        two new arrays wrapping the other two buffers. If some referenced input argument
        is not an array but a possibly nested tuple of arrays, all its internal arrays
        are considered aliased and are returned as a flat tuple of arrays.
        Note that tuples can also be used instead of lists, but remember that if a
        tuple's first element is not another iterable, it is always treated as a
        coordinate descriptor, which may not be what you want. For example,
        `input_output_aliases=(0,1)` is always treated as a reference to an array at
        the second position of a tuple passed to the first kernel parameter, while
        `input_output_aliases=[0,1]` references two arrays passed to the first and
        second kernel parameters respectively. Hence, for clarity, it is recommended to
        use tuples only when describing a single input array coordinate.
      Note that if a kernel needs to start working with an initially zeroed read-write
      buffer to be returned as an output (for example, for sparse updates), you should
      designate a purely output argument for it with a proper `out_shape` specification
      and list its coordinate in `zeroed_outputs`.

      Attention: You as a user is responsible for ensuring that all arrays listed in
      input_output_aliases are used strictly once anywhere in *args, **kwargs (i.e.
      for each array A in (*args, **kwargs) from input_output_aliases, id(A) is not
      equal to ANY other id(B) of an array B from (*args, **kwargs)). Failure to adhere
      to that rule might lead to an undefined behavior.

    zeroed_outputs: A sequence of kernel signature coordinates of output arguments, or
      a function taking a dict of metaparameters and returning a sequence of such
      coordinates, for outputs that should be zeroed before the kernel is launched.

      BREAKING: This argument does NOT support zeroing input-output (i.e. aliased
      through `input_output_aliases`) arguments anymore, since this breaks the semantics
      of input-output parameters: you cannot pass information from the host to the
      kernel through the buffer, effectively turning an input-output argument into a
      purely output argument. Note the slight terminological ambiguity here:
      input-output arguments are named from the backend's perspective, whose job is to
      manage memory buffers and arrange data copying between host and device memory for
      all array parameters. A kernel can always read from what is called a purely output
      argument, so if you need the kernel to start from a zeroed buffer, declare it as a
      purely output argument in `out_shape` and list its coordinate in `zeroed_outputs`.

      Note that a callable form of `zeroed_outputs` is allowed to modify the
      metaparameters dict passed to it if and only if the modification does not affect
      kernel compilation (for example, by not changing parameters that affect kernel
      specialization), and the modification is propagated to the kernel call. If the
      modification does affect kernel compilation, the behavior is undefined.

    num_warps: The number of warps used to execute the Triton kernel.
    num_stages: The number of stages emitted by the Triton compiler.
    num_ctas: The size of thread blocks per cluster to be used on GPUs with
      compute capabilities >= 9.0. It must be less or equal to 8.
    enable_fp_fusion: Whether to enable floating-point operands fusion for the kernel.
    debug: Prints out intermediate IRs if True for debugging purposes. Also passed as a
      backend options argument.
    serialized_metadata: Arbitrary metadata that will be added into the
      serialized kernel call.

    kwargs: Additional keyword arguments (num_warps, num_stages, num_ctas, debug, and
      enable_fp_fusion are added automatically) that are provided to:
      - a `grid` (if it is a function),
      - the backend options constructor (only recognized arguments are forwarded),
      - the Triton kernel as `constexpr` arguments (constexprs must always be scalars)
        or regular runtime arguments.

  Returns:
    Outputs from the Triton kernel. First go all purely output arguments, then all
    input-output arguments.
  """
  # TODO(Arech): improve error reporting and check for assumption violations. We make
  # many assumptions about user behavior. Most, if not all, of these assumptions can be
  # verified. We skip this for performance or other reasons, which hurts the user
  # experience. Extensive validation is possible in at least two cases: (1) some kernel
  # launch parameters don't vary much across calls because they reflect fixed kernel
  # properties; their processing could be cached. The uncached first run could do full
  # validation and all subsequent runs would be fast. (2) the `debug` parameter is
  # designed precisely for this.

  # Python guarantees these keys don't exist yet. The original Triton has a single
  # namespace for both constexprs and backend options; we do the same to unify
  # processing. Defaults are set early because we use these values early.
  kwargs["num_warps"] = num_warps if num_warps is not None else 4
  kwargs["num_stages"] = num_stages if num_stages is not None else 3
  kwargs["num_ctas"] = num_ctas
  kwargs["enable_fp_fusion"] = enable_fp_fusion
  kwargs["debug"] = debug

  if input_output_aliases is None:
    input_output_aliases = []

  jtfu = JTJITFunction(kernel)
  pure_out_shapes, aliased_shapes = canonicalize_out_shape(
    jtfu, out_shape, out_names, args, input_output_aliases
  )
  # The structure of `pure_out_shapes` will let us reconstruct the output variables for
  # Triton using ctx.avals_out instead of arrays; the dict keys will let us place them
  # correctly into args/kwargs and properly structure the returned outputs.

  if input_output_aliases or input_output_aliases == 0:
    aliased_shapes, aliases = make_aliased_shapes(
      jtfu, input_output_aliases, pure_out_shapes, aliased_shapes, args, kwargs
    )
  else:  # quick shortcut
    aliases = {}
    if aliased_shapes:
      raise ValueError("out_shape isn't coherent with input_output_aliases!")

  # The structure of `aliased_shapes` helps return aliased arrays in the correct form;
  # flattened values are the shapes passed to .bind(). `aliases` helps generate the
  # properly array-only indexed `operand_output_aliases` needed for the FFI interface.
  # We match aliased (from XLA's point of view) arrays using Python's id() function.
  # From the point of view of constructing `operand_output_aliases`, it's a benign
  # situation when the same input-only array is used several times as a kernel parameter
  # (pure inputs have no role in `operand_output_aliases` and when their id()'s collapse
  # to a single value, it's still ignored), however, we require a user to never use
  # an `input_output_aliases`-ed array twice or more as a kernel parameter, since this
  # is already a UB from XLA's and compiler's point of view (when two arguments refer
  # to the same memory location, in general case, write to which argument takes
  # precedence?).
  abs_args_kwargs, array_id2idx, args_kwargs_meta = serialize_args_kwargs(
    jtfu, args, kwargs
  )

  pure_out_shapes_flat, pure_out_tree = tree_util.tree_flatten(pure_out_shapes)
  num_pure_outputs = len(pure_out_shapes_flat)

  aliased_shapes_flat, aliased_tree = tree_util.tree_flatten(aliased_shapes)
  aliased_shape_id2idx = {  # shape id -> raw output array idx
    id(s): i + num_pure_outputs for i, s in enumerate(aliased_shapes_flat)
  }
  operand_output_aliases = {
    array_id2idx[arr_id]: aliased_shape_id2idx[shp_id]
    for arr_id, shp_id in aliases.items()
  }

  if num_pure_outputs + aliased_tree.num_leaves == 0:
    raise ValueError(
      "No outputs specified for the kernel, DCE will eliminate the call entirely"
    )

  out_flat = triton_kernel_call_p.bind(
      *abs_args_kwargs,
      fn=kernel,
      kernel_call_name=name,
      # out_shapes must be a flat sequence of shapes of ALL arrays to be returned:
      # purely output + input-output (aliased) arrays.
      out_shapes=(*pure_out_shapes_flat, *aliased_shapes_flat),
      grid=grid,
      compute_capability=compute_capability,
      operand_output_aliases=tuple(operand_output_aliases.items()),
      zeroed_outputs=zeroed_outputs,
      serialized_metadata=serialized_metadata,
      args_kwargs=args_kwargs_meta,
      out_info=(num_pure_outputs, pure_out_tree),
  )

  pure_outs = (
    tree_util.tree_unflatten(pure_out_tree, out_flat[:num_pure_outputs])
    if num_pure_outputs > 0
    else {}
  )
  aliased_outs = (
    tree_util.tree_unflatten(aliased_tree, out_flat[num_pure_outputs:])
    if aliased_tree.num_leaves > 0
    else ()
  )
  ret = tuple(pure_outs.values()) + aliased_outs
  return ret[0] if len(ret) == 1 else ret
