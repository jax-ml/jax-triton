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
from typing import Any, Protocol, Union
import zlib

import jax
from jax import tree_util
from jax._src import core, util
from jax._src.interpreters import partial_eval as pe
from jax._src.lib import gpu_triton as triton_kernel_call_lib
import jax.extend as jex
from jax.interpreters import ad, batching, mlir, xla
import jax.numpy as jnp
import numpy as np


import triton
from triton.compiler import compiler as tc
# import triton.language as tl
from triton.runtime import autotuner
import triton.runtime.jit as triton_runtime_jit
import triton._C.libtriton as _triton
import triton.backends.nvidia.compiler as cb
import triton.backends.amd.compiler as hb

import triton.experimental.gluon._runtime as gl_runtime


if "TRITON_CACHE_DIR" in os.environ:
  del os.environ["TRITON_CACHE_DIR"]

_JAX_TRITON_DUMP_DIR = os.environ.get("JAX_TRITON_DUMP_DIR")
map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

Grid = Union[int, tuple[int], tuple[int, int], tuple[int, int, int]]
GridOrLambda = Union[Grid, Callable[[dict[str, Any]], Grid]]

# a coordinate of an element in kernel parameters coordinate space. The first int indexes
# kernel parameter in order of its declaration in the signature, and the following ints
# (if any) indexes into nested tuples inside the actual argument. So this is a kernel
# call attribute, as it depends on actual argument values, not a kernel attribute (like
# parameters are)
# Triton calls a coordinate a path, so have to follow
CanonicalKernelArgPath = tuple[int, ...]
# CanonicalKernelArgPaths = tuple[CanonicalKernelArgPath, ...]
KernelArgPath = tuple[int | str, *tuple[int, ...]]
InOutSpec = int | str | tuple[int | str | KernelArgPath, ...]

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


# Use of types is slightly messy here. First of all, Triton itself uses exact dtypes
# for arrays, but for scalars (no matter if constexpr, or runtime), it accepts native
# Python objects only. The actual type of a kernel parameter is determined from the
# parameter type annotation only. So to feed specialization engine properly, we have to
# typecast all scalars to Python types. Caveats:
# 1. we must not typecast constexprs as it's a separate type influencing specialization.
# 2. Triton's runtime specialization for scalars unconditionally turns all floats into
# fp32. It's unclear if this is a bug or not - the respective code doesn't even have a
# string to describe fp64. Probably Triton's kernel launcher doesn't support doubles,
# but JAX's `create_scalar_parameter()` support doubles perfectly well. However, Triton
# still makes binaries that expect floats only, so there's no point to handle it
# differently.
#
# Fortunately, both (Triton and JAX) use the same (likely LLVM/MLIR based) type
# identifiers, so the function below serves both goals - for Triton's specialization and
# for JAX's launcher.
def get_type_id(obj: Any) -> str:
  """Returns a Triton/JAX type identifier for the given object. Expected to be used with
  run-time values only. Constexprs (string are also constexprs only in Triton) don't
  need type info."""
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
    return "fp32"  # Triton unconditionally treat all floats as fp32, see the note above
  raise NotImplementedError(f"could not compute type name for {obj}: {type(obj)}")


def to_python_type(arg: Any) -> Any:
  """Typecasts a scalar to a native Python type."""
  # Note that typecasting ints/floats to respective types also convert JAX's subclasses
  # like TypedInt and TypedFloat, which choke nanobind's type caster in a default
  # strict-mode.
  if isinstance(arg, (bool, np.bool_)):
    arg = bool(arg)
  elif isinstance(arg, (int, np.integer)):
    arg = int(arg)
  elif isinstance(arg, (float, np.floating)):
    arg = float(arg)
  # else return as is and let it possibly, but not necessarily fail (constexprs and
  # strings pass and the rest isn't expected to be here, so sparing cycles on that)
  return arg


def normalize_grid(grid: GridOrLambda, metaparams) -> tuple[int, int, int]:
  if callable(grid):
    grid = grid(metaparams)
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))


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
    core.ShapedArray(out_shape.shape, out_shape.dtype) for out_shape in out_shapes
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
    opt_ttir = cuda_backend.make_ttir(ttir, metadata, cuda_options, compute_capability)
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
) -> tuple[triton.compiler.BaseBackend, triton.compiler.GPUTarget, int]:
  """Resolves compute_capability and spawns triton's Backend and GPUTarget objects"""

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


class JTKernel:
  def __init__(self, kernel, triton_call_kwargs):
    if not isinstance(
      kernel,
      (
        triton.JITFunction,
        gl_runtime.GluonJITFunction,
        autotuner.Heuristics,
        autotuner.Autotuner,
        JTKernel,
      ),
    ):
      raise ValueError(
        "`kernel` must be a `JTKernel` or Triton's `JITFunction`, `GluonJITFunction`, `Heuristics` or `Autotuner` object."
      )

    if isinstance(kernel, JTKernel):
      triton_call_kwargs = {**kernel.triton_call_kwargs, **triton_call_kwargs}
      kernel = kernel.kernel

    self.kernel = kernel

    if "grid" in triton_call_kwargs:
      raise ValueError("grid must be provided as an indexing argument to this object")

    # since this code works only once on a kernel declaration point, we could add as
    # much validation as possible here, to avoid doing it on every kernel launch instead
    if "out_names" in triton_call_kwargs:
      # we rely on a user correctly specifying that the order of elements in "out_names"
      # reflect that in the kernel's signature. Checking that here as much as possible
      triton_call_kwargs = self._validate_out_names(kernel, triton_call_kwargs)

    self.triton_call_kwargs = triton_call_kwargs

  @staticmethod
  def _validate_out_names(kernel, triton_call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validates as much as possible the order of elements in
    triton_call_kwargs['out_names']"""
    out_names = triton_call_kwargs["out_names"]
    if isinstance(out_names, (str, int)):
      out_names = (out_names,)
    elif isinstance(out_names, list):  # won't do this in triton_call(), but here it
      out_names = tuple(out_names)  # isn't expensive.
    elif not isinstance(out_names, tuple):
      raise ValueError("out_names must be a string, or a tuple of specific format")
    triton_call_kwargs["out_names"] = out_names

    jtfu = JTJITFunction(kernel)  # refactor this out if other validation funcs appear
    arg_names = jtfu.arg_names
    last_idx = -1
    for i, out_elm in enumerate(out_names):
      is_tuple = isinstance(out_elm, tuple)
      if is_tuple and len(out_elm) <= 0:
        raise ValueError(f"out_names[{i}] must be an nonempty tuple")
      if is_tuple and any(not isinstance(e, int) for e in out_elm[1:]):
        raise ValueError(
          "A tuple form of out_names must contain only ints beyond the first element"
        )
      is_first_string = is_tuple and isinstance(out_elm[0], str)
      if isinstance(out_elm, str) or is_first_string:
        try:
          param_idx = arg_names.index(out_elm[0] if is_first_string else out_elm)
        except ValueError:
          raise ValueError(f"out_names[{i}] must be a valid kernel parameter name")
        # without args/kwargs we can't look deeper anyway
      else:
        param_idx = out_elm[0] if is_tuple else out_elm
        if not isinstance(out_elm, int):
          raise ValueError(f"out_names[{i}] must be an int or a string")

      if param_idx <= last_idx:
        raise ValueError(
          "The order of elements in out_names must follow the kernel's signature"
        )
      last_idx = param_idx
    return triton_call_kwargs

  def __getitem__(self, grid):
    return lambda *args, **kwargs: triton_call(
      *args, kernel=self.kernel, grid=grid, **self.triton_call_kwargs, **kwargs
    )


def kernel(fn=None, **kwargs):
  """
  A decorator that wraps a Triton/Gluon kernel function and returns a JTKernel object
  preserving kwargs to pass to triton_call() launcher.
  Can be used in two ways:
  - @jt.kernel
    @triton.jit  # or @triton.experimental.gluon.jit
    def kernel_func(x, y, z):
      ...
  - @jt.kernel(out_names="z")   # or any other set of key-value pairs for triton_call()
    @triton.jit
    def kernel_func(x, y, z):
      ...
  The JTKernel object can be used to launch the kernel with a given grid using a
  familiar Triton syntax kernel_func[grid](x, y) (Note that output arguments are still
  implicit and should not be passed to the launcher)
  """
  if len(kwargs) == 0 or fn is not None:
    if fn is None:
      raise ValueError("`kernel` decorator must be used with a kernel function")
    return JTKernel(fn, kwargs)

  return functools.partial(JTKernel, triton_call_kwargs=kwargs)


class JTJITFunction:
  """A wrapper around Triton's JITFunction/GluonJITFunction object to isolate the rest
  of the code from Triton's internals and provide a unified interface to bits needed.

  Additionally, it provides a persistence layer to ensure that a certain data doesn't
  have to be re-created on each kernel launch. A user may assume that even when they
  create a new JTJITFunction object for an object previously used JITFunction object,
  the persistent data is reused.

  Since we don't instantiate JTJITFunction objects the way JITFunction objects are
  instantiated and JTJITFunction have a short lives, we have to store the data in the
  very JITFunction object itself. For this we use custom attributes on the JITFunction
  object, prefixed with `_jT_`. The capitalized letter `T` breaks conventions to reduce
  a possibility of clashing with anything else. When we're ready to remove
  `triton_call()` in favor of corresponding method of JTJITFunction to launch a kernel
  similarly to how the upstream Triton does it, we'll be able to remove this since
  JTJITFunction object will have a proper lifetime.
  """

  def __init__(
    self,
    fn: autotuner.Heuristics
    | autotuner.Autotuner
    | triton.JITFunction
    | gl_runtime.GluonJITFunction,
  ):
    # peel off wrappers to get to the JITFunction object
    if isinstance(fn, JTKernel):
      fn = fn.kernel
    if isinstance(fn, JTJITFunction):
      fn = fn.fn
    if isinstance(fn, autotuner.Autotuner):
      fn = fn.fn
    if isinstance(fn, autotuner.Heuristics):
      fn = fn.fn

    if not isinstance(fn, (triton.JITFunction, gl_runtime.GluonJITFunction)):
      raise ValueError(
        "`kernel` must be a Triton's `JITFunction`, `GluonJITFunction`, `Heuristics` or `Autotuner` object."
      )
    self.fn = fn

  @cached_property
  def arg_names(self) -> list[str]:
    """Returns a list of the kernel parameter names in the order they are declared in
    the kernel's signature."""
    # JITFunction::arg_names is deprecated, as per comment
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

  @property  # it's needed only for compiling once, so it's better have this like that
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
    if "_cOmpute_capability" in kwargs:  # capitalization is intended!
      raise ValueError("'_cOmpute_capability' key is reserved in the options/kwargs!")
    kwargs["_cOmpute_capability"] = compute_capability

    if not hasattr(self.fn, "_jT_kernel_cache_key"):
      self.fn._jT_kernel_cache_key = {}

    # two step key resolution is important for supporting callables
    key = triton_runtime_jit.compute_cache_key(
      self.fn._jT_kernel_cache_key, specialization, kwargs
    )
    del kwargs["_cOmpute_capability"]

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
    """Returns a dictionary of the assembly files for the kernel if these were saved
    during compilation.
    Not the best way to share access to that info, but likely no one needs it except for
    our tests. If this needs a better support, we could make a getter to fetch at least
    ttir from a corresponding triton_kernel_call_lib.TritonKernel object instead.
    """
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
    # We have to always make attributes, since we need them for launching the kernel
    attrvals = [x[1] for x in specialization]
    attrs = triton_runtime_jit.find_paths_if(attrvals, lambda _, x: isinstance(x, str))
    attrs = {
      k: backend.parse_attr(triton_runtime_jit.get_iterable_path(attrvals, k))
      for k in attrs
    }
    # attrs keys are tuples of integers representing a path into the (possibly nested)
    # kernel argument list as it is seen by the Triton compiler (i.e. including all
    # constexprs and whatnot, - everything as in the signature)

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
    # This is important part needs to be fully understood.
    # create_function_from_signature() is a very inventive optimization in Triton
    # that one generates once per kernel a so called binder function that simultaneously
    # and very efficiently:
    # 1. assigns default values to all unspecified kernel arguments,
    # 2. assembles a complete dict of parameter_name->value mapping for all arguments
    # 3. runs a correct specialization pipeline for all arguments, respecting all
    # user's annotations.
    # 4. finds key-value elements of kwargs that don't have a match in the kernel
    # signature.
    # The dynamically generated binder function exists purely for performance — it
    # avoids the overhead of branched algo of building the specialization and
    # application of defaults on every kernel launch, while still computing the
    # specialization tuples from the actual runtime argument values.
    if not hasattr(self.fn, "_jT_binder"):
      # In the upstream, a binder is per-device. Since JAX doesn't support mixed
      # execution yet, we can safely ignore it.
      self.fn._jT_binder = triton_runtime_jit.create_function_from_signature(
        self.fn.signature, self.fn.params, backend
      )
    return self.fn._jT_binder

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
    # named_args is complete dict of kernel param names -> actual values.
    # other_kwargs is the kwargs with keys corresponding to kernel parameters removed
    # specialization is a list[tuple[str, Any]], one entry per kernel parameter, that
    # captures two things about each argument at call time:
    #   1. Element 0 — the type string: e.g. "i32", "i64", "*fp16", "*ki32"
    #     (pointer to const), "u1", "fp32", "constexpr", "tensordesc<...>".
    #   2. Element 1 — the specialization value (the "key"): an attribute that may
    #     trigger a separately compiled kernel variant. Currently possible values are:
    #     - None — no runtime specialization (used for bools, floats, do_not_specialize
    #         params)
    #     - "D" — value/pointer is divisible by 16 (alignment hint)
    #     - "" (empty string) — value/pointer is NOT divisible by 16
    #     - The actual Python value itself — for constexpr parameters and int(1)
    #     - A cache_key — for JITCallable (nested kernel) arguments. Note that in that
    #         case the specialization value is a hash string (hopefully!!! lowercase).
    #         Likely by a coincidence this doesn't hurt to `attrs` building later!
    # The specialization values are determined at call time by native_specialize_impl in
    # C++ (triton/python/src/specialize.cc).
    # Specialization serves 3 goals: (1) affect kernel caching key, (2) discover
    # additional vars to be turned into constexprs by the compiler, (3) source for the
    # `attrs` spec.

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
          "implicit output arguments are no longer required for aliased args."
        )

      signature, constexprs = self._make_signature_constexprs(named_args, sigvals)

      backend_fields = _BACKEND_OPTIONS_FIELD_NAMES[gpu_target.backend]
      unrecognized = set(kwargs.keys()) - set(named_args.keys()) - backend_fields
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

      # while triton_kernel_call_lib.TritonKernel() stores ttir internally,
      # unfortunately, it doesn't provide access to it, so we have to store it manually
      # elsewhere for proper testing.

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
  assert isinstance(fn, autotuner.Autotuner)

  # Note this is different .arg_names than in JITFunction!
  # key_idxs = [fn.arg_names.index(k) for k in fn.keys]
  # if any(idx not in key_idxs for idx in scalar_args):
  #  logging.warning(
  #      "Auto-tuning key does not include all scalar arguments. "
  #      "We may perform redundant auto-tuning."
  #  )
  # ^^^ The upstream doesn't care about this. Why should we?

  prev_early_config_prune_fn = fn.early_config_prune

  def prune_configs(configs, named_args, **conf_kwargs):
    pruned_configs = []
    for config in configs:
      if config.pre_hook is not None:
        raise NotImplementedError("`pre_hook` is not supported")

      # Keep the config IFF for every user-provided kwargs(k->v), the config
      # either doesn't specify k at all, or specifies the same value v. This ensures
      # the config is coherent with explicit user choices
      if all(config.kwargs.get(k, v) == v for k, v in kwargs.items()):
        pruned_configs.append(config)
    if prev_early_config_prune_fn is not None:
      pruned_configs = prev_early_config_prune_fn(pruned_configs, named_args)
    return pruned_configs

  fn.early_config_prune = prune_configs
  fn.nargs = named_args
  configs = fn.prune_configs(kwargs)
  return configs


def make_configs_from_heuristics(
  fn: autotuner.Heuristics,
  configs: list[triton.Config],
  orig_kwargs: dict[str, Any],
  named_args: dict[str, Any],
) -> list[triton.Config]:
  """Applies heuristics to the configs and returns the updated configs."""
  assert isinstance(fn, autotuner.Heuristics)

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
) -> list[dict[str, Any]]:
  """Make kernel call parameters for each config."""
  zeroed_outputs_callable = callable(zeroed_outputs)

  kernel_params = []
  for config in configs:
    config_metaparams = {**kwargs, **config.kwargs}
    # filling back backend related config params
    config_metaparams["num_warps"] = config.num_warps
    config_metaparams["num_stages"] = config.num_stages
    config_metaparams["num_ctas"] = config.num_ctas
    if config.maxnreg is None:
      if "maxnreg" in config_metaparams:
        del config_metaparams["maxnreg"]
    else:
      config_metaparams["maxnreg"] = config.maxnreg

    config_grid = normalize_grid(grid, config_metaparams)

    config_zeroed_outputs = (
      zeroed_outputs(config_metaparams) if zeroed_outputs_callable else zeroed_outputs
    )

    # zeroed_params_with_sizes is a dict raw_array_idx -> aval_size_bytes
    # config_zeroed_outputs is a list of ordinal numbers of output arguments
    zeroed_params_with_sizes = {
      i + outputs_offset: aval_size_bytes(ctx.avals_out[i])
      for i in config_zeroed_outputs
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
    pure_out_tree, (JTArray(ctx.avals_out, i) for i in range(num_pure_outputs))
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
  custom_call_target_name,
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

  args, kwargs = deserialize_args_kwargs(ctx, abstract_args, args_kwargs)

  # extending with outputs
  kwargs = add_output_args(jtfu, ctx, out_info, args, kwargs)
  if zeroed_outputs:
    zeroed_outputs = convert_zeroed_outputs(jtfu, kwargs, zeroed_outputs)

  if not isinstance(fn, (triton.JITFunction, gl_runtime.GluonJITFunction)):
    # Note `fn.arg_names` below isn't `JITFunction::arg_names`: Autotuner and
    # Heuristics classes have their own field and it's NOT deprecated.
    named_args = dict(unsafe_zip(fn.arg_names, args))
    # named_args are needed only for Autotuner/Heuristics handling.
    # note that the above construction of named_args is based on positional args only.
    # This is fully coherent with how upstream does it in Autotuner/Heuristics classes.
    # These are genuinely name-value pairs for passed positional arguments only, and not
    # `bound_args` constructed using kernel's signature.

  if isinstance(fn, autotuner.Autotuner):
    # We should unpack autotuner first and heuristics second to ensure we'd
    # eventually get to a correct actual kernel `fn`.
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
    configs = make_configs_from_heuristics(fn, configs, kwargs, named_args)
    fn = fn.fn

  kernel_params = make_kernel_params(
    ctx,
    len(abstract_args),
    kwargs,
    grid,
    zeroed_outputs,
    configs,
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
            # TODO improve above ^^
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
    # named_scalar_args seems to be only used for naming of the autotune call, which is
    # only used for logging, so this doesn't seem too important use-case to care
    # specifically about scalars here.
    # named_scalar_args = {fn.arg_names[i]: v for i, v in scalar_args.items()}
    # TODO agree on that on PR review

    input_output_aliases_with_sizes = tuple(
      (input_idx, output_idx, aval_size_bytes(ctx.avals_in[input_idx]))
      for input_idx, output_idx in operand_output_aliases.items()
    )
    kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
      # f"{kernel_call_name} ({fn.fn.__name__}) {named_scalar_args}",
      f"{kernel_call_name} ({fn.fn.__name__})",
      [(call, str(config)) for call, config in zip(kernel_calls, configs)],
      input_output_aliases_with_sizes,
    )
  else:
    kernel_call = kernel_calls[0]

  call_proto = kernel_call.to_proto(kernel_call_name, serialized_metadata)
  # Note, `operand_output_aliases` parameter uses raw positional indices into the
  # operands that are seen by MLIR custom_call(). Scalars in JAX-Triton are embedded
  # into `backend_config`/`call_proto` and aren't passed as MLIR operands, hence indices
  # of `operand_output_aliases` count only arrays among all inputs.
  rule = jax.ffi.ffi_lowering(
    custom_call_target_name,
    api_version=2,
    backend_config=zlib.compress(call_proto),
    operand_output_aliases=operand_output_aliases,
  )
  return rule(ctx, *abstract_args)


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


batching.primitive_batchers[triton_kernel_call_p] = triton_kernel_call_raise_on_vmap


class ShapeDtype(Protocol):
  @property
  def shape(self) -> tuple[int, ...]: ...

  @property
  def dtype(self) -> np.dtype: ...


def serialize_args_kwargs(jtfu: JTJITFunction, args: list, kwargs: dict):
  """Prepares args and kwargs for passing through JAX's primitive system by separating
  things that must be traced from things that must be passed as is.

  Returns two sequences of traced values, a mapping: array id -> flat array index to
  construct operand_output_aliases mapping, and a tuple of hashable reconstruction info.
  Assumes that a caller adheres to the Triton's rules about argument types, so no
  special checks for hashability are done for performance reasons.

  Note that the function consumes kwargs, i.e. its state is modified in place.
  """
  assert isinstance(jtfu, JTJITFunction), "jtfu must be a JTJITFunction object"
  # JAX's Primitive.bind(*args, **params) has a strict dichotomy:
  # - *args (positional) are dynamic operands — values that participate in JAX's
  # tracing/transformation system (jit, grad, vmap). During JIT compilation, they become
  # abstract values (ShapedArray) and then MLIR SSA values in the lowered IR.
  # - **params (keyword) are static parameters — they must be hashable, are passed
  # through verbatim, and retain their concrete Python values across the entire
  # compilation pipeline.
  # Hence to pass data through JAX's primitive system we must separate things that must
  # be traced from things that must be passed as is. The restriction is that since JAX
  # arrays must be managed by XLA, they must be passed as traced items. The rest could
  # go concrete helping to specialize the kernel properly.
  # We must implement the separation in a way that (1) allows to reconstruct both `args`
  # and `kwargs` back exactly on the other end to process them properly, and (2) all JAX
  # array arguments go into .bind() call in exactly the same order as the kernel expects
  # them (so we don't have to reshuffle them later during the lowering). In that we
  # assume that no JAX array could be specialized to a constexpr for compiling, which
  # holds in the upstream Triton too (small arrays could be passed as tuples/lists
  # though to constexpr annotated params).

  flat_args, args_tree = tree_util.tree_flatten(args, is_leaf=lambda x: x is None)
  # We must let Nones to pass through flattening and `is_leaf` semantic is additive.

  # Since there's usually much more non-vector parameters than vectors, extract
  # vectors from flat_args. Jax explicitly guarantee that `isinstance(x, jax.Array)`
  # check behaves correctly for raw arrays as well as for tracers. For jitting all/most
  # of Python scalars are mapped to ShapedArray avals too, so they are treated as arrays
  # by the check. However, there's no use-case where we might want to pass a scalar as a
  # traceable thing into the kernel call, so we can safely assume a user will mark it as
  # a static argument for jitting (the old implementation relied on that too).
  abs_args = {i: v for i, v in enumerate(flat_args) if isinstance(v, jax.Array)}
  # counting raw arrays to build `operand_output_aliases` mapping
  array_id2idx = {id(v): idx for idx, v in enumerate(abs_args.values())}
  # Note that we assume user won't pass the same array several times in args or in
  # kwargs AND would want to alias it, so not checking this here and below

  static_args = (to_python_type(v) for v in flat_args if not isinstance(v, jax.Array))
  # we're going to pass flat_args as static params now. There's a caveat at least for
  # tl.constexpr() objects: JAX's lowering code seems to require only hashability
  # and correct equality semantics for static objects, and this is true for
  # tl.constexpr() with a nuance: on comparison it returns not a bool True/False, but
  # another tl.constexpr() object with a bool value. So if there's a silly check in JAX
  # that the `(a==b) is True`, or `type(a==b) is bool` - it'll break.

  # Now do the same for kwargs with a caveat - we must traverse them in order of kernel
  # parameters declaration to ensure arrays ends up in a correct order
  def contain_arrays(v):
    if isinstance(v, jax.Array):
      return True
    elif isinstance(v, tuple):  # among pytrees only tuples could be passed as arguments
      return any(contain_arrays(x) for x in v)
    return False

  class _FakeVal: ...

  abs_kwargs_list = []  # only values to abstract
  abs_kwargs_meta = {}  # hashable reconstruction info, keyed by a parameter name
  idx_offset = len(array_id2idx)
  # iterating over param names ensures correct serialization ordering
  for aname in jtfu.arg_names:
    # if kwargs has that parameter AND the value contains an array somewhere (including
    # nested tuples/lists), we handle the whole arg value differently to be able to
    # reconstruct it correctly. Otherwise we just skip the arg to pass it along with
    # other kwargs, such as backend options.
    arg_val = kwargs.get(aname, _FakeVal())
    if contain_arrays(arg_val):
      del kwargs[aname]  # handle the value differently
      if isinstance(arg_val, jax.Array):  # shortcutting if the value is a raw array
        abs_kwargs_list.append(arg_val)
        abs_kwargs_meta[aname] = None
        array_id2idx[id(arg_val)] = idx_offset
        idx_offset += 1
      else:  # do full flattening
        flat_v, the_tree = tree_util.tree_flatten(arg_val)
        abs_v = {i: v for i, v in enumerate(flat_v) if isinstance(v, jax.Array)}
        array_id2idx.update(
          (id(v), idx + idx_offset) for idx, v in enumerate(abs_v.values())
        )
        idx_offset += len(abs_v)
        static_v = (to_python_type(v) for v in flat_v if not isinstance(v, jax.Array))
        abs_kwargs_list.extend(abs_v.values())
        abs_kwargs_meta[aname] = (tuple(static_v), tuple(abs_v.keys()), the_tree)

  # Primitive's bind() doesn't support passing unhashable values as key-value arguments,
  # however, kwargs might also contain backend options, and some of them are dicts. So
  # far, there's only one such option - `extern_libs`, both in AMD and NVIDIA backends.
  # While it might be possible to just use `tree_util.tree_flatten(metaparams)` here,
  # it's too expensive for handling such a rare case. Instead, we do this manually.
  # This is somewhat fragile and might require some maintenance to adapt to new options
  # but the kernel launcher is a hot path, so this seem tolerable.
  if "extern_libs" in kwargs:
    kwargs["extern_libs"] = tuple(kwargs["extern_libs"].items())

  return (
    abs_args.values(),  # intentionally not materialized
    abs_kwargs_list,
    array_id2idx,  # mapping: array id -> flat array index
    (  # first go args info
      args_tree,  # structure of args
      tuple(abs_args.keys()),  # array positions in args
      tuple(static_args),  # leftover args after arrays removed
      # then kwargs info
      tuple(abs_kwargs_meta.items()),
      tuple(kwargs.items()),  # leftover kwargs
    ),
  )


class JTArray:
  _default_alignment = 16

  def __init__(self, avals: list, idx: int):
    self.dtype = get_type_id(avals[idx])
    self.index = idx  # index into a corresponding avals arrays.
    # TODO(sharadmv,zhangqiaorjc): handle differently aligned pointers
    self.data_ptr = lambda: JTArray._default_alignment
    # self.shape = aval.shape  # might be redundant


def deserialize_args_kwargs(
  ctx, abstract_args: list, args_kwargs_meta: tuple
) -> tuple[list, dict]:
  """Reconstructs args and kwargs from the serialized form. Abstract arguments are
  replaced by stubs."""
  args_tree, abs_args_keys, static_args, abs_kwargs_meta, kwargs = args_kwargs_meta
  abs_kwargs_meta = dict[str, None | tuple[tuple, tuple, Any]](abs_kwargs_meta)
  # abstract_args is a concatenation of abs_args.values() and abs_kwargs_list, but
  # abs_kwargs_meta has a more complex inner structure and its length could be smaller
  # than a corresponding portion of abstract_args storing abs_kwargs_list.
  assert len(abstract_args) == len(ctx.avals_in)

  args = list(static_args)
  kwargs_ofs = len(abs_args_keys)
  abs_v = [JTArray(ctx.avals_in, i) for i in range(kwargs_ofs)]
  for i, v in unsafe_zip(abs_args_keys, abs_v):
    args.insert(i, v)
  args = list(tree_util.tree_unflatten(args_tree, args))

  # this reverses additional kwargs modifications by serializer
  assert isinstance(kwargs, tuple), "kwargs must be a tuple here"
  kwargs = dict(kwargs)
  if "extern_libs" in kwargs:
    kwargs["extern_libs"] = dict(kwargs["extern_libs"])

  # now restore the rest of kwargs
  for aname, meta in abs_kwargs_meta.items():
    assert aname not in kwargs
    if meta is None:
      kwargs[aname] = JTArray(ctx.avals_in, kwargs_ofs)
      kwargs_ofs += 1
    else:
      static_v, abs_v_keys, the_tree = meta
      n_keys = len(abs_v_keys)
      abs_v = [JTArray(ctx.avals_in, kwargs_ofs + j) for j in range(n_keys)]
      kwargs_ofs += n_keys
      static_v = list(static_v)
      for k, v in unsafe_zip(abs_v_keys, abs_v):  # assume sorted keys
        static_v.insert(k, v)
      kwargs[aname] = tree_util.tree_unflatten(the_tree, static_v)
  assert kwargs_ofs == len(abstract_args)

  return args, kwargs


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
  """For an :in-spec: element `elm` traverses the `args`/`kwargs` to find all arrays
  addressed by it and updates a list of ShapeDtypeStruct objects describing the arrays
  and a dict mapping from array identifiers to the ShapeDtypeStruct identifiers.
  """
  # this code runs each kernel launch, so proper validation is expensive here, so
  # we skip most of checks allowing it just to fail, should the spec be incorrect.
  # We might check some things later on the go.
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

  def _unpack_arg(_, elm: Any):
    nonlocal shapes, aliases
    if isinstance(elm, jax.Array):
      shape = jax.ShapeDtypeStruct(elm.shape, elm.dtype)
      aid = id(elm)
      if aid in aliases:
        raise ValueError(f"Array {elm} found under path {path} can't be aliased twice")
      aliases[aid] = id(shape)
      shapes.append(shape)
    elif not isinstance(elm, tuple):
      raise ValueError(
        f"Aliased element {elm} under path {path} is not a jax.Array or a tuple of them"
      )

  if isinstance(arg, jax.Array):
    _unpack_arg(None, arg)
  elif not isinstance(arg, tuple):
    raise ValueError(
      f"Aliased element {arg} at path {path} is not a jax.Array or a tuple of them"
    )
  else:
    functools.reduce(_unpack_arg, arg, None)


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
      nonlocal aliases
      _in_spec_to_ShapeDtypeStruct(
        elm, args, kwargs, name2idx, idx2name, aliases, inner_shapes
      )
      inner_shapes = inner_shapes[0] if len(inner_shapes) == 1 else tuple(inner_shapes)
      shapes.append(inner_shapes)

  aliased_shapes = []
  tuple(_make_aliased(e, aliased_shapes) for e in in_spec)
  # aliased_shapes = _make_aliased(in_spec) if in_spec else []

  # TODO perhaps remove it to spare cycles?
  # if it's a dict, validating that out_shapes match to aliased_shapes
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
  # then len(input_output_aliases) is the number of aliased buffers/shapes. Else -
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
  arguments (those have corresponding parameters in the kernel's signature), and a
  sequence of shapes/dtypes of aliased arguments (those don't have separate output
  parameters in the kernel's signature since they reuse some input parameters).
  """
  assert isinstance(jtfu, JTJITFunction), "jtfu must be a JTJITFunction object"
  if isinstance(out_names, (int, str)):
    out_names = (out_names,)
  # a basic validation that should be evicted on compilation for jitting is fine
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
      if frozenset(out_names) != frozenset(out_shape.keys()):
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
  # no use of out_shape below this line

  def _to_ShapeDtypeStruct(a: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(a.shape, a.dtype)

  if aliased_shapes:
    aliased_shapes = tree_util.tree_map(_to_ShapeDtypeStruct, aliased_shapes)
  out_values = tree_util.tree_map(_to_ShapeDtypeStruct, out_values)

  # now finally canonicalizing out_names using the out_values.
  pure_out_shapes: dict[CanonicalKernelArgPath, Sequence[jax.ShapeDtypeStruct]] = {}
  name2idx = jtfu.arg_name_to_index
  for i, outn in enumerate(out_names):
    ospec = _canonicalize_out_spec_element(outn, name2idx)
    pure_out_shapes[ospec] = out_values[i]
  # While ordering of pure_out_shapes here reflects ordering of out_names, it doesn't
  # mean that this ordering will be used for passing arguments to the kernel and
  # producing outputs! There are two things that affect that:
  # First, we'll use `tree_util.tree_flatten()` to serialize compound data structures
  # including output args information, and JAX is built on assumption that the
  # flattening also sorts keys of dictionaries (otherwise it'll break whole Primitive
  # calls caching system). Keys here are tuple(int,...) where ints are kernel signature
  # parameter indices, so it'll make them sorted accordingly. An escape hatch of
  # OrderedDict() could be used here to bypass it, but that won't help, since later
  # we'll make Triton variables for the output arguments by injecting corresponding
  # arrays into `kwargs`. This will also nullify any imposed ordering by using
  # kernel signature ordering. Later upon the kernel launch, we'll iterate over kernel
  # arguments in the kernel signature order too, to properly pass arguments to the
  # kernel. Hence, effectively, there's only one ordering - kernel signature's one.

  return pure_out_shapes, aliased_shapes


def _canonicalize_out_spec_element(
  elm: InOutSpec,
  name2idx: dict[str, int],
) -> CanonicalKernelArgPath:
  orig_name = None
  if isinstance(elm, int):
    path = (elm,)
  elif isinstance(elm, str):
    orig_name = elm
    path = (name2idx[elm],)  # user is responsible for correct indexing
  elif isinstance(elm, tuple):
    prim_idx = elm[0]
    if isinstance(prim_idx, str):
      orig_name = prim_idx
      path = (name2idx[prim_idx], *elm[1:])
    else:
      if not isinstance(prim_idx, int):
        raise ValueError(f"Invalid first element of this out-spec: {elm}")
      path = elm
  else:
    raise ValueError(f"Invalid out-spec element: {elm}")
  # we don't need to dive deeper. We'll just create a variable for Triton with whatever
  # a corresponding element of out_shape has
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
  custom_call_target_name: str = "triton_kernel_call",
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

  You can find much more use examples in the tests directory.

  Kernel outputs and aliasing specification is a mandatory complexity on top of Triton
  to interoperate with JAX properly. To describe it, a concept of "Kernel's signature
  coordinate system" is used - it's a hashable method of addressing an individual
  array in a random (*args,**kwargs) position of kernel's arguments, no matter how
  deeply nested it is. A "coordinate" refers to either a single array or all arrays
  nested into the addressed tuple. It can be:
    - an int: indexes a kernel signature parameter.
    - a string - parameter name in the kernel's signature.
    - a tuple, with [0] being a string, as above, identifies parameter name in the
      kernel's signature; otherwise must be an int indexing a kernel signature
      parameter. All other elements of the tuple must be ints indexing into the
      compound argument corresponding to the parameter mentioned in the [0] tuple's
      element.
      For example: a tuple `("Ptrs", 0, 1)` refers to `Ptrs` parameter of the kernel,
      and expects a corresponding argument to be a tuple, with its 0-th being another
      tuple, whose 1-st element being an array, or a tuple of arrays/tuples.

  `triton_call()` parameters:
    *args: Positional inputs for the Triton kernel. In the kernel signature, purely
      output parameters should go after the last input parameter to be passed as
      a positional argument to `triton_call()`.

    kernel: Triton (e.g. a function decorated with `triton.jit`) or a Gluon kernel.
      All static values should be annotated with `triton.language.constexpr` or
      `triton.experimental.gluon.language.constexpr`. All other Triton annotations are
      also supported.

    out_shape: Description of shapes and dtypes of output parameters of the kernel.
      Could be one of the following:

      (1) a single ShapeDtype-like object (something that has `.shape` and
      `.dtype` attributes) describing a single output parameter of the kernel; or an
      ordered, potentially nested, sequence of ShapeDtype-like objects corresponding to
      a one or many output parameters, where each root-level element of the sequence
      describes one kernel output parameter in its signature (the nesting defines a form
      of the individual argument, for example, `out_shape=((a,b),c)` defines two output
      arguments, the first is a tuple of arrays having shapes and dtypes of arrays a
      and b, and the second is an array having shape and dtype of array c). Ordering of
      elements in the sequence must correspond to the ordering of the output parameters
      in the kernel's signature.

      (2) a dictionary mapping kernel's output parameter names or indices to
      one ShapeDtype-like object or an ordered, potentially nested, sequence of such
      objects. The nested structure of elements in sequences of the dictionary values
      defines the structure of the output argument. For example,
      `out_shape={"a": (x,y), "b": z}` defines that the kernel has two output
      parameters: param named as "a" is a tuple of 2 arrays, borrowing their shapes and
      dtypes from arrays x and y, and "b" is a single `z`-like array.
      If `out_names` is specified, the `out_shape` must have the same keys. Ordering
      of the kernel signature parameters takes precedence over the ordering of the
      dictionary keys.
      If `input_output_aliases` is needed and a dictionary form of `out_shape` is used,
      the `input_output_aliases` could only use a non-dictionary form, enumerating
      input arrays only.

      Contrary to the Triton itself, JAX/XLA must manage memory buffers and organize
      data copying between host and device memory for all array parameters, so
      `out_shape` helps it with providing the necessary information.
      Also due to a JAX custom call API there's currently a restriction associated with
      ordering of input/output parameters in the kernel signature: all purely output
      parameters must reside after the last non-constexpr input parameter in a flattened
      sequence of kernel's signature parameters. Interleaving input and output
      parameters in the signature is not supported.

      If `out_shape` uses a dictionary form, or if `out_names` is provided, then there
      are no other restrictions. Other cases also require that all input arguments not
      specialized as constexprs, are passed strictly as positional `args`. `out_names`
      are inferred from an assumption that output parameters follow the positional
      parameter values passed as `args` to `triton_call()`.

      Note that no matter which form of `out_shape` is used, arguments corresponding to
      strictly output parameters should never be passed to a `triton_call()` as a part
      of `args` or `kwargs`, - these arguments are added implicitly by the launcher. If
      an argument is going to be an input-output argument, set `input_output_aliases`
      accordingly.

    out_names: A sequence of output parameter names.
      This is another way to specify the output parameter names of the kernel. This one
      is useful to employ in conjunction with the `@kernel` decorator.
      If `out_shape` is a dictionary, its keys must be in sync with `out_names`.

    grid: An integer, tuple of up to 3 integers, or a function that returns a
      tuple of up to 3 integers. When `grid` is an integer, `kernel` is
      invoked in `grid`-many parallel executions. When `grid` is a sequence of
      integers, `kernel` is launched in a `prod(grid)`-many parallel execution.
      When `grid` is a function, it is passed `kwargs` and should return a tuple of up
      to 3 integers.

    input_output_aliases: tells JAX/XLA backend which input arguments not only supply
      input data to the kernel, but also serve as output buffers (hence requiring the
      backend to arrange not only `host->device` data copying before kernel launch, but
      also `device->host` data copying after kernel execution). Might take two forms:

      (1 - deprecated): A dictionary mapping an input array coordinate in the kernel's
      signature coordinate space to the `out_shape` sequence coordinate space.
      Important note: if the kernel also has purely output arguments, they must go first
      in the `out_shape` sequence, and aliased buffers must go last in the sequence.
      Interleaving output with in/out argument shapes in `out_shape` isn't supported.
      User is responsible for adhering to the requirement.

      (2 - recommended): an ordered, potentially nested, sequence of input array
      coordinates in the kernel's signature coordinate space. When this form is used
      there should be no corresponding output arguments in the `out_shape`. Shapes and
      dtypes are inferred from referenced input arguments automatically (JAX/XLA doesn't
      support typecasting of aliased buffers anyway, so `out_shape` for aliased buffers
      are always redundant). This could be specifically:
      - a single int, or a string, or a tuple having an int or a string as a first
        element: describes an input array in the kernel's signature coordinate space.
      - a list: ordered, possibly nested, sequence of the above. The nested structure of
        the sequence describes the structure of the corresponding output argument,
        containing aliased buffers as it's returned by the `triton_call()`. For example,
        `input_output_aliases=["out_ptr",[(2,0), "out_ptr2"]]` describes three aliased
        (input-output) buffers: the first is an input array, passed to `out_ptr` kernel
        parameter, the second is an array passed at index 0 of a tuple passed to the
        third positional parameter, and the third is an input array passed for
        `out_ptr2` kernel parameter. These three buffers (assuming all are single
        arrays) are returned from the `triton_call()` as a tuple of two elements: [0] is
        a new array wrapping around the same buffer as was used for `out_ptr` kernel
        parameter, and [1] is a tuple of two new arrays wrapping the other two array
        buffers. If some referenced input argument is not an array, but a possibly
        nested tuple of arrays, all its internal arrays are considered to be aliased,
        and are returned as a flat tuple of arrays by this call.
        Note that tuples can also be used instead of lists, just remember that if a
        tuple's first element is not another iterable, it's always treated as a
        coordinate descriptor, which might be not what you want. For example,
        `input_output_aliases=(0,1)` is always treated as a reference to an array at
        the second position of a tuple passed to the first kernel parameter, but
        `input_output_aliases=[0,1]` references two arrays passed to the first and the
        second kernel parameters respectively. Hence for clarity it's recommended,
        to use tuples only when describing a single input array coordinate.
      Note that if a kernel needs to start working with an initially zeroed read-write
      buffer that should be returned as an output (for example, for doing sparse
      updates), you should designate a purely output argument for it with a proper
      `out_shape` specification, and mention the buffer coordinate in `out_shape` in
      `zeroed_outputs` parameter.

    zeroed_outputs: A sequence of kernel signature coordinates of output arguments, or
      a function taking a dict of metaparameters and returning a sequence of such
      coordinates, for outputs that should be zeroed before the kernel is launched.

      BREAKING: This argument does NOT support zeroing input-output (i.e. aliased
      through `input_output_aliases`) arguments anymore since this breaks the semantic
      of the input-output parameters - with that you can't pass an information in the
      buffer from the host to the kernel, effectively turning an input-output argument
      into a purely output argument. Mind a little of terminological mess here:
      inputs-outputs are considered from the backend's perspective, whose job
      is to manage memory buffers and arrange data copying between host and device
      memory for all array parameters. A kernel could always read from what is called a
      purely output argument above, so if you need the kernel to start working with a
      buffer from a definite zeroed state, just declare the buffer as a purely output
      argument in `out_shape` and mention the buffer's coordinate in `zeroed_outputs`.

      Note that a callable form of `zeroed_outputs` is allowed to modify the
      metaparameters dict passed to it if and only if the modification does not affect
      kernel compilation (for example, through changing parameters affecting kernel
      specialization variant), and that modification is propagated to the
      kernel call. If the modification does affect kernel compilation, the behavior is
      undefined.

    num_warps: The number of warps used to execute the Triton kernel.
    num_stages: The number of stages emitted by the Triton compiler.
    num_ctas: The size of thread blocks per cluster to be used on GPUs with
      compute capabilities >= 9.0. It must be less or equal to 8.
    enable_fp_fusion: Whether to enable floating-point operands fusion for the kernel.
    debug: Prints out intermediate IRs if True for debugging purposes. Also used as a
      the backend options argument.
    serialized_metadata: Arbitrary metadata that will be added into the
      serialized kernel call.

    kwargs: Key-value pairs (num_warps, num_stages, num_ctas, debug and enable_fp_fusion
      arguments are added there automatically) that will be provided to:
      - a `grid` (if it is a function),
      - backend options constructor (only recognized arguments are passed),
      - the Triton kernel as `constexpr` arguments (constexprs must always be scalars)
        or regular runtime arguments.

  Returns:
    Outputs from the Triton kernel. First go all purely output arguments, then all
    input-output arguments.
  """
  # TODO(Arech) improve error reporting and check assumptions violation. We have a ton
  # of assumptions, requiring a certain behavior from a user. For most if not all of the
  # assumptions it's possible to verify them. We just don't do this b/c of performance
  # or other reasons and that hurts user experience. We can have an extensive validation
  # in at least two cases: (1) some kernel launching parameters aren't vary much between
  # all possible call, b/c they reflect certain properties of the kernel, so processing
  # of such parameters could be cached. In that case we can have a proper validation
  # with the uncached run and all subsequent runs will be blazingly fast. (2) we already
  # have a `debug` parameter that is meant precisely for that.

  # Python guarantees the keys don't exist. The original Triton has a single namespace
  # for both constexprs and backend options, and we're doing the same to unify processing
  # We are setting defaults here early, since we use values early.
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
  # now the structure of `pure_out_shapes` would let us reconstruct the output variables
  # for Triton using ctx.avals_out instead of arrays, and the dict keys would let us
  # place them correctly into args/kwargs. And also to properly structure returned
  # outputs from the call

  if input_output_aliases or input_output_aliases == 0:
    aliased_shapes, aliases = make_aliased_shapes(
      jtfu, input_output_aliases, pure_out_shapes, aliased_shapes, args, kwargs
    )
  else:  # quick shortcut
    aliases = {}
    if aliased_shapes:
      raise ValueError("out_shape isn't coherent with input_output_aliases!")

  # structure of `aliased_shapes` would help to return aliased arrays in a proper form,
  # and flattened values are simply the shapes to pass to the .bind() method.
  # `aliases` would help to generate properly array-only indexed
  # `operand_output_aliases` property to use ffi interface
  abs_args, abs_kwargs, array_id2idx, args_kwargs_meta = serialize_args_kwargs(
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
  num_aliased_outputs = len(aliased_shapes_flat)

  if num_pure_outputs + num_aliased_outputs == 0:
    raise ValueError(
      "No outputs specified for the kernel, DCE will eliminate the call entirely"
    )

  out_flat = triton_kernel_call_p.bind(
    *abs_args,
    *abs_kwargs,
    fn=kernel,
    kernel_call_name=name,
    custom_call_target_name=custom_call_target_name,
    # out_shapes must be a flat sequence of shapes of ALL arrays whose result must be
    # returned, i.e. output + input-output=aliased arrays.
    out_shapes=tuple(pure_out_shapes_flat) + tuple(aliased_shapes_flat),
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
    if num_aliased_outputs > 0
    else ()
  )
  ret = tuple(pure_outs.values()) + aliased_outs
  return ret[0] if len(ret) == 1 else ret
