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

"""
Main tests for triton_call()
"""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from copy import deepcopy
import jax
from jax import config, random, tree_util
import jax.numpy as jnp
import jax_triton as jt
import jax_triton.triton_lib as jttl
import numpy as np
import triton
import triton.language as tl
import types

config.parse_flags_with_absl()


def setUpModule():
  config.update("jax_enable_x64", True)


def tearDownModule():
  config.update("jax_enable_x64", False)


class ArgsKwargsTest(parameterized.TestCase):
  def test_serialize_deserialize(self):
    """Test that serialize_args_kwargs() and deserialize_args_kwargs() are inverses of
    each other, caring about the order of kwargs and args."""
    args = [
      1,
      3.14,
      jnp.array([1, 2, 3], dtype=jnp.int8),
      (3, 4),
      "string",
      (jnp.array([5, 6], dtype=jnp.int16),),
      (9, jnp.array([7, 8], dtype=jnp.int32), jnp.array([70, 80], dtype=jnp.float32)),
      (10, (jnp.array([7, 8], dtype=jnp.int64), 11)),
    ]
    kwargs = {  # keys aren't in the order of declaration in the signature
      "b": 3.14,
      "g": (
        9,
        jnp.array([7, 8], dtype=jnp.uint32),
        jnp.array([70, 80], dtype=jnp.float16),
      ),
      "h": (10, (jnp.array([7, 8], dtype=jnp.uint64), 11)),
      "a": 1,
      "e": "string",
      "c": jnp.array([1, 2, 3], dtype=jnp.uint8),
      "d": (3, 4),
      "f": (jnp.array([5, 6], dtype=jnp.uint16),),
    }

    # kwargs are expected to be sorted
    @triton.jit
    def kernel(int_, fl, a1, t1, st, t2, t3, t4, *, a, b, c, d, e, f, g, h):
      pass

    class FakeAvals:
      def __init__(self):
        # extracting only arrays from args sequence in kernel's signature order.
        # This ensures that during ser/des, arrays are ordered in signature's
        # order too, so no further reshuffling to match signature order to call the
        # kernel is needed.
        self.flat = [
          v
          for v in tree_util.tree_flatten((
            args,
            {k: kwargs[k] for k in sorted(kwargs.keys())},  # restore signature order
          ))[0]
          if isinstance(v, jax.Array)
        ]

        # check the core test assumption that types in args are unique and get_type_id()
        # is a bijection.
        unique_types = frozenset(jttl.get_type_id(v) for v in self.flat)
        assert len(unique_types) == len(self.flat), (
          "Duplicate types in the args sequence"
        )

      def __getitem__(self, i):
        return self.flat[i]

      def __len__(self):
        return len(self.flat)

    def _make_expected_array_id2idx(args, kwargs):
      array_id2idx = {}
      for coll in [args, kwargs]:
        flat, _ = tree_util.tree_flatten(coll)
        arrays = [v for v in flat if isinstance(v, jax.Array)]
        l = len(array_id2idx)
        array_id2idx.update({id(v): idx + l for idx, v in enumerate(arrays)})
      return array_id2idx

    kwargs_copy = deepcopy(kwargs)
    expected_array_id2idx = _make_expected_array_id2idx(args, kwargs_copy)

    abs_args, abs_kwargs, array_id2idx, args_kwargs_meta = jttl.serialize_args_kwargs(
      jttl.JTJITFunction(kernel),
      args,
      kwargs_copy,
      # kwargs could be modified, hence must copy
    )

    assert array_id2idx == expected_array_id2idx

    ctx = types.SimpleNamespace(avals_in=FakeAvals())
    dargs, dkwargs = jttl.deserialize_args_kwargs(
      ctx, [*abs_args, *abs_kwargs], args_kwargs_meta
    )

    def assert_correct(a, b):
      if isinstance(a, jax.Array):
        assert jttl.get_type_id(a) == b.dtype
        # can't test actual values here, they were abstracted away
      elif isinstance(a, tuple):
        assert isinstance(b, tuple)
        assert len(a) == len(b)
        for ai, bi in zip(a, b):
          assert_correct(ai, bi)
      else:
        assert isinstance(a, type(b))
        assert a == b

    assert len(args) == len(dargs)
    for ai in range(len(args)):
      assert_correct(args[ai], dargs[ai])

    assert len(kwargs) == len(dkwargs)
    for k in kwargs.keys():
      assert_correct(kwargs[k], dkwargs[k])


@triton.jit
def add_kernel(x_ptr, y_ptr, n_elements, output_ptr, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def add_inplace_kernel(
  x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr, INPLACE_Y: tl.constexpr
):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  if INPLACE_Y:
    tl.store(y_ptr + offsets, output, mask=mask)
  else:
    tl.store(x_ptr + offsets, output, mask=mask)


def add(x, y, *, kernel=add_kernel, **kwargs):
  if kernel is add_kernel or kernel is add_inplace_kernel:
    kwargs.setdefault("BLOCK_SIZE", 8)
  if kernel is add_inplace_kernel or (  # handling autotuner & possibly other wrappers
    hasattr(kernel, "fn") and kernel.fn is add_inplace_kernel
  ):
    kwargs.setdefault("INPLACE_Y", False)

  default_grid = lambda meta: triton.cdiv(x.size, meta["BLOCK_SIZE"])
  return jt.triton_call(
    x,
    y,
    x.size,
    kernel=kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    grid=kwargs.pop("grid", default_grid),
    **kwargs,
  )


@triton.jit
def matmul_kernel(
  a_ptr,
  b_ptr,
  M,
  N,
  K,
  stride_am,
  stride_ak,
  stride_bk,
  stride_bn,
  stride_cm,
  stride_cn,
  c_ptr,
  BLOCK_SIZE_M: tl.constexpr,
  BLOCK_SIZE_N: tl.constexpr,
  BLOCK_SIZE_K: tl.constexpr,
  GROUP_SIZE_M: tl.constexpr,
  K_EXACTLY_DIVISIBLE_BY_BLOCK: tl.constexpr,
):
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
  num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
  num_pid_in_group = GROUP_SIZE_M * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * GROUP_SIZE_M
  group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m

  offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  offs_k = tl.arange(0, BLOCK_SIZE_K)
  a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
  for k_remaining in range(K, 0, -BLOCK_SIZE_K):
    if K_EXACTLY_DIVISIBLE_BY_BLOCK:
      a = tl.load(a_ptrs)
      b = tl.load(b_ptrs)
    else:
      mask = tl.arange(0, BLOCK_SIZE_K) < k_remaining
      a = tl.load(a_ptrs, mask=mask[None, :], other=0.0)
      b = tl.load(b_ptrs, mask=mask[:, None], other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
  c = accumulator.to(tl.float16)
  offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
  tl.store(c_ptrs, c, mask=c_mask)


def matmul(x, y, *, kernel=matmul_kernel, **kwargs):
  m, k = x.shape
  _, n = y.shape

  def grid(meta):
    cdiv = triton.cdiv
    return cdiv(m, meta["BLOCK_SIZE_M"]) * cdiv(n, meta["BLOCK_SIZE_N"])

  return jt.triton_call(
    x,
    y,
    m,
    n,
    k,
    k,  # stride_am
    1,  # stride_ak
    n,  # stride_bk
    1,  # stride_bn
    n,  # stride_cm
    1,  # stride_cn
    kernel=kernel,
    out_shape=jax.ShapeDtypeStruct((m, n), dtype=x.dtype),
    grid=grid,
    GROUP_SIZE_M=8,
    **kwargs,
  )


def create_random_inputs(shape1, shape2=None, *, dtype="float32"):
  if shape2 is None:
    shape2 = shape1

  k1, k2 = random.split(random.PRNGKey(0), 2)
  if dtype in {"float32", "float16", "float64"}:
    x = random.normal(k1, shape1, dtype=dtype)
    y = random.normal(k2, shape2, dtype=dtype)
  elif dtype in {"int32", "int64"}:
    x = random.randint(k1, shape1, -100, 100, dtype=dtype)
    y = random.randint(k2, shape2, -100, 100, dtype=dtype)
  return x, y


GLOBAL_DEFAULT_ARG = 1


class TritonKernelCallTest(parameterized.TestCase):
  @parameterized.product(
    size=[1, 5, 100, 1024],
    dtype=["int32", "float32", "float16", "int64", "float64"],
    block_size=[1, 32, 256],
  )
  def test_add(self, size, dtype, block_size):
    x, y = create_random_inputs([size], dtype=dtype)
    out = jax.jit(lambda x, y: add(x, y, BLOCK_SIZE=block_size))(x, y)
    expected = x + y
    np.testing.assert_allclose(out, expected)

  @parameterized.product(
    m=[512, 1024],
    k=[512],
    n=[512],
    dtype=["float32", "float16"],
    block_size_m=[64, 128],
    block_size_n=[128, 256],
    block_size_k=[32],
  )
  def test_matmul(
    self,
    m,
    n,
    k,
    dtype,
    block_size_m,
    block_size_n,
    block_size_k,
  ):
    if jt.get_compute_capability(0) < 70:
      self.skipTest("Matmul only works on GPUs with capability >= sm70")

    x, y = create_random_inputs([m, k], [k, n], dtype=dtype)
    out = matmul(
      x,
      y,
      BLOCK_SIZE_M=block_size_m,
      BLOCK_SIZE_N=block_size_n,
      BLOCK_SIZE_K=block_size_k,
      K_EXACTLY_DIVISIBLE_BY_BLOCK=k % block_size_k == 0,
    )
    expected = jnp.matmul(x, y)
    np.testing.assert_allclose(out, expected, atol=0.05, rtol=0.05)

  @parameterized.product(
    size=[1, 5, 100, 1024],
    dtype=["int32", "float32", "float16", "int64", "float64"],
    block_size=[1, 32, 256],
  )
  def test_pmap(self, size, dtype, block_size):
    n_devices = jax.local_device_count()
    if n_devices < 2:
      self.skipTest("Not enough devices")

    x, y = create_random_inputs([n_devices, size], dtype=dtype)
    out = jax.pmap(lambda x, y: add(x, y, BLOCK_SIZE=block_size))(x, y)
    expected = x + y
    np.testing.assert_allclose(out, expected)

  @parameterized.parameters("int", "tuple", "function_int", "function_tuple")
  def test_grid_types(self, grid_type):
    size = 8
    block_size = 1
    x, y = create_random_inputs([size])

    if grid_type == "int":
      grid = triton.cdiv(size, block_size)
    elif grid_type == "tuple":
      grid = (triton.cdiv(size, block_size),)
    elif grid_type == "function_int":
      grid = lambda meta: triton.cdiv(size, meta["BLOCK_SIZE"])
    elif grid_type == "function_tuple":
      grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    out = add(x, y, BLOCK_SIZE=block_size, grid=grid)
    expected = x + y
    np.testing.assert_allclose(out, expected)

  @parameterized.parameters(42.0, np.float32(42.0))
  def test_add_float_scalar(self, scalar):
    @jt.kernel
    @triton.jit
    def add_scalar_kernel(x_ptr, y, output_ptr):
      tl.store(output_ptr, tl.load(x_ptr) + y)

    x = jnp.array([1.0])
    np.testing.assert_allclose(add_scalar_kernel[1](x, scalar, out_shape=x), x + scalar)

  @parameterized.product(
    mul=[None, 1, 1.0, 3.14],
    ofs=[None, 1, 1.0, 2.71],
    numel=[1, 5, 257],
    block_size=[1, 32],
  )
  def test_scalar_ordering_1None(self, mul, ofs, numel, block_size):
    """Test that the order of passing scalars doesn't not matter, as well as that
    special values of 1 and None for runtime arguments are handled correctly.
    Since most kernels in tests follow `inputs first, then scalars` scheme, this test
    only needs to check the reverse order of that."""

    @triton.jit
    def affine_kernel(mul, ofs, in_ptr, in_numel, out_ptr, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      start = pid * BLOCK_SIZE
      end = min(start + BLOCK_SIZE, in_numel)
      for i in range(start, end):
        x = tl.load(in_ptr + i)
        if mul is not None:
          x *= mul
        if ofs is not None:
          x += ofs
        tl.store(out_ptr + i, x)

    def affine(mul, ofs, x, BLOCK_SIZE):
      return jt.triton_call(
        mul,
        ofs,
        x,
        x.size,
        kernel=affine_kernel,
        out_shape=x,
        grid=(triton.cdiv(x.size, BLOCK_SIZE),),
        BLOCK_SIZE=BLOCK_SIZE,
      )

    x = jnp.arange(numel, dtype=jnp.float32)
    y = affine(mul, ofs, x, block_size)
    expected = x
    if mul is not None:
      expected *= mul
    if ofs is not None:
      expected += ofs
    np.testing.assert_allclose(y, expected, rtol=2e-07)

  def test_function_arguments(self):
    # mostly taken from Triton with necessary changes and a test without tl.constexpr
    @triton.jit
    def func1():
      return 1

    @triton.jit
    def func2():
      return 2

    @triton.jit
    def func3(x):
      return x

    @triton.jit
    def func4(x, y):
      return x + y

    @triton.jit  # callables are explicitly constexpr
    def kernel(fn_args, Y, fn: tl.constexpr):
      tl.store(Y, fn(*fn_args))

    @triton.jit  # callables aren't annotated (made constexpr automatically)
    def kernel2(fn_args, Y, fn):
      tl.store(Y, fn(*fn_args))

    def launch_kernel(fn, fn_args, kernel=kernel):
      return jt.triton_call(
        fn_args,
        fn=fn,
        kernel=kernel,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        grid=1,
      )

    rets = [None] * 5
    rets[0] = launch_kernel(func1, tuple())
    rets[1] = launch_kernel(func2, tuple())
    rets[2] = launch_kernel(func3, (3,))
    rets[3] = launch_kernel(func4, (3, 4))
    rets[4] = launch_kernel(func1, tuple())
    self.assertEqual(jttl.JTJITFunction(kernel).compiled_kernels_cache_size, 4)
    np.testing.assert_array_equal(rets, [1, 2, 3, 7, 1])

    rets = [None] * 5
    rets[0] = launch_kernel(func1, tuple(), kernel=kernel2)
    rets[1] = launch_kernel(func2, tuple(), kernel=kernel2)
    rets[2] = launch_kernel(func3, (3,), kernel=kernel2)
    rets[3] = launch_kernel(func4, (3, 4), kernel=kernel2)
    rets[4] = launch_kernel(func1, tuple(), kernel=kernel2)
    self.assertEqual(jttl.JTJITFunction(kernel2).compiled_kernels_cache_size, 4)
    np.testing.assert_array_equal(rets, [1, 2, 3, 7, 1])

  def test_jit_function_arg(self):
    @triton.jit
    def mul_jit_function(x, y):
      return x * y

    @triton.jit
    def apply_binary_op(x, combine_op):
      return combine_op(x, x)

    @jt.kernel
    @triton.jit
    def square_kernel_jit_function(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
      offsets = tl.arange(0, BLOCK_SIZE)
      in_data = tl.load(in_ptr + offsets)
      # pass a JITFunction into another JITFunction
      out_data = apply_binary_op(in_data, mul_jit_function)
      tl.store(out_ptr + offsets, out_data)

    BLOCK_SIZE = 16
    x = jnp.full((BLOCK_SIZE,), 3.0)
    expect = jnp.full((BLOCK_SIZE,), 9.0, dtype=x.dtype)

    out = square_kernel_jit_function[(1,)](x, BLOCK_SIZE=BLOCK_SIZE, out_shape=x)
    np.testing.assert_allclose(out, expect)

  def test_explicit_compute_capability(self):
    scalar = np.float32(8)

    @jt.kernel(compute_capability=jt.get_compute_capability(0))
    @triton.jit
    def add_scalar_kernel(x_ptr, y, output_ptr):
      tl.store(output_ptr, tl.load(x_ptr) + y)

    x = jnp.array([1.0])
    np.testing.assert_allclose(add_scalar_kernel[1](x, scalar, out_shape=x), x + scalar)

  def test_single_namespace_for_constexprs_and_backend_options(self):
    """Checks that metaparams of triton_call() are passed to the kernel and to backend
    options. Validates that non-hashable backend options work too."""

    @triton.jit
    def kernel(in_ptr, out_ptr, num_warps: tl.constexpr):
      tl.store(out_ptr, tl.load(in_ptr) * num_warps)

    extern_libs = {"some wild lib": "/path/not/exists/trust_me"}

    def call_kernel(x, mul):
      return jt.triton_call(
        x,
        kernel=kernel,
        out_shape=x,
        grid=1,
        num_warps=mul,
        extern_libs=extern_libs,
      )

    import triton.backends.nvidia.compiler as cb
    import triton.backends.amd.compiler as hb

    # now we need the platform ID, and I didn't find a better way than this ugliness
    backends = jax.extend.backend.backends()
    default = jax.extend.backend.get_backend()
    name = next((n for n, c in backends.items() if c is default))
    # getting relevant backend options object for hooking its initialization method
    OptionsObj = {"cuda": cb.CUDAOptions, "rocm": hb.HIPOptions}[name]

    orig_opts_init = OptionsObj.__post_init__

    my_num_warps = 1

    def my_opts_init(self):
      assert self.num_warps == my_num_warps
      assert self.extern_libs == extern_libs
      # cleaning it up to prevent unexpected effects of error handling
      object.__setattr__(self, "extern_libs", None)
      orig_opts_init(self)

    x = jnp.array([2.5])

    with mock.patch.object(OptionsObj, "__post_init__", new=my_opts_init):
      my_num_warps = 1
      np.testing.assert_allclose(call_kernel(x, my_num_warps), x * my_num_warps)
      my_num_warps = 2
      np.testing.assert_allclose(call_kernel(x, my_num_warps), x * my_num_warps)
      my_num_warps = 4
      np.testing.assert_allclose(call_kernel(x, my_num_warps), x * my_num_warps)

  def test_kernel_cache_equivalent_kernels(self):
    # Create unique JITFunction to avoid conflicts with other tests.
    my_add_kernel = triton.jit(add_kernel.fn)
    fn1 = jax.jit(lambda x, y: add(x, y, BLOCK_SIZE=32, kernel=my_add_kernel))
    fn2 = jax.jit(lambda x, y: add(x, y, BLOCK_SIZE=32, kernel=my_add_kernel))
    fn3 = jax.jit(lambda x, y: add(x, y, BLOCK_SIZE=64, kernel=my_add_kernel))

    jt_kernel = jttl.JTJITFunction(my_add_kernel)
    jt_cache_size = lambda: jt_kernel.compiled_kernels_cache_size
    self.assertEqual(jt_cache_size(), 0)

    x1, y1 = create_random_inputs([42])
    x2, y2 = create_random_inputs([43])

    compile_ttir_inplace = jttl.compile_ttir_inplace

    call_count = [0]

    def my_compile(*args, **kwargs):
      call_count[0] += 1
      return compile_ttir_inplace(*args, **kwargs)

    with mock.patch.object(jttl, "compile_ttir_inplace", new=my_compile):
      _ = fn1(x1, y1)
      self.assertEqual(call_count[0], 1)
      self.assertEqual(jt_cache_size(), 1)

      _ = fn2(x2, y2)
      self.assertEqual(call_count[0], 1)  # Second call hits the cache.
      self.assertEqual(jt_cache_size(), 1)

      _ = fn3(x1, y1)
      self.assertEqual(call_count[0], 2)  # Third call misses (block size).
      self.assertEqual(jt_cache_size(), 2)

  def test_kernel_cache_same_kernel_different_params(self):
    @jt.kernel(out_names="output_ptr")
    @triton.jit
    def silly_add_kernel(x_ptr, y_ptr, output_ptr):
      pid = tl.program_id(axis=0)
      tl.store(output_ptr + pid, tl.load(x_ptr + pid) + tl.load(y_ptr + pid))

    def silly_add(n, dtype="float32"):
      x, y = create_random_inputs([n], dtype=dtype)
      return silly_add_kernel[x.size](x, y, out_shape=x), x, y

    jt_kernel = jttl.JTJITFunction(silly_add_kernel)
    jt_cache_size = lambda: jt_kernel.compiled_kernels_cache_size
    self.assertEqual(jt_cache_size(), 0)

    get_or_create_triton_kernel = jttl.JTJITFunction.get_or_create_triton_kernel

    call_count = [0]

    def my_get_or_create_triton_kernel(*args, **kwargs):
      call_count[0] += 1
      return get_or_create_triton_kernel(*args, **kwargs)

    with mock.patch.object(
      jttl.JTJITFunction,
      "get_or_create_triton_kernel",
      new=my_get_or_create_triton_kernel,
    ):
      ret, x, y = silly_add(42)
      np.testing.assert_array_equal(ret, x + y)
      self.assertEqual(call_count[0], 1)
      self.assertEqual(jt_cache_size(), 1)

      ret, x, y = silly_add(42)
      np.testing.assert_array_equal(ret, x + y)
      self.assertEqual(call_count[0], 1)  # Second call hits the Primitive cache.
      self.assertEqual(jt_cache_size(), 1)  # and the lowering doesn't even run

      ret, x, y = silly_add(43)
      np.testing.assert_array_equal(ret, x + y)
      # Third call differs in grid size and misses the Primitive's cache, but hits
      # the internal kernel cache
      self.assertEqual(call_count[0], 2)
      self.assertEqual(jt_cache_size(), 1)

      ret, x, y = silly_add(42, "int32")
      np.testing.assert_array_equal(ret, x + y)
      self.assertEqual(call_count[0], 3)  # Misses both caches due to a different dtype
      self.assertEqual(jt_cache_size(), 2)

  def test_autotune(self):
    autotune_configs = [
      triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
      triton.Config({"BLOCK_SIZE": 64}, num_warps=1),
      triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
    ]
    kernel = triton.autotune(autotune_configs, key=("n_elements",))(add_kernel)

    x, y = create_random_inputs([1024])
    expected = x + y
    out = add(x, y, kernel=kernel)
    np.testing.assert_allclose(out, expected)

  def test_regression_issue_128(self):
    autotune_configs = [
      triton.Config({"BLOCK_SIZE": 1024}, num_warps=1),
      triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
    ]
    kernel = triton.autotune(autotune_configs, key=("n_elements",))(add_kernel)

    x, y = create_random_inputs([1024])
    expected = x + y

    # Keep alive so each iteration is written to an uninitialized buffer.
    outs = []
    for _ in range(10):
      outs.append(add(x, y, kernel=kernel))
      np.testing.assert_allclose(outs[-1], expected)

  def test_autotune_pre_hook_error(self):
    autotune_configs = [
      triton.Config({"BLOCK_SIZE": 32}, num_warps=1, pre_hook=lambda _: None),
    ]
    kernel = triton.autotune(autotune_configs, key=("n_elements",))(add_kernel)

    x, y = create_random_inputs([1024])
    with self.assertRaises(NotImplementedError):
      _ = add(x, y, kernel=kernel)

  def test_heuristics(self):
    heuristic_returned_values = []

    def heuristic_fn(args):
      heuristic_returned_values.append(args["K"] % args["BLOCK_SIZE_K"] == 0)
      return heuristic_returned_values[-1]

    heuristics = {"K_EXACTLY_DIVISIBLE_BY_BLOCK": heuristic_fn}
    kernel = triton.heuristics(heuristics)(matmul_kernel)

    def do_matmul(m, n, k):
      x, y = create_random_inputs([m, k], [k, n])
      return matmul(
        x,
        y,
        kernel=kernel,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
      )

    _ = do_matmul(m=128, n=128, k=128)
    _ = do_matmul(m=128, n=128, k=144)
    self.assertEqual(heuristic_returned_values, [True, False])

  def test_autotune_with_heuristics(self):
    heuristic_returned_values = []

    def heuristic_fn(args):
      heuristic_returned_values.append(args["K"] % args["BLOCK_SIZE_K"] == 0)
      return heuristic_returned_values[-1]

    heuristics = {"K_EXACTLY_DIVISIBLE_BY_BLOCK": heuristic_fn}
    autotune_configs = [
      triton.Config({"BLOCK_SIZE_K": 32}, num_warps=1),
      triton.Config({"BLOCK_SIZE_K": 64}, num_warps=1),
    ]
    kernel = triton.autotune(autotune_configs, key=("M", "N", "K"))(
      triton.heuristics(heuristics)(matmul_kernel)
    )

    def do_matmul(m, n, k):
      x, y = create_random_inputs([m, k], [k, n])
      return matmul(
        x,
        y,
        kernel=kernel,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
      )

    _ = do_matmul(m=128, n=128, k=128)
    _ = do_matmul(m=128, n=128, k=160)
    self.assertEqual(heuristic_returned_values, [True, True, True, False])

  def test_heuristics_does_not_modify_autotune_configs(self):
    def heuristic_fn(args):
      return args["K"] % args["BLOCK_SIZE_K"] == 0

    heuristics = {"K_EXACTLY_DIVISIBLE_BY_BLOCK": heuristic_fn}
    autotune_config = triton.Config({"BLOCK_SIZE_K": 32}, num_warps=1)
    kernel = triton.autotune([autotune_config], key=("M", "N", "K"))(
      triton.heuristics(heuristics)(matmul_kernel)
    )

    def do_matmul(m, n, k):
      x, y = create_random_inputs([m, k], [k, n])
      return matmul(
        x,
        y,
        kernel=kernel,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
      )

    _ = do_matmul(m=128, n=128, k=128)
    self.assertEqual(autotune_config.kwargs, {"BLOCK_SIZE_K": 32})

  def test_autotune_with_input_output_aliasing(self):
    autotune_configs = [
      triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
      triton.Config({"BLOCK_SIZE": 64}, num_warps=1),
    ]
    kernel = triton.autotune(autotune_configs, key=("n_elements",))(add_inplace_kernel)

    x, y = create_random_inputs([1024])
    expected = x + y
    out = add(x, y, kernel=kernel, input_output_aliases={0: 0})
    np.testing.assert_allclose(out, expected)

  def test_kernel_in_thread(self):
    # inspired by Triton's test

    # Test calling in a new thread sets a valid device context
    buf = jnp.zeros((38016 * 1024,), dtype=jnp.float32)

    @jt.kernel(out_names="out")
    @triton.jit
    def _kernel(P, BLOCK: tl.constexpr, out):
      pid = tl.program_id(0).to(tl.int64)
      offset = pid * BLOCK + tl.arange(0, BLOCK)

      p = tl.load(P + offset)
      tl.store(out + offset, p + 1)

    def call_triton():
      nonlocal buf
      N = buf.size
      grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
      out = _kernel[grid](buf, BLOCK=1024, out_shape=buf)
      np.testing.assert_array_equal(buf + 1, out)

    from concurrent.futures import ThreadPoolExecutor

    call_triton()
    with ThreadPoolExecutor(1) as pool:
      future = pool.submit(call_triton)
      future.result()

  def test_autodiff_exception(self):
    x, y = create_random_inputs([10, 100], dtype="float32")
    with self.assertRaisesRegex(
      NotImplementedError,
      r"jax_triton.triton_call does not support automatic differentiation.*"
      r"jax\.custom_jvp or jax\.custom_vjp.*",
    ):
      jax.grad(lambda x, y: jnp.sum(add(x, y, BLOCK_SIZE=32)))(x, y)

  def test_batching_exception(self):
    x, y = create_random_inputs([10, 100], dtype="float32")
    with self.assertRaisesRegex(
      NotImplementedError,
      r"jax_triton.triton_call does not support batching.*"
      r"jax\.custom_batching\.custom_vmap.*",
    ):
      jax.vmap(lambda x, y: add(x, y, BLOCK_SIZE=32))(x, y)

  def test_memory_leak(self):
    # inspired by Triton's test

    @jt.kernel(out_names="out_ptr0")
    @triton.jit
    def kernel(in_ptr0, xnumel, XBLOCK: tl.constexpr, out_ptr0):
      xnumel = 10
      xoffset = tl.program_id(0) * XBLOCK
      xindex = xoffset + tl.arange(0, XBLOCK)[:]
      xmask = xindex < xnumel
      x0 = xindex
      tmp0 = tl.load(in_ptr0 + (x0), xmask)
      tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)

    import gc
    import tracemalloc

    tracemalloc.start()
    try:
      inp = random.normal(random.key(0), (10,))
      out = kernel[(10,)](inp, 10, XBLOCK=16, out_shape=inp)
      gc.collect()
      begin, _ = tracemalloc.get_traced_memory()
      for _ in range(200):  # originally was 100
        out = kernel[(10,)](inp, 10, XBLOCK=16, out_shape=inp)
      del out
      gc.collect()
      end, _ = tracemalloc.get_traced_memory()
      assert end - begin < 71000  # originally this was 30000, but we add a bunch
      # of data into a kernel object and there's JAX's internal caching in the Primitive
      # system on top of that, so +41k seem reasonable (it was estimated with the
      # original 100 reps in the loop above, then the range was set to 200).
      # Note that the margin is just about 100 bytes on my system, so it's ok if on the
      # other system it's slightly different (i.e. a bit bigger and causes a failure,
      # - then just update the threshold above).
    finally:
      tracemalloc.stop()

  @parameterized.product(
    signed=[False, True],
    width=[8, 16, 32, 64, 1],
  )
  def test_int_annotation(self, signed, width):
    # Inspired by an original Triton test.
    # We could have similar for float, but it seem redundant as now we let the original
    # Triton code do the job and if ints work - all other should work too.
    if width == 1 and signed:
      return

    def annotated_function(return_type=None, **arg_types):
      """A decorator to add annotations to a function."""

      def decorator(func):
        func.__annotations__ = {**arg_types, "return": return_type}
        return func

      return decorator

    @jt.kernel
    @triton.jit
    @annotated_function(v=f"tl.{'' if signed else 'u'}int{width}")
    def _kernel(v, X):
      tl.store(X + v, v)

    _kernel[(1,)](3, out_shape=jnp.zeros(1), _store_asm=True)

    asm = jttl.JTJITFunction(_kernel).asm
    assert len(asm) == 1
    ttir = next(iter(asm.values())).get("ttir")

    pfx = "si" if signed else "ui"
    if not signed and width < 64:
      assert "arith.extui %v" in ttir
    assert f"%v: i{width}" in ttir
    assert f"arith.{pfx}tofp" in ttir

  def test_err_constexpr_and_do_not_specialize(self):
    # again inspired by the original test; essentially verifies that our code doesn't
    # break `do_not_specialize` functionality. It's somewhat related to the previous,
    # as for this we just rely on the original Triton processing and this tests that
    # this isn't broken.
    @jt.kernel
    @triton.jit(do_not_specialize=["N"])
    def kernel(out_ptr, N: tl.constexpr):
      pass

    with self.assertRaisesRegex(
      triton.compiler.errors.CompilationError,
      r"N marked as constexpr and listed in do_not_specialize",
    ):
      kernel[(1,)](5, out_shape=jnp.zeros(1))  # out_shape is needed to prevent DCE

  def test_kernel_default_arg(self):
    # inspired by an original Triton test
    global GLOBAL_DEFAULT_ARG

    @jt.kernel
    @triton.jit
    def kernel(X, i: tl.constexpr = GLOBAL_DEFAULT_ARG):
      tl.store(X, i)

    x = kernel[(1,)](out_shape=jnp.zeros(1))
    assert x == jnp.ones_like(x)

    # Changing the global variable should not change the default argument in
    # `kernel`.  That value gets set at the time the function is declared.
    GLOBAL_DEFAULT_ARG = 2
    x = kernel[(1,)](out_shape=jnp.zeros(1))
    assert x == jnp.ones_like(x)

    assert jttl.JTJITFunction(kernel).compiled_kernels_cache_size == 1

  def test_readme(self):
    """Code for readme example, basically an integration test of many features at once,
    including not otherwise tested passing of strings and precise partial aliasing +
    pure outputs"""
    from triton.language.extra import libdevice
    from typing import NamedTuple
    import time

    class Function(NamedTuple):
      fn: tl.constexpr
      captured: tuple

    @triton.jit
    def func1(x_ptr, y_ptr: tl.const, SIZE: tl.constexpr):
      off = tl.arange(0, SIZE)
      x = tl.load(x_ptr + off)
      y = tl.load(y_ptr + off)
      x1 = libdevice.sin(x)
      x2 = libdevice.cos(x)
      z = x1 * x1 + x2 * x2
      tl.store(x_ptr + off, z)
      y = libdevice.asin(y) + libdevice.acos(y)
      return z, libdevice.floor(2 * y)

    @triton.jit
    def floor_of_func(values, SIZE: tl.constexpr, FUNC_NAME: tl.constexpr):
      off = tl.arange(0, SIZE)
      return libdevice.floor(getattr(libdevice, FUNC_NAME)(values))

    @triton.jit
    def aggregate(Ptrs):
      z = tl.zeros([], tl.float32)
      for i in tl.static_range(len(Ptrs)):
        z += Ptrs[i]
      return z

    @jt.kernel
    @triton.jit
    def kernel(capture, out_ptr, SIZE: tl.constexpr, FUNC_NAME: tl.constexpr):
      off = tl.arange(0, SIZE)
      t1, t2 = capture.fn(*capture.captured, SIZE=SIZE)
      t3 = floor_of_func(t1, SIZE=SIZE, FUNC_NAME=FUNC_NAME)
      t4 = t2 * t3
      result = aggregate((t4, t4 * t4)).to(tl.int32)
      tl.store(out_ptr + off, result)

    size = 8
    k1, k2 = random.split(random.key(time.perf_counter_ns()), 2)
    x = random.uniform(k1, (size,), dtype=jnp.float32)
    y = random.uniform(k2, (size,), dtype=jnp.float32)

    fn = Function(func1, (x, y))
    out, x = kernel[(1,)](
      fn,  # essentially a tuple of (func_name, (2 arrays in a subtuple))
      SIZE=size,
      FUNC_NAME="exp",
      out_shape=jnp.zeros(size, dtype=jnp.int32),
      input_output_aliases=("capture", 1, 0),  # a path inside `capture` argument, 0th
      # element of 1st subtuple, i.e. `x`. Note that since this is a tuple, it
      # references just a single element (or all its embedded arrays if it's a tuple,
      # but in this invocation it references array `x`)
    )
    np.testing.assert_array_equal(out, jnp.full((size,), 42, dtype=jnp.int32))
    assert out.dtype == jnp.int32
    np.testing.assert_allclose(x, jnp.full((size,), 1.0, dtype=jnp.float32), rtol=5e-7)
    assert x.dtype == jnp.float32


if __name__ == "__main__":
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
  absltest.main()
