# Copyright 2024 The jax_triton Authors.
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

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import config
from jax import random
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
from triton.compiler import code_generator as code_gen
import triton.language as tl

config.parse_flags_with_absl()


def setUpModule():
  config.update("jax_enable_x64", True)


def tearDownModule():
  config.update("jax_enable_x64", False)


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


def add(x, y, *, kernel=add_kernel, **kwargs):
  if kernel is add_kernel:
    kwargs.setdefault("BLOCK_SIZE", 8)

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
    @triton.jit
    def add_scalar_kernel(x_ptr, y, output_ptr):
      tl.store(output_ptr, tl.load(x_ptr) + y)

    def add_scalar(x, y):
      return jt.triton_call(
          x,
          y,
          kernel=add_scalar_kernel,
          out_shape=jax.ShapeDtypeStruct((), x.dtype),
          grid=1,
      )

    x = jnp.array([1.0])
    np.testing.assert_allclose(add_scalar(x, scalar), x + scalar)

  def test_explicit_compute_capability(self):
    scalar = np.float32(8)

    @triton.jit
    def add_scalar_kernel(x_ptr, y, output_ptr):
      tl.store(output_ptr, tl.load(x_ptr) + y)

    def add_scalar(x, y):
      return jt.triton_call(
          x,
          y,
          kernel=add_scalar_kernel,
          compute_capability=jt.get_compute_capability(0),
          out_shape=jax.ShapeDtypeStruct((), x.dtype),
          grid=1,
      )

    x = jnp.array([1.0])
    np.testing.assert_allclose(add_scalar(x, scalar), x + scalar)

  def test_input_output_aliasing(self):
    @triton.jit
    def add_inplace_kernel(_, n_elements, output_ptr, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(axis=0)
      block_start = pid * BLOCK_SIZE
      offsets = block_start + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      x = tl.load(output_ptr + offsets, mask=mask)
      output = x + 1
      tl.store(output_ptr + offsets, output, mask=mask)

    size = 8
    x = random.normal(random.PRNGKey(0), [size])
    expected = x + 1
    out = jt.triton_call(
        x,
        size,
        kernel=add_inplace_kernel,
        out_shape=x,
        grid=(8,),
        BLOCK_SIZE=1,
        input_output_aliases={0: 0},
    )
    np.testing.assert_allclose(out, expected)

  @parameterized.parameters(False, True)
  def test_zeroed_outputs(self, use_function):
    x, y = create_random_inputs([1000000])
    # We alias `y` with the output, so are performing the add in-place.
    # If we zero the output before the kernel, the result is `x + 0`.
    out = add(
        x,
        y,
        input_output_aliases={1: 0},
        zeroed_outputs=(lambda _: (0,)) if use_function else (0,),
    )
    np.testing.assert_allclose(out, x)

  def test_multiple_outputs(self):
    @triton.jit
    def copy_twice_kernel(a_ptr, x_ptr, y_ptr):
      a = tl.load(a_ptr)
      tl.store(x_ptr, a)
      tl.store(y_ptr, a)

    a = jnp.array([42])
    x, y = jt.triton_call(
        a,
        kernel=copy_twice_kernel,
        out_shape=[a, a],
        grid=(1,),
    )
    np.testing.assert_array_equal(a, x)
    np.testing.assert_array_equal(a, y)

  def test_kernel_cache_equivalent_kernels(self):
    # Create unique JITFunction to avoid conflicts with other tests.
    my_add_kernel = triton.jit(add_kernel.fn)
    fn1 = jax.jit(lambda x, y: add(x, y, BLOCK_SIZE=32, kernel=my_add_kernel))
    fn2 = jax.jit(lambda x, y: add(x, y, BLOCK_SIZE=32, kernel=my_add_kernel))
    fn3 = jax.jit(lambda x, y: add(x, y, BLOCK_SIZE=64, kernel=my_add_kernel))

    x1, y1 = create_random_inputs([42])
    x2, y2 = create_random_inputs([43])

    compile_ttir_inplace = jt.triton_lib.compile_ttir_inplace

    call_count = [0]

    def my_compile(*args, **kwargs):
      call_count[0] += 1
      return compile_ttir_inplace(*args, **kwargs)

    with mock.patch.object(
        jt.triton_lib, "compile_ttir_inplace", new=my_compile
    ):
      _ = fn1(x1, y1)
      self.assertEqual(call_count[0], 1)
      _ = fn2(x2, y2)
      self.assertEqual(call_count[0], 1)  # Second call hits the cache.
      _ = fn3(x1, y1)
      self.assertEqual(call_count[0], 2)  # Third call misses (block size).

  def test_kernel_cache_same_kernel_different_params(self):
    @triton.jit
    def silly_add_kernel(x_ptr, y_ptr, output_ptr):
      pid = tl.program_id(axis=0)
      tl.store(output_ptr + pid, tl.load(x_ptr + pid) + tl.load(y_ptr + pid))

    def silly_add(n):
      x, y = create_random_inputs([n])
      return jt.triton_call(
          x,
          y,
          kernel=silly_add_kernel,
          out_shape=x,
          grid=x.size,
      )

    get_or_create_triton_kernel = jt.triton_lib.get_or_create_triton_kernel

    call_count = [0]

    def my_get_or_create_triton_kernel(*args, **kwargs):
      call_count[0] += 1
      return get_or_create_triton_kernel(*args, **kwargs)

    with mock.patch.object(
        jt.triton_lib,
        "get_or_create_triton_kernel",
        new=my_get_or_create_triton_kernel,
    ):
      _ = silly_add(42)
      self.assertEqual(call_count[0], 1)
      _ = silly_add(42)
      self.assertEqual(call_count[0], 1)  # Second call hits the cache.
      _ = silly_add(43)
      self.assertEqual(call_count[0], 2)  # Third call misses (grid size).

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
    kernel = triton.autotune(autotune_configs, key=("n_elements",))(add_kernel)

    x, y = create_random_inputs([1024])
    expected = x + y
    out = add(x, y, kernel=kernel, input_output_aliases={0: 0})
    np.testing.assert_allclose(out, expected)

  def test_specialization(self):
    do_not_specialize = (
        0,  # a_ptr
        2,  # M
        6,  # stride_ak
        7,  # stride_bk
        11,  # c_ptr
    )
    kernel = triton.jit(do_not_specialize=do_not_specialize)(matmul_kernel.fn)

    m, n, k = 128, 128, 99
    x, y = create_random_inputs([m, k], [k, n])

    with mock.patch.object(code_gen, "ast_to_ttir") as mock_compile:
      try:
        _ = matmul(
            x,
            y,
            kernel=kernel,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_K=32,
            # K_EXACTLY_DIVISIBLE_BY_BLOCK=False,
        )
      except TypeError:
        pass  # Error thrown as the mocked method's return value is invalid.

    mock_compile.assert_called_once()
    specialization = mock_compile.call_args[1]['specialization']

    # Pointers are assumed to divide by 16, as do `M`, `N`, `stride_{bk,cm}`.
    # However, we've marked `a_ptr`, `M`, `stride_bk`, and `c_ptr` as "do not
    # specialize", leaving `b_ptr`, `N`, and `stride_cm`.
    self.assertEqual(specialization.attrs.divisible_by_16, (1, 3, 9))
    # `stride_{ak,bn,cn}` equal 1, but we've marked `stride_ak` as "do not
    # specialize" leaving `stride_{bn,cn}`.
    self.assertEqual(specialization.attrs.equal_to_1, (8, 10))


if __name__ == "__main__":
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
  absltest.main()
