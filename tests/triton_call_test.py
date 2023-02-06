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

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
from jax.config import config
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl
try:
  import torch
except ModuleNotFoundError:
  torch = None

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


def add(x, y, *, block_size):
  return jt.triton_call(
      x,
      y,
      x.size,
      kernel=add_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      grid=triton.cdiv(x.size, block_size),
      BLOCK_SIZE=block_size,
  )


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    M,
    N,
    K,
    c_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
  stride_am = K
  stride_ak = 1
  stride_bk = N
  stride_bn = 1
  stride_cm = N
  stride_cn = 1

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
  for _ in range(0, K, BLOCK_SIZE_K):
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator += tl.dot(a, b)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
  c = accumulator.to(tl.float16)
  offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
  tl.store(c_ptrs, c, mask=c_mask)


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
    out = jax.jit(lambda x, y: add(x, y, block_size=block_size))(x, y)
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
      group_size_m=[8],
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
      group_size_m,
  ):
    # TODO(sharadmv): expose this information in `jaxlib`
    if torch is not None and torch.cuda.get_device_capability() < (7, 0):
      self.skipTest("Matmul only works on GPUs with capability >= sm70")

    x, y = create_random_inputs([m, k], [k, n], dtype=dtype)
    out = jt.triton_call(
        x,
        y,
        m,
        n,
        k,
        kernel=matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype=dtype),
        grid=(triton.cdiv(m, block_size_m) * triton.cdiv(n, block_size_n),),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
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
    out = jax.pmap(lambda x, y: add(x, y, block_size=block_size))(x, y)
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

    out = jt.triton_call(
        x,
        y,
        x.size,
        kernel=add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        BLOCK_SIZE=block_size,
    )
    expected = x + y
    np.testing.assert_allclose(out, expected)

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
    out = jt.triton_call(
        x,
        size,
        kernel=add_inplace_kernel,
        out_shape=x,
        grid=(8,),
        BLOCK_SIZE=1,
        input_output_aliases={0: 0},
    )
    expected = x + 1
    np.testing.assert_allclose(out, expected)


if __name__ == "__main__":
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
  absltest.main()
