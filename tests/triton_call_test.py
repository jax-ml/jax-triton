# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
from jax.config import config
import jax.numpy as jnp
import numpy as np
import triton
import triton.language as tl
import torch

import jax_triton as jt

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr,
                       block_size: tl.constexpr, n_elements: tl.constexpr):
  pid = tl.program_id(axis=0)  # we use a 1d launch grid so axis is 0
  block_start = pid * block_size
  offsets = block_start + tl.arange(0, block_size)
  mask = offsets < n_elements
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr):
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

def triton_call(*args, **kwargs):
  return jax.jit(lambda *args: jt.triton_call(*args, **kwargs))(*args)


class TritonKernelCallTest(parameterized.TestCase):

  @parameterized.named_parameters(*[
    (f"size_{size}_dtype_{dtype}_blocksize_{block_size}", size, dtype, block_size)
     for size in [1, 2, 5, 8, 100, 256, 1024]
     for dtype in ['int32', 'float32', 'float16', 'int64', 'float64']
     for block_size in [1, 8, 32, 256]
    ])
  def test_add_vectors(self, size, dtype, block_size):

    grid = lambda meta: (size // meta["BLOCK_SIZE"] + 1,)
    k1, k2 = random.split(random.PRNGKey(0), 2)
    if dtype in {"float32", "float16", "float64"}:
      x, y = random.normal(k1, [size], dtype=dtype), random.normal(k2, [size], dtype=dtype)
    elif dtype in {"int32", "int64"}:
      x, y = random.randint(k1, [size], -100, 100, dtype=dtype), random.randint(k2, [size], -100, 100, dtype=dtype)

    out = triton_call(x, y, kernel=add_kernel, out_shape=x,
                      grid=grid, BLOCK_SIZE=block_size, n_elements=size)
    expected = x + y
    np.testing.assert_allclose(out, expected)

  @parameterized.named_parameters(*[
    (f"m_{m}_n_{n}_k_{k}_dtype_{dtype}_bm_{block_size_m}_"
     f"bn_{block_size_n}_bk_{block_size_k}_gm_{group_size_m}", m, n, k, dtype,
     block_size_m, block_size_n, block_size_k, group_size_m)
     for m in [512, 1024]
     for k in [512]
     for n in [512]
     for dtype in ['float32', 'float16']
     for block_size_m in [64, 128]
     for block_size_n in [128, 256]
     for block_size_k in [32]
     for group_size_m in [8]
     if block_size_m < m and block_size_n < n and block_size_k < k
    ])
  def test_matmul(self, m, n, k, dtype, block_size_m, block_size_n,
      block_size_k, group_size_m):

    if torch.cuda.get_device_capability() < (7, 0):
      raise unittest.SkipTest("Matmul only works on GPUs with capability >= sm70")

    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']) * triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    k1, k2 = random.split(random.PRNGKey(0), 2)
    x = random.normal(k1, [m, k], dtype=dtype)
    y = random.normal(k2, [k, n], dtype=dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), dtype=dtype)

    out = triton_call(x, y, kernel=matmul_kernel, out_shape=out_shape,
                      grid=grid, M=m, N=n, K=k,
                      BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n,
                      BLOCK_SIZE_K=block_size_k, GROUP_SIZE_M=group_size_m)
    expected = jnp.matmul(x, y)
    np.testing.assert_allclose(out, expected, atol=0.05, rtol=0.05)


if __name__ == '__main__':
  absltest.main()
