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

"""Matrix multiplication example."""
import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
    group_size_m: tl.constexpr,
    activation: tl.constexpr,
):  # pylint: disable=g-doc-args
  """Kernel for computing the matmul C = A x B.

    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
  """
  # -----------------------------------------------------------
  # Map program ids `pid` to the block of C it should compute.
  # This is done in a grouped ordering to promote L2 data reuse
  # See above `L2 Cache Optimizations` section for details
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv(m, block_size_m)
  num_pid_n = tl.cdiv(n, block_size_n)
  num_pid_in_group = group_size_m * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * group_size_m
  group_size_m = min(num_pid_m - first_pid_m, group_size_m)
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m

  # ----------------------------------------------------------
  # Create pointers for the first blocks of A and B.
  # We will advance this pointer as we move in the K direction
  # and accumulate
  # a_ptrs is a block of [block_size_m, block_size_k] pointers
  # b_ptrs is a block of [block_size_k, BLOCK_SIZE_n] pointers
  # see above `Pointer Arithmetics` section for details
  offs_am = pid_m * block_size_m + tl.arange(0, block_size_m)
  offs_bn = pid_n * block_size_n + tl.arange(0, block_size_n)
  offs_k = tl.arange(0, block_size_k)
  a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

  # -----------------------------------------------------------
  # Iterate to compute a block of the C matrix
  # We accumulate into a `[block_size_m, block_size_n]` block
  # of fp32 values for higher accuracy.
  # `accumulator` will be converted back to fp16 after the loop
  accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
  for k in range(0, k, block_size_k):
    # Note that for simplicity, we don't apply a mask here.
    # This means that if K is not a multiple of block_size_k,
    # this will access out-of-bounds memory and produce an
    # error or (worse!) incorrect results.
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    # We accumulate along the K dimension
    accumulator += tl.dot(a, b)
    # Advance the ptrs to the next K block
    a_ptrs += block_size_k * stride_ak
    b_ptrs += block_size_k * stride_bk
  # you can fuse arbitrary activation functions here
  # while the accumulator is still in FP32!
  if activation:
    accumulator = activation(accumulator)
  c = accumulator.to(tl.float16)

  # -----------------------------------------------------------
  # Write back the block of the output matrix C
  offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
  offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)
  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
  tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def relu(x):
  return tl.where(x >= 0, x, 0)


def matmul(a, b, activation=None):
  """Performs a Triton matmul."""
  block_size_m = 128
  block_size_n = 256
  block_size_k = 32
  group_size_m = 8
  m, k = a.shape
  n, _ = b.shape
  out_shape = jax.ShapeDtypeStruct(shape=(m, n), dtype=a.dtype)
  grid = (m //  block_size_m * n // block_size_n,)
  return jt.triton_call(
      a,
      b,
      kernel=matmul_kernel,
      out_shape=out_shape,
      grid=grid,
      num_warps=8,
      num_stages=3,
      m=m,
      n=n,
      k=k,
      stride_am=k,
      stride_ak=1,
      stride_bk=n,
      stride_bn=1,
      stride_cm=n,
      stride_cn=1,
      block_size_m=block_size_m,
      block_size_n=block_size_n,
      block_size_k=block_size_k,
      group_size_m=group_size_m,
      activation=activation)


def main(unused_argv):
  k1, k2 = jax.random.split(jax.random.PRNGKey(0))
  a_val = jax.random.normal(k1, (512, 512), dtype=jnp.float32)
  b_val = jax.random.normal(k2, (512, 512), dtype=jnp.float32)
  print(matmul(a_val, b_val, relu).block_until_ready())
  print(
      jax.jit(matmul, static_argnums=2)(a_val, b_val, relu).block_until_ready())


if __name__ == "__main__":
  from absl import app
  app.run(main)
