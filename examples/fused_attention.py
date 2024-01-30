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

"""Flash attention example."""
import functools

import jax
from jax import random
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl


def _strides(shape):
  size = np.prod(shape)
  for s in shape:
    size = size // s
    yield int(size)


@triton.jit
def fused_attention_kernel(
  Q, K, V,
  stride_qz, stride_qh, stride_qm, stride_qk,
  stride_kz, stride_kh, stride_kn, stride_kk,
  stride_vz, stride_vh, stride_vk, stride_vn,
  stride_oz, stride_oh, stride_om, stride_on,
  Z, H, N_CTX,
  L, M,
  Out,
  BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
  BLOCK_N: tl.constexpr,
):
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  # initialize offsets
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  offs_d = tl.arange(0, BLOCK_DMODEL)
  off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
  off_k = off_hz * stride_qh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
  off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
  # Initialize pointers to Q, K, V
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  # initialize pointer to m and l
  m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
  acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
  # load q: it will stay in SRAM throughout
  q = tl.load(q_ptrs)
  # loop over k, v and update accumulator
  for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
    # -- compute qk ----
    k = tl.load(k_ptrs)
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    # compute new m
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    # correct old l
    l_prev *= tl.exp(m_prev - m_curr)
    # attention weights
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    # rescale operands of matmuls
    l_rcp = 1. / l_curr
    p *= l_rcp
    acc *= (l_prev * l_rcp)[:, None]
    # update acc
    p = p.to(tl.float16)
    v = tl.load(v_ptrs)
    acc += tl.dot(p, v)
    # update m_i and l_i
    l_prev = l_curr
    m_prev = m_curr
    # update pointers
    k_ptrs += BLOCK_N * stride_kn
    v_ptrs += BLOCK_N * stride_vk
  # rematerialize offsets to save registers
  start_m = tl.program_id(0)
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  # write back l and m
  l_ptrs = L + off_hz * N_CTX + offs_m
  m_ptrs = M + off_hz * N_CTX + offs_m
  tl.store(l_ptrs, l_prev)
  tl.store(m_ptrs, m_prev)
  # initialize pointers to output
  offs_n = tl.arange(0, BLOCK_DMODEL)
  off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc)

@functools.partial(jax.jit, static_argnames=["sm_scale"])
def fused_attention(q: jnp.ndarray, k: jnp.ndarray,
                    v: jnp.ndarray) -> jnp.ndarray:
  """Flash attention."""
  block_size = 128
  grid = (jt.cdiv(q.shape[2], block_size), q.shape[0] * q.shape[1])
  out_shape = [
      jax.ShapeDtypeStruct(
          shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=jnp.float32),
      jax.ShapeDtypeStruct(
          shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=jnp.float32),
      jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  ]

  metaparams = dict(
      BLOCK_M=block_size,
      BLOCK_N=block_size,
      BLOCK_DMODEL=q.shape[-1],
      num_warps=4,
      num_stages=2)
  _, _, output = jt.triton_call(
      q, k, v,
      *jt.strides_from_shape(q.shape),
      *jt.strides_from_shape(k.shape),
      *jt.strides_from_shape(v.shape),
      *jt.strides_from_shape(q.shape),
      q.shape[0], q.shape[1], q.shape[2],
      kernel=fused_attention_kernel,
      out_shape=out_shape,
      grid=grid,
      **metaparams)
  return output


def main(unused_argv):
  q_key, k_key, v_key = random.split(random.PRNGKey(0), 3)
  B, H, S, D = 2, 3, 1024, 128
  q = random.normal(q_key, (B, H, S, D), dtype=jnp.float16)
  k = random.normal(k_key, (B, H, S, D), dtype=jnp.float16)
  v = random.normal(v_key, (B, H, S, D), dtype=jnp.float16)
  print(jax.jit(fused_attention)(q, k, v))

if __name__ == "__main__":
  from absl import app
  app.run(main)
