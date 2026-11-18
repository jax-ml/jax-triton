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

"""Flash attention example."""

import functools
import math

import jax
from jax import random
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
# pylint: disable=invalid-name,missing-function-docstring
def fused_attention_kernel(
    Q,
    K,
    V,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    H,
    N_CTX,
    sm_scale,
    L,
    M,
    Out,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // H
  off_h = off_hz % H
  q_offset = off_z * stride_qz + off_h * stride_qh
  k_offset = off_z * stride_kz + off_h * stride_kh
  v_offset = off_z * stride_vz + off_h * stride_vh
  o_offset = off_z * stride_oz + off_h * stride_oh
  # initialize offsets
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  offs_d = tl.arange(0, BLOCK_DMODEL)
  off_q = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
  off_k = k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
  off_v = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
  # Initialize pointers to Q, K, V
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  # initialize pointer to m and l
  m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
  acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
  # load q: it will stay in SRAM throughout
  q_mask = offs_m[:, None] < N_CTX
  q = tl.load(q_ptrs, mask=q_mask, other=0.0)
  # loop over k, v and update accumulator
  end_n = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if CAUSAL else N_CTX
  for start_n in range(0, end_n, BLOCK_N):
    # -- compute qk ----
    offs_n_curr = start_n + offs_n
    kv_mask = offs_n_curr < N_CTX
    k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0)
    qk = tl.dot(q, k) * sm_scale
    mask = kv_mask[None, :]
    if CAUSAL:
      mask = mask & (offs_m[:, None] >= offs_n_curr[None, :])
    qk = tl.where(mask, qk, float("-inf"))
    # compute new m
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    # correct old l and acc
    alpha = tl.exp(m_prev - m_curr)
    l_prev *= alpha
    # attention weights
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    acc *= alpha[:, None]
    # update acc
    p = p.to(V.dtype.element_ty)
    v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
    acc += tl.dot(p, v)
    # update m_i and l_i
    l_prev = l_curr
    m_prev = m_curr
    # update pointers
    k_ptrs += BLOCK_N * stride_kn
    v_ptrs += BLOCK_N * stride_vk
  acc = acc / l_prev[:, None]
  # write back l and m
  l_ptrs = L + off_hz * N_CTX + offs_m
  m_ptrs = M + off_hz * N_CTX + offs_m
  tl.store(l_ptrs, l_prev, mask=offs_m < N_CTX)
  tl.store(m_ptrs, m_prev, mask=offs_m < N_CTX)
  # initialize pointers to output
  off_o = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


@functools.partial(jax.jit, static_argnames=["sm_scale", "causal"])
def fused_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    sm_scale: float | None = None,
    causal: bool = False,
) -> jnp.ndarray:
  """Flash attention."""
  if sm_scale is None:
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
  block_size = 64
  grid = (triton.cdiv(q.shape[2], block_size), q.shape[0] * q.shape[1])
  out_type = [
      jax.ShapeDtypeStruct(
          shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=jnp.float32
      ),
      jax.ShapeDtypeStruct(
          shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=jnp.float32
      ),
      jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
  ]

  metaparams = dict(
      BLOCK_M=block_size,
      BLOCK_N=block_size,
      BLOCK_DMODEL=q.shape[-1],
      CAUSAL=causal,
      num_warps=4,
      num_stages=2,
  )
  _, _, output = jt.triton_call(
      q,
      k,
      v,
      *jt.strides_from_shape(q.shape),
      *jt.strides_from_shape(k.shape),
      *jt.strides_from_shape(v.shape),
      *jt.strides_from_shape(q.shape),
      q.shape[1],
      q.shape[2],
      sm_scale,
      kernel=fused_attention_kernel,
      out_type=out_type,
      grid=grid,
      **metaparams
  )
  return output


def main(unused_argv):
  q_key, k_key, v_key = random.split(random.PRNGKey(0), 3)
  B, H, S, D = 2, 3, 1024, 128  # pylint: disable=invalid-name
  q = random.normal(q_key, (B, H, S, D), dtype=jnp.float16)
  k = random.normal(k_key, (B, H, S, D), dtype=jnp.float16)
  v = random.normal(v_key, (B, H, S, D), dtype=jnp.float16)
  print(jax.jit(fused_attention)(q, k, v))


if __name__ == "__main__":
  from absl import app  # pylint: disable=g-import-not-at-top

  app.run(main)
