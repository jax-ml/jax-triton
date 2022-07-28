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

from types import SimpleNamespace
import triton
import triton.language as tl
import jax_triton as jt

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

def _strides(shape):
  all = np.prod(shape)
  for s in shape:
    all = all // s
    yield int(all)

@triton.jit
def fused_attention_kernel(
    Q, K, V,
    TMP, L, M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    Out,
    stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,
    stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kk: tl.constexpr, stride_kn: tl.constexpr,
    stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vk: tl.constexpr, stride_vn: tl.constexpr,
    stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
    Z: tl.constexpr, H: tl.constexpr, N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_qm = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_qm * BLOCK_M + tl.arange(0, BLOCK_M)
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
    t_ptrs = TMP + off_hz * N_CTX + offs_m

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    q = tl.load(q_ptrs)
    for start_n in range(0, start_qm + 1):
        # -- compute qk ----
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)
        qk += tl.where(offs_m[:, None] >= (start_n * BLOCK_N + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        p = p.to(tl.float16)
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        # r_ptrs += BLOCK_N
        l_i = l_i_new
        m_i = m_i_new

    start_qm = tl.program_id(0)
    offs_m = start_qm * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_out = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_out
    tl.store(out_ptrs, acc)
    
def fused_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
  BLOCK = 128
  Lq, Lk = q.shape[-1], k.shape[-2]
  assert Lq == Lk
  grid = lambda _: (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
  out_shape = [
      SimpleNamespace(shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=q.dtype),
      SimpleNamespace(shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=q.dtype),
      SimpleNamespace(shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=q.dtype),
      SimpleNamespace(shape=q.shape, dtype=q.dtype)]
  stride_qz, stride_qh, stride_qm, stride_qk = _strides(q.shape)
  stride_kz, stride_kh, stride_kk, stride_kn = _strides(k.shape)
  stride_vz, stride_vh, stride_vk, stride_vn = _strides(v.shape)
  stride_oz, stride_oh, stride_om, stride_on = _strides(out_shape[-1].shape)
  
  metaparams = dict(
    BLOCK_M=BLOCK,
    BLOCK_N=BLOCK,
    BLOCK_DMODEL=64,
    stride_qz=stride_qz, stride_qh=stride_qh, stride_qm=stride_qm, stride_qk=stride_qk,
    stride_kz=stride_kz, stride_kh=stride_kh, stride_kk=stride_kk, stride_kn=stride_kn,
    stride_vz=stride_vz, stride_vh=stride_vh, stride_vk=stride_vk, stride_vn=stride_vn,
    stride_oz=stride_oz, stride_oh=stride_oh, stride_om=stride_om, stride_on=stride_on,
    Z=q.shape[0], H=q.shape[0], N_CTX=q.shape[0],
    num_warps=4, num_stages=1
  )
  _, _, _, output = jt.triton_call(q, k, v, kernel=fused_attention_kernel,
      out_shape=out_shape, grid=grid, **metaparams)
  return output


q_key, k_key, v_key = random.split(random.PRNGKey(0), 3)
q = random.normal(q_key, (2, 3, 1024, 64), dtype=jnp.float16)
k = random.normal(k_key, (2, 3, 64, 1024), dtype=jnp.float16)
v = random.normal(v_key, (2, 3, 1024, 64), dtype=jnp.float16)
print(fused_attention(q, k, v))
print(jax.jit(fused_attention)(q, k, v))
