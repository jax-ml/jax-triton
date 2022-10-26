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

"""Flash attention example."""
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
    q_ptr,
    k_ptr,
    v_ptr,
    tmp_ptr,
    l_ptr,
    m_ptr,  # NOTE: tmp_ptr is a scratchpad buffer to workaround a compiler bug
    out_ptr,
    stride_qz: tl.constexpr,  # pylint: disable=unused-argument
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,  # pylint: disable=unused-argument
    stride_kh: tl.constexpr,  # pylint: disable=unused-argument
    stride_kk: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_vz: tl.constexpr,  # pylint: disable=unused-argument
    stride_vh: tl.constexpr,  # pylint: disable=unused-argument
    stride_vk: tl.constexpr,
    stride_vn: tl.constexpr,  # pylint: disable=unused-argument
    stride_oz: tl.constexpr,  # pylint: disable=unused-argument
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    z: tl.constexpr,  # pylint: disable=unused-argument
    h: tl.constexpr,  # pylint: disable=unused-argument
    n_ctx: tl.constexpr,
    block_m: tl.constexpr,
    block_dmodel: tl.constexpr,
    block_n: tl.constexpr,
):
  """Flash attention kernel."""
  start_qm = tl.program_id(0)
  off_hz = tl.program_id(1)
  # initialize offsets
  offs_m = start_qm * block_m + tl.arange(0, block_m)
  offs_n = tl.arange(0, block_n)
  offs_d = tl.arange(0, block_dmodel)
  off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[
      None, :] * stride_qk
  off_k = off_hz * stride_qh + offs_n[
      None, :] * stride_kn + offs_d[:, None] * stride_kk
  off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[
      None, :] * stride_qk
  # Initialize pointers to q_ptr, k_ptr, v_ptr
  q_ptrs = q_ptr + off_q
  k_ptrs = k_ptr + off_k
  v_ptrs = v_ptr + off_v
  # initialize pointer to m and l
  t_ptrs = tmp_ptr + off_hz * n_ctx + offs_m

  acc = tl.zeros([block_m, block_dmodel], dtype=tl.float32)
  m_i = tl.zeros([block_m], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([block_m], dtype=tl.float32)

  q = tl.load(q_ptrs)
  for start_n in range(0, start_qm + 1):
    # -- compute qk ----
    k = tl.load(k_ptrs)
    qk = tl.dot(q, k)
    qk += tl.where(offs_m[:, None] >= (start_n * block_n + offs_n[None, :]), 0,
                   float("-inf"))
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
    k_ptrs += block_n * stride_kn
    v_ptrs += block_n * stride_vk
    # r_ptrs += block_n
    l_i = l_i_new
    m_i = m_i_new

  start_qm = tl.program_id(0)
  offs_m = start_qm * block_m + tl.arange(0, block_m)
  # write back l and m
  l_ptrs = l_ptr + off_hz * n_ctx + offs_m
  m_ptrs = m_ptr + off_hz * n_ctx + offs_m
  tl.store(l_ptrs, l_i)
  tl.store(m_ptrs, m_i)
  # initialize pointers to output
  offs_n = tl.arange(0, block_dmodel)
  off_out = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[
      None, :] * stride_on
  out_ptrs = out_ptr + off_out
  tl.store(out_ptrs, acc)


def fused_attention(q: jnp.ndarray, k: jnp.ndarray,
                    v: jnp.ndarray) -> jnp.ndarray:
  """Flash attention."""
  block_size = 128
  grid = (triton.cdiv(q.shape[2], block_size), q.shape[0] * q.shape[1])
  out_shape = [
      jax.ShapeDtypeStruct(
          shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=q.dtype),
      jax.ShapeDtypeStruct(
          shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=q.dtype),
      jax.ShapeDtypeStruct(
          shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=q.dtype),
      jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  ]
  stride_qz, stride_qh, stride_qm, stride_qk = _strides(q.shape)
  stride_kz, stride_kh, stride_kk, stride_kn = _strides(k.shape)
  stride_vz, stride_vh, stride_vk, stride_vn = _strides(v.shape)
  stride_oz, stride_oh, stride_om, stride_on = _strides(out_shape[-1].shape)

  metaparams = dict(
      block_m=block_size,
      block_n=block_size,
      block_dmodel=64,
      stride_qz=stride_qz,
      stride_qh=stride_qh,
      stride_qm=stride_qm,
      stride_qk=stride_qk,
      stride_kz=stride_kz,
      stride_kh=stride_kh,
      stride_kk=stride_kk,
      stride_kn=stride_kn,
      stride_vz=stride_vz,
      stride_vh=stride_vh,
      stride_vk=stride_vk,
      stride_vn=stride_vn,
      stride_oz=stride_oz,
      stride_oh=stride_oh,
      stride_om=stride_om,
      stride_on=stride_on,
      z=q.shape[0],
      h=q.shape[0],
      n_ctx=q.shape[0],
      num_warps=4,
      num_stages=1)
  _, _, _, output = jt.triton_call(
      q,
      k,
      v,
      kernel=fused_attention_kernel,
      out_shape=out_shape,
      grid=grid,
      **metaparams)
  return output


if __name__ == "__main__":
  q_key, k_key, v_key = random.split(random.PRNGKey(0), 3)
  q = random.normal(q_key, (2, 3, 1024, 64), dtype=jnp.float16)
  k = random.normal(k_key, (2, 3, 64, 1024), dtype=jnp.float16)
  v = random.normal(v_key, (2, 3, 1024, 64), dtype=jnp.float16)
  print(fused_attention(q, k, v))
  print(jax.jit(fused_attention)(q, k, v))
