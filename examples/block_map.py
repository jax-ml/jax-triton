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

import functools

from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import jax_triton as jt
from jax_triton import pallas as pl
from jax._src.lax.control_flow import for_loop

def matmul_tile(x_tile_ref, y_tile_ref, o_tile_ref):
  x_tile = x_tile_ref[:]
  y_tile = y_tile_ref[:]
  o_tile_ref[:] = jnp.dot(x_tile, y_tile)

def matmul(x, y, *, block_shape):
  l, r = block_shape
  return pl.pallas_call(matmul_tile,
                        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]),
                                                       x.dtype),
                        in_specs=[
                          pl.BlockSpec(lambda i, j: (i, 0), (l, x.shape[1])),
                          pl.BlockSpec(lambda i, j: (0, j), (y.shape[0], r))
                        ],
                        out_specs=pl.BlockSpec(lambda i, j: (i, j), (l, r)),
                        grid=(x.shape[0] // l, y.shape[1] // r),
                        debug=True)(x, y)


@functools.partial(jax.jit, static_argnames=['sm_scale'])
def mha_reference(q, k, v, sm_scale=1.0):
  logits = jnp.einsum('bqhc,bkhc->bhqk', q, k).astype(jnp.float32)
  weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
  return jnp.einsum('bhqk,bkhc->bqhc', weights, v)

def mha_kernel(
    q_tile_ref, k_tile_ref, v_tile_ref,  # Input arrays
    o_tile_ref, tmp_tile_ref, # Output arrays
    *, sm_scale, block_k):
  d_model = q_tile_ref.shape[-1]
  seq_len = k_tile_ref.shape[1]

  q = q_tile_ref[0, :, 0, :]

  m_i = jnp.zeros(q.shape[0], dtype=jnp.float32) - float('inf')
  l_i = jnp.zeros(q.shape[0], dtype=jnp.float32)
  # acc is the buffer where we accumulate the output on sram.
  acc = jnp.zeros(q.shape, dtype=jnp.float32)

  def body(i, refs):
    acc_ref, m_i_ref, l_i_ref = refs
    acc, m_i, l_i = acc_ref[:], m_i_ref[:], l_i_ref[:]
    start_k = pl.multiple_of(i * block_k, block_k)
    span_k = start_k + jnp.arange(block_k)
    k = pl.load(k_tile_ref, (0, span_k, 0, jnp.arange(d_model)))
    p_ij = jnp.zeros([q.shape[0], block_k], dtype=jnp.float32)
    if sm_scale == 1.0:
      p_ij += pl.dot(q, k, trans_b=True)   # [block_q, block_k]
    else:
      p_ij += sm_scale * pl.dot(q, k, trans_b=True)  # [block_q, block_k]
    # Bring closer to XLA:GPU numerics.
    p_ij = p_ij.astype(q.dtype)
    p_ij = p_ij.astype(jnp.float32)
    # -- compute m_ij, p, l_ij
    m_ij = jnp.max(p_ij, axis=1)  # Row max, shape [block_q].
    p_ij = jnp.exp(p_ij - m_ij[:, None])  # Shape [block_q, block_k].
    l_ij = jnp.sum(p_ij, axis=1)  # Shape [block_q].

    # NOTE: Flash attention begins.
    # -- update m_i and l_i
    m_i_new = jnp.maximum(m_i, m_ij)  # Shape [block_q].
    alpha = jnp.exp(m_i - m_i_new)  # Shape [block_q].
    beta = jnp.exp(m_ij - m_i_new)  # Shape [block_q].
    l_i_new = alpha * l_i + beta * l_ij  # Shape [block_q].
    # -- update output accumulator --
    # The two terms in the accumulation of o are processed separately.
    p_scale = beta / l_i_new  # Shape [block_q].
    p_ij = p_ij * p_scale[:, None]  # Shape [block_q].
    # Update the scaling of the output buffer acc.
    acc_scale = l_i / l_i_new * alpha  # Shape [block_q].
    # Compiler bug! Use tmp real quck
    tmp_tile_ref[0, 0, :] =  acc_scale
    acc_scale = tmp_tile_ref[0, 0, :]

    acc = acc * acc_scale[:, None]
    l_i_ref[:] = l_i_new  # Update m_i and l_i for the next block_k.
    m_i_ref[:] = m_i_new
    # # NOTE: Flash attention ends.

    # Add the new block of attention weights.
    v = pl.load(v_tile_ref, (0, span_k, 0, jnp.arange(d_model)))
    acc_ref[()] = acc + jnp.dot(p_ij.astype(q_tile_ref.dtype), v)

  acc, m_i, l_i = for_loop.for_loop(seq_len // block_k, body, (acc, m_i, l_i))
  o_tile_ref[0, :, 0, :] = acc.astype(o_tile_ref.dtype)

def mha(q, k, v, *,
    sm_scale: float = 1.0,
    block_q: int = 128,
    block_k: int = 128,
    num_warps: Optional[int] = None,
    num_stages: int = 1,
    grid=None,
    ):
  batch_size, seq_len, num_heads, head_dim = q.shape
  # Heuristics.
  if grid is None:
    grid = (jt.cdiv(seq_len, block_q), num_heads, batch_size)


  def q_index_map(seq_index, head_index, batch_index):
    return (batch_index, seq_index, head_index, 0)
  def k_index_map(_, head_index, batch_index):
    return (batch_index, 0, head_index, 0)
  def v_index_map(_, head_index, batch_index):
    return (batch_index, 0, head_index, 0)
  def o_index_map(seq_index, head_index, batch_index):
    return (batch_index, seq_index, head_index, 0)
  def tmp_index_map(seq_index, _, __):
    return (0, 0, seq_index * block_q)

  if num_warps is None:
    num_warps = 4 if head_dim <= 64 else 8
  kernel = functools.partial(mha_kernel, sm_scale=sm_scale,
                             block_k=block_k)
  out_shape = [
      jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len),
                           dtype=jnp.float32)
  ]
  out, _ = pl.pallas_call(
      kernel, out_shape,
      in_specs=[
        pl.BlockSpec(q_index_map, (1, block_q, 1, head_dim)),
        pl.BlockSpec(k_index_map, (1, seq_len, 1, head_dim)),
        pl.BlockSpec(v_index_map, (1, seq_len, 1, head_dim)),
      ],
      out_specs=[
        pl.BlockSpec(o_index_map, (1, block_q, 1, head_dim)),
        pl.BlockSpec(tmp_index_map, (1, 1, block_q)),
      ],
      num_warps=num_warps, num_stages=num_stages,
      grid=grid, debug=True)(q, k, v)
  return out

if __name__ == "__main__":
  k1, k2 = random.split(random.PRNGKey(0), 2)
  x = random.normal(k1, (1024, 512))
  y = random.normal(k2, (512, 2048))
  out = matmul(x, y, block_shape=(32, 16))
  ref = jnp.matmul(x, y)
  print(out)
  print()
  print(ref)

  dtype = jnp.float16
  batch, seq_len, n_heads, head_dim = 384, 384, 4, 32
  shape = (batch, seq_len, n_heads, head_dim)

  q_key, k_key, v_key = random.split(jax.random.PRNGKey(0), 3)
  q = random.normal(q_key, shape, dtype=dtype)
  k = random.normal(k_key, shape, dtype=dtype)
  v = random.normal(v_key, shape, dtype=dtype)

  o = mha(q, k, v).block_until_ready()
  o_ref = mha_reference(q, k, v).block_until_ready()
