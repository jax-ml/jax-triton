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
from jax import lax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import for_loop
import jax_triton as jt

from jax_triton import pallas as pl


def mha_kernel(
    q_ref, k_ref, v_ref,  # Input arrays
    o_ref, tmp_ref, # Output arrays
    *, sm_scale, block_q, block_d, block_k):
  start_q = pl.program_id(0)
  off_h = pl.program_id(1)  # int in [0, num_heads)
  off_b = pl.program_id(2)  # int in [0, batch_size)

   # Each thread block processes a block of block_q queries.
  span_q = start_q * block_q + jnp.arange(block_q)
  # Model dimension is read in one op. Model dimension must be power of 2.
  span_d = jnp.arange(0, block_d)  # block_d == head_dim.

  tmp_idxs = (
      lax.broadcast_in_dim(off_b, (block_q,), ()),
      lax.broadcast_in_dim(off_h, (block_q,), ()),
      span_q)

  # acc is the buffer where we accumulate the output on sram.
  # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
  m_i = jnp.zeros([block_q], dtype=jnp.float32) - float('inf')
  l_i = jnp.zeros([block_q], dtype=jnp.float32)
  # acc is the buffer where we accumulate the output on sram.
  acc = jnp.zeros([block_q, block_d], dtype=jnp.float32)

  # Load q: it will stay in L1 throughout. Indices form a matrix because we
  # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
  # q tile has shape [block_q, block_d], block_d == head_dim.
  q_idxs = (
      lax.broadcast_in_dim(off_b, (block_q, block_d), ()),
      lax.broadcast_in_dim(span_q, (block_q, block_d), (0,)),
      lax.broadcast_in_dim(off_h, (block_q, block_d), ()),
      lax.broadcast_in_dim(span_d, (block_q, block_d), (1,)))
  q = q_ref[q_idxs]
  # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
  # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
  # Here we only loop over blocks of kv to process entire seq_len, the loop over
  # blocks of q is carried out by the grid.
  def body(i, refs):
    acc_ref, m_i_ref, l_i_ref = refs
    acc, m_i, l_i = acc_ref[:], m_i_ref[:], l_i_ref[:]
    start_k = pl.multiple_of(i * block_k, block_k)
    span_k = start_k + jnp.arange(0, block_k)
    k_idxs = (
        lax.broadcast_in_dim(off_b, (block_k, block_d), ()),
        lax.broadcast_in_dim(span_k, (block_k, block_d), (0,)),
        lax.broadcast_in_dim(off_h, (block_k, block_d), ()),
        lax.broadcast_in_dim(span_d, (block_k, block_d), (1,)))
    k = k_ref[k_idxs]
    p_ij = jnp.zeros([block_q, block_k], dtype=jnp.float32)
    if sm_scale == 1.0:
      p_ij += pl.dot(q, k, trans_b=True)   # [block_q, block_k]
    else:
      p_ij += sm_scale * pl.dot(q, k, trans_b=True)  # [block_q, block_k]
    # Bring closer to XLA:GPU numerics.
    p_ij = p_ij.astype(q_ref.dtype)
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
    tmp_ref[tmp_idxs] =  acc_scale
    acc_scale = tmp_ref[tmp_idxs]

    acc = acc * acc_scale[:, None]
    l_i_ref[:] = l_i_new  # Update m_i and l_i for the next block_k.
    m_i_ref[:] = m_i_new
    # # NOTE: Flash attention ends.

    # Add the new block of attention weights.
    v = v_ref[k_idxs]
    acc_ref[()] = acc + jnp.dot(p_ij.astype(q_ref.dtype), v)

  acc, m_i, l_i = for_loop(seq_len // block_k, body, (acc, m_i, l_i))
  start_q = pl.program_id(0)
  span_q = start_q * block_q + jnp.arange(block_q)
  # Write output to dram.
  span_d = jnp.arange(block_d)
  acc = acc.astype(o_ref.dtype)
  o_idxs = (
      lax.broadcast_in_dim(off_b, (block_q, block_d), ()),
      lax.broadcast_in_dim(span_q, (block_q, block_d), (0,)),
      lax.broadcast_in_dim(off_h, (block_q, block_d), ()),
      lax.broadcast_in_dim(span_d, (block_q, block_d), (1,)))
  o_ref[o_idxs] = acc

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

  if num_warps is None:
    num_warps = 4 if head_dim <= 64 else 8
  kernel = functools.partial(mha_kernel, sm_scale=sm_scale,
                             block_q=block_q, block_k=block_k,
                             block_d=head_dim)
  out_shape = [
      jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len),
                           dtype=jnp.float32)
  ]
  out, _ = pl.pallas_call(kernel, num_warps=num_warps, num_stages=num_stages,
                          grid=grid, out_shapes=out_shape, debug=True)(q, k, v)
  return out

@functools.partial(jax.jit, static_argnames=['sm_scale'])
def mha_reference(q, k, v, sm_scale=1.0):
  logits = jnp.einsum('bqhc,bkhc->bhqk', q, k).astype(jnp.float32)
  weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
  return jnp.einsum('bhqk,bkhc->bqhc', weights, v)

if __name__ == "__main__":
  dtype = jnp.float16
  batch, seq_len, n_heads, head_dim = 384, 384, 4, 32
  shape = (batch, seq_len, n_heads, head_dim)

  q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(q_key, shape, dtype=dtype)
  k = jax.random.normal(k_key, shape, dtype=dtype)
  v = jax.random.normal(v_key, shape, dtype=dtype)

  o = mha(q, k, v)
  o.block_until_ready()
  o_ref = mha_reference(q, k, v)
  print(o.block_until_ready())
