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

"""Module containing fused attention forward and backward pass."""
import functools

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import for_loop

import jax_triton as jt
from jax_triton import pallas as pl

def mha_forward_kernel(
    q_ref, k_ref, v_ref,  # Input arrays
    o_ref, # Output
    tmp_ref, # Temporary scratch space to deal with compiler bug
    *residual_refs, # Residual outputs
    sm_scale: float, block_q: int, block_d: int, block_k: int):
  seq_len = q_ref.shape[0]
  start_q = pl.program_id(0)

  # acc is the buffer where we accumulate the output on sram.
  # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
  m_i = jnp.zeros([block_q], dtype=jnp.float32) - float('inf')
  l_i = jnp.zeros([block_q], dtype=jnp.float32)
  # acc is the buffer where we accumulate the output on sram.
  acc = jnp.zeros([block_q, block_d], dtype=jnp.float32)

  # Load q: it will stay in L1 throughout. Indices form a matrix because we
  # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
  # q tile has shape [block_q, block_d], block_d == head_dim.
  q = pl.load(q_ref, (pl.dslice(start_q * block_q, block_q), pl.dslice(None)))
  # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
  # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
  # Here we only loop over blocks of kv to process entire seq_len, the loop over
  # blocks of q is carried out by the grid.
  def body(i, refs):
    acc_ref, m_i_ref, l_i_ref = refs
    acc, m_i, l_i = acc_ref[:], m_i_ref[:], l_i_ref[:]
    start_k = pl.multiple_of(i * block_k, block_k)
    k = pl.load(k_ref, (pl.dslice(start_k, block_k), pl.dslice(block_d)))
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
    p_ij_sub = p_ij - m_ij[:, None]  # Shape [block_q, block_k].
    p_ij = jnp.exp(p_ij_sub)
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
    # Compiler bug! Use tmp real quick

    tmp_idx = (pl.dslice(start_q * block_q, block_q),)
    pl.store(tmp_ref, tmp_idx, acc_scale)
    acc_scale = pl.load(tmp_ref, tmp_idx)

    acc = acc * acc_scale[:, None]
    l_i_ref[:] = l_i_new  # Update m_i and l_i for the next block_k.
    m_i_ref[:] = m_i_new
    # # NOTE: Flash attention ends.

    # Add the new block of attention weights.
    v = pl.load(v_ref, (pl.dslice(start_k, block_k), pl.dslice(block_d)))
    acc_ref[()] = acc + jnp.dot(p_ij.astype(q_ref.dtype), v)
  acc, m_i, l_i = for_loop(seq_len // block_k, body, (acc, m_i, l_i))

  if residual_refs:
    l_ref, m_ref = residual_refs
    pl.store(l_ref, (pl.ds(start_q * block_q, block_q),), l_i)
    pl.store(m_ref, (pl.ds(start_q * block_q, block_q),), m_i)
  # Write output to dram.
  acc = acc.astype(o_ref.dtype)
  pl.store(o_ref, (pl.dslice(start_q * block_q, block_q), pl.dslice(None)), acc)

@functools.partial(jax.custom_vjp, nondiff_argnums=[3, 4, 5, 6, 7, 8, 9, 10, 11])
@functools.partial(jax.jit, static_argnames=["sm_scale", "block_q", "block_k",
                                             "backward_pass_impl",
                                             "num_warps", "num_stages", "grid",
                                             "interpret", "debug"])
def mha(q, k, v,
        sm_scale: float = 1.0,
        block_q: int = 128,
        block_k: int = 128,
        backward_pass_impl: str = "triton",
        num_warps: Optional[int] = None,
        num_stages: int = 1,
        grid=None,
        interpret: bool = False,
        debug: bool = False):
  del backward_pass_impl
  batch_size, seq_len, num_heads, head_dim = q.shape
  block_q = min(block_q, seq_len)
  block_k = min(block_k, seq_len)
  # Heuristics.
  grid_ = grid
  if grid_ is None:
    grid_ = (jt.cdiv(seq_len, block_q), batch_size, num_heads)

  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 4 if head_dim <= 64 else 8
  kernel = functools.partial(mha_forward_kernel, sm_scale=sm_scale,
                             block_q=block_q, block_k=block_k,
                             block_d=head_dim)
  out_shape = [
      jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
      jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len),
                           dtype=jnp.float32)
  ]
  out, _ = pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=[
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
      ],
      out_specs=[
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
      ],
      num_warps=num_warps_,
      num_stages=num_stages,
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_forward")(q, k, v)
  return out

def _mha_forward(q, k, v, sm_scale: float, block_q: int, block_k: int,
                 backward_pass_impl: str,
                 num_warps: Optional[int], num_stages: int, grid: Any,
                 interpret: bool, debug: bool):
  del backward_pass_impl
  batch_size, seq_len, num_heads, head_dim = q.shape
  block_q = min(block_q, seq_len)
  block_k = min(block_k, seq_len)
  # Heuristics.
  grid_ = grid
  if grid_ is None:
    grid_ = (jt.cdiv(seq_len, block_q), num_heads, batch_size)

  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 4 if head_dim <= 64 else 8
  kernel = functools.partial(mha_forward_kernel, sm_scale=sm_scale,
                             block_q=block_q, block_k=block_k,
                             block_d=head_dim)
  out_shape = [
      jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype), # out
      jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), # tmp
                           dtype=jnp.float32),
      jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), # l
                           dtype=jnp.float32),
      jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), # m
                           dtype=jnp.float32)
  ]
  out, _, l, m = pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=[
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
      ],
      out_specs=[
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
      ],
      num_warps=num_warps_,
      num_stages=num_stages,
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_forward")(q, k, v)
  return out, (q, k, v, out, l, m)

def _preprocess_backward_kernel(out_ref, dout_ref, l_ref,
                                new_dout_ref, delta_ref, *,
                                block_q: int):
  pid_m = pl.program_id(0)

  off_m = pl.ds(pid_m * block_q, block_q)
  # load
  o = pl.load(out_ref, (off_m, slice(None))).astype(jnp.float32)
  do = pl.load(dout_ref, (off_m, slice(None))).astype(jnp.float32)
  denom = pl.load(l_ref, (off_m,)).astype(jnp.float32)
  # compute
  do = do / denom[:, None]
  delta = jnp.sum(o * do, axis=1)
  # write-back
  pl.store(new_dout_ref, (off_m, slice(None)),
           do.astype(new_dout_ref.dtype))
  pl.store(delta_ref, (off_m,), delta.astype(delta_ref.dtype))

def _preprocess_backward(out, do, l, block_q: int,
                         debug: bool, interpret: bool):
  batch_size, seq_len, num_heads, head_dim = out.shape
  out_shape = [
      jax.ShapeDtypeStruct(do.shape, do.dtype),
      jax.ShapeDtypeStruct(l.shape, l.dtype),
  ]
  do_scaled, delta = pl.pallas_call(
      functools.partial(_preprocess_backward_kernel, block_q=block_q),
      grid=(jt.cdiv(seq_len, block_q), batch_size, num_heads),
      in_specs=[
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
      ],
      out_specs=[
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
      ],
      num_warps=4,
      num_stages=3,
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_preprocess_backward")(out, do, l)
  return do_scaled, delta

def mha_backward_kernel(
    # Inputs
    q_ref, k_ref, v_ref, out_ref, do_scaled_ref,
    l_ref, m_ref, delta_ref, _,
    # Outputs
    dq_ref, dk_ref, dv_ref,
    *, sm_scale: float,
    block_q: int, block_d: int, block_k: int
):
  del out_ref, l_ref  # Not needed
  seq_len = q_ref.shape[0]

  def outer_loop(start_k, _):

    dv = jnp.zeros([block_k, block_d], dtype=jnp.float32)
    dk = jnp.zeros([block_k, block_d], dtype=jnp.float32)
    k = pl.load(k_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
    v = pl.load(v_ref, (pl.ds(start_k * block_k, block_k), slice(None)))

    def inner_loop(start_q, refs):
      dv_ref, dk_ref = refs
      q = pl.load(q_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
      qk = pl.dot(q, k, trans_b=True)
      qk = qk.astype(q_ref.dtype)
      qk = qk.astype(jnp.float32)
      if sm_scale != 1.0:
        qk *= sm_scale
      m = pl.load(m_ref, (pl.ds(start_q * block_q, block_q),))
      p = jnp.exp(qk - m[:, None])
      do = pl.load(do_scaled_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
      dv_ref[()] += pl.dot(p.astype(do.dtype), do, trans_a=True)
      di = pl.load(delta_ref, (pl.ds(start_q * block_q, block_q),))
      dp = jnp.zeros((block_q, block_k), dtype=jnp.float32) - di[:, None]
      dp = dp + pl.dot(do, v, trans_b=True)
      ds = p * dp
      if sm_scale != 1.0:
        ds = ds * sm_scale
      dk_ref[:] += pl.dot(ds.astype(q_ref.dtype), q, trans_a=True)
      dq = pl.load(dq_ref, (pl.ds(start_q * block_q, block_q),
                            slice(None)), eviction_policy="evict_last")
      dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)
      pl.store(dq_ref, (pl.ds(start_q * block_q, block_q),
                        slice(None)), dq, eviction_policy="evict_last")
    dv, dk = for_loop(jt.cdiv(seq_len, block_q), inner_loop, (dv, dk))
    pl.store(dv_ref, (pl.ds(start_k * block_k, block_k),
                      slice(None)), dv.astype(dv_ref.dtype))
    pl.store(dk_ref, (pl.ds(start_k * block_k, block_k),
                      slice(None)), dk.astype(dk_ref.dtype))
  for_loop(jt.cdiv(seq_len, block_k), outer_loop, ())

def _mha_backward(sm_scale: float, block_q: int, block_k: int,
                  backward_pass_impl: str, num_warps: Optional[int],
                  num_stages: int, grid: Any, interpret: bool,
                  debug: bool, res, do):
  del num_warps, num_stages, grid
  q, k, v, out, l, m = res

  batch_size, seq_len, num_heads, head_dim = q.shape
  block_q = min(block_q, seq_len)
  block_k = min(block_k, seq_len)
  do_scaled, delta = _preprocess_backward(out, do, l, block_q, debug, interpret)

  if backward_pass_impl == "xla":
    return jax.vjp(mha_reference, q, k, v)[1](do)
  elif backward_pass_impl == "triton":
    # We accumulate into dq so we need to initialize it to zeros.
    dq = jnp.zeros(q.shape, jnp.float32)
    out_shapes = [
      jax.ShapeDtypeStruct(dq.shape, dq.dtype),
      jax.ShapeDtypeStruct(k.shape, k.dtype),
      jax.ShapeDtypeStruct(v.shape, v.dtype),
    ]

    grid = (batch_size, num_heads)
    num_warps = 8
    dq, dk, dv = pl.pallas_call(
        functools.partial(mha_backward_kernel, block_q=block_q, block_d=head_dim,
                          block_k=block_k, sm_scale=sm_scale),
        grid=grid,
        out_shape=out_shapes,
        in_specs=[
          pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
          pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
          pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
          pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        ],
        out_specs=[
          pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        ],
        name="mha_backward",
        debug=debug,
        interpret=interpret,
        num_warps=num_warps,
        num_stages=1,
        input_output_aliases={8: 0})(q, k, v, out, do_scaled, l, m, delta, dq)
  else:
    raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
  return dq.astype(q.dtype), dk, dv
mha.defvjp(_mha_forward, _mha_backward)


@functools.partial(jax.jit, static_argnames=['sm_scale'])
def mha_reference(q, k, v, sm_scale=1.0):
  logits = jnp.einsum('bqhc,bkhc->bhqk', q, k).astype(jnp.float32)
  weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
  return jnp.einsum('bhqk,bkhc->bqhc', weights, v)
