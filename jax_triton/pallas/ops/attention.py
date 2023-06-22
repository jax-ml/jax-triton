# Copyright 2023 The jax_triton Authors.
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
from jax import lax

import jax_triton as jt
from jax_triton import pallas as pl

def mha_forward_kernel(
    q_ref, k_ref, v_ref,  # Input arrays
    o_ref, # Output
    *residual_refs, # Residual outputs
    sm_scale: float, causal: bool,
    block_q: int, block_d: int, block_k: int):
  seq_len = q_ref.shape[0]
  start_q = pl.program_id(0)

  # acc is the buffer where we accumulate the output on sram.
  # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
  m_i = jnp.zeros(block_q, dtype=jnp.float32) - float('inf')
  l_i = jnp.zeros(block_q, dtype=jnp.float32)
  # acc is the buffer where we accumulate the output on sram.
  acc = jnp.zeros((block_q, block_d), dtype=jnp.float32)

  # Load q: it will stay in L1 throughout. Indices form a matrix because we
  # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
  # q tile has shape [block_q, block_d], block_d == head_dim.
  q = pl.load(q_ref, (pl.dslice(start_q * block_q, block_q), pl.dslice(None)))
  # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
  # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
  # Here we only loop over blocks of kv to process entire seq_len, the loop over
  # blocks of q is carried out by the grid.
  def body(start_k, carry):
    acc, m_prev, l_prev = carry

    k = pl.load(k_ref, (pl.dslice(start_k * block_k, block_k), slice(None)))
    qk = jnp.zeros([block_q, block_k], dtype=jnp.float32)
    qk += pl.dot(q, k.T)   # [block_q, block_k]
    if sm_scale != 1.:
      qk *= sm_scale # [block_q, block_k]

    if causal:
      span_q = start_q * block_q + jnp.arange(block_q)
      span_k = start_k * block_k + jnp.arange(block_k)
      qk = jnp.where(span_q[:, None] >= span_k[None, :], qk, float('-inf'))
    # Bring closer to XLA:GPU numerics.
    qk = qk.astype(q_ref.dtype)
    qk = qk.astype(jnp.float32)
    m_curr = jnp.maximum(jnp.max(qk, axis=1), m_prev)
    l_prev *= jnp.exp(m_prev - m_curr)
    p = jnp.exp(qk - m_curr[:, None])
    l_curr = jnp.sum(p, axis=1) + l_prev

    l_rcp = 1. / l_curr
    p = p * l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = p.astype(jnp.float16)

    v = pl.load(v_ref, (pl.dslice(start_k * block_k, block_k), pl.dslice(block_d)))
    acc = acc + pl.dot(p.astype(v.dtype), v)
    return acc, m_curr, l_curr
  if causal:
    upper_bound = lax.div(block_q * start_q, block_k) + 1
  else:
    upper_bound = jt.cdiv(seq_len, block_k)
  acc, m_i, l_i = lax.fori_loop(0, upper_bound, body,
                                (acc, m_i, l_i))

  if residual_refs:
    l_ref, m_ref = residual_refs
    pl.store(l_ref, (pl.ds(start_q * block_q, block_q),), l_i)
    pl.store(m_ref, (pl.ds(start_q * block_q, block_q),), m_i)
  # Write output to dram.
  acc = acc.astype(o_ref.dtype)
  pl.store(o_ref, (pl.dslice(start_q * block_q, block_q), pl.dslice(None)), acc)

@functools.partial(jax.custom_vjp, nondiff_argnums=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
@functools.partial(jax.jit, static_argnames=["sm_scale", "causal", "block_q", "block_k",
                                             "backward_pass_impl",
                                             "num_warps", "num_stages", "grid",
                                             "interpret", "debug"])
def mha(q, k, v,
        sm_scale: float = 1.0,
        causal: bool = True,
        block_q: int = 128,
        block_k: int = 128,
        backward_pass_impl: str = "triton",
        num_warps: Optional[int] = None,
        num_stages: int = 2,
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
                             block_d=head_dim,
                             causal=causal)
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  return pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=[
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
      ],
      out_specs=pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
      num_warps=num_warps_,
      num_stages=num_stages,
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_forward")(q, k, v)

def _mha_forward(q, k, v, sm_scale: float, causal: bool, block_q: int,
                 block_k: int, backward_pass_impl: str, num_warps: Optional[int],
                 num_stages: int, grid: Any, interpret: bool, debug: bool):
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
                             causal=causal, block_q=block_q, block_k=block_k,
                             block_d=head_dim)
  out_shape = [
      jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype), # out
      jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), # l
                           dtype=jnp.float32),
      jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), # m
                           dtype=jnp.float32)
  ]
  out, l, m = pl.pallas_call(
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

def mha_backward_kernel_dq(
    # Inputs
    q_ref, k_ref, v_ref, out_ref, do_scaled_ref,
    l_ref, m_ref, delta_ref, _,
    # Outputs
    dq_ref,
    *, sm_scale: float, causal: bool,
    block_q: int, block_d: int, block_k: int
):
  del out_ref, l_ref  # Not needed
  seq_len = q_ref.shape[0]

  start_q = pl.program_id(0)
  q = pl.load(q_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
  span_q = start_q * block_q + jnp.arange(block_q)
  m = pl.load(m_ref, (pl.ds(start_q * block_q, block_q),))
  do = pl.load(do_scaled_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
  di = pl.load(delta_ref, (pl.ds(start_q * block_q, block_q),))
  dq = pl.load(dq_ref, (pl.ds(start_q * block_q, block_q),
                        slice(None)), eviction_policy="evict_last")

  def inner_loop(start_k, carry):
      dq = carry
      span_k = start_k * block_k + jnp.arange(block_k)
      k = pl.load(k_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
      v = pl.load(v_ref, (pl.ds(start_k * block_k, block_k), slice(None)))

      qk = pl.dot(q, k.T)
      qk = qk.astype(q_ref.dtype)
      qk = qk.astype(jnp.float32)
      if sm_scale != 1.0:
        qk *= sm_scale
      if causal:
        qk = jnp.where(span_q[:, None] >= span_k[None, :], qk, float('-inf'))
      p = jnp.exp(qk - m[:, None])
      dp = jnp.zeros((block_q, block_k), dtype=jnp.float32) - di[:, None]
      dp = dp + pl.dot(do, v.T)
      ds = p * dp
      if sm_scale != 1.0:
        ds = ds * sm_scale
      dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)
      return dq
  if causal:
      upper_bound = lax.div(start_q * block_q, block_k) + 1
  else:
      upper_bound = jt.cdiv(seq_len, block_k)
  dq = lax.fori_loop(0, upper_bound, inner_loop, dq)
  pl.store(dq_ref, (pl.ds(start_q * block_q, block_q),
                        slice(None)), dq, eviction_policy="evict_last")


def mha_backward_kernel_dkv(
    # Inputs
    q_ref, k_ref, v_ref, out_ref, do_scaled_ref,
    l_ref, m_ref, delta_ref,
    # Outputs
    dk_ref, dv_ref,
    *, sm_scale: float, causal: bool,
    block_q: int, block_d: int, block_k: int
):
  del out_ref, l_ref  # Not needed
  seq_len = q_ref.shape[0]
  start_k = pl.program_id(0)

  dv = jnp.zeros([block_k, block_d], dtype=jnp.float32)
  dk = jnp.zeros([block_k, block_d], dtype=jnp.float32)
  k = pl.load(k_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
  v = pl.load(v_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
  span_k = start_k * block_k + jnp.arange(block_k)

  def inner_loop(start_q, carry):
      dv, dk = carry
      q = pl.load(q_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
      qk = pl.dot(q, k.T)
      qk = qk.astype(q_ref.dtype)
      qk = qk.astype(jnp.float32)
      if sm_scale != 1.0:
        qk *= sm_scale
      if causal:
        span_q = start_q * block_q + jnp.arange(block_q)
        qk = jnp.where(span_q[:, None] >= span_k[None, :], qk, float('-inf'))
      m = pl.load(m_ref, (pl.ds(start_q * block_q, block_q),))
      p = jnp.exp(qk - m[:, None])
      do = pl.load(do_scaled_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
      dv = dv + pl.dot(p.astype(do.dtype).T, do)
      di = pl.load(delta_ref, (pl.ds(start_q * block_q, block_q),))
      dp = jnp.zeros((block_q, block_k), dtype=jnp.float32) - di[:, None]
      dp = dp + pl.dot(do, v.T)
      ds = p * dp
      if sm_scale != 1.0:
        ds = ds * sm_scale
      dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q)
      return dv, dk
  if causal:
      lower_bound = lax.div(start_k * block_k, block_q)
  else:
      lower_bound = 0
  dv, dk = lax.fori_loop(lower_bound, jt.cdiv(seq_len, block_q), inner_loop,
                           (dv, dk))
  pl.store(dv_ref, (pl.ds(start_k * block_k, block_k),
                      slice(None)), dv.astype(dv_ref.dtype))
  pl.store(dk_ref, (pl.ds(start_k * block_k, block_k),
                      slice(None)), dk.astype(dk_ref.dtype))


def _mha_backward(sm_scale: float, causal: bool, block_q: int, block_k: int,
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
    return jax.vjp(functools.partial(mha_reference, sm_scale=sm_scale,
                                     causal=causal), q, k, v)[1](do)
  elif backward_pass_impl == "triton":
    # We accumulate into dq so we need to initialize it to zeros.
    dq = jnp.zeros(q.shape, jnp.float32)
    out_shapes_q = jax.ShapeDtypeStruct(dq.shape, dq.dtype)

    grid_q = (jt.cdiv(seq_len, block_q), batch_size, num_heads)
    # grid_q = (batch_size, num_heads)
    # TODO(sharadmv): figure out why num_warps=8 doesn't work!
    num_warps = 4
    dq = pl.pallas_call(
        functools.partial(mha_backward_kernel_dq, block_q=block_q, block_d=head_dim,
                          block_k=block_k, sm_scale=sm_scale, causal=causal),
        grid=grid_q,
        out_shape=out_shapes_q,
        in_specs=[
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
          pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
          pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        ],
        out_specs=[
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        ],
        name="mha_backward",
        debug=debug,
        interpret=interpret,
        num_warps=num_warps,
        num_stages=1,
        input_output_aliases={8: 0})(q, k, v, out, do_scaled, l, m, delta, dq)

    grid_kv = (jt.cdiv(seq_len, block_k), batch_size, num_heads)
    out_shapes_kv = [
      jax.ShapeDtypeStruct(k.shape, k.dtype),
      jax.ShapeDtypeStruct(v.shape, v.dtype),
    ]
    dk, dv = pl.pallas_call(
        functools.partial(mha_backward_kernel_dkv, block_q=block_q, block_d=head_dim,
                          block_k=block_k, sm_scale=sm_scale, causal=causal),
        grid=grid_kv,
        out_shape=out_shapes_kv,
        in_specs=[
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
          pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
          pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        ],
        out_specs=[
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
          pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        ],
        name="mha_backward",
        debug=debug,
        interpret=interpret,
        num_warps=num_warps,
        num_stages=1)(q, k, v, out, do_scaled, l, m, delta)

  else:
    raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
  return dq.astype(q.dtype), dk, dv
mha.defvjp(_mha_forward, _mha_backward)


@functools.partial(jax.jit, static_argnames=['sm_scale', 'causal'])
def mha_reference(q, k, v, sm_scale=1.0, causal: bool = True):
  q_seq_len = q.shape[1]
  kv_seq_len = k.shape[1]
  logits = jnp.einsum('bqhc,bkhc->bhqk', q, k).astype(jnp.float32)
  if causal:
    mask = jnp.tril(jnp.ones((1, 1, q_seq_len, kv_seq_len), dtype=bool))
    mask = jnp.broadcast_to(mask, logits.shape)
    logits = jnp.where(mask, logits, float('-inf'))
  weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
  return jnp.einsum('bhqk,bkhc->bqhc', weights, v)
