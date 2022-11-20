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
import argparse
import functools
import timeit

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import random
import jax
from jax import lax
from jax._src.lax.control_flow import for_loop
import jax.numpy as jnp
import numpy as np

import jax_triton as jt
from jax_triton import pallas as pl

def lstm_kernel(
    w_ii_ref, u_hi_ref, b_hi_ref,
    w_if_ref, u_hf_ref, b_hf_ref,
    w_ig_ref, u_hg_ref, b_hg_ref,
    w_io_ref, u_ho_ref, b_ho_ref,
    input_ref, h_prev_ref, c_prev_ref,
    output_ref, c_next_ref, *,
    block_m: int, block_n: int,
    block_k: int,
    block_h: int):
  if h_prev_ref.shape[1] // block_h != input_ref.shape[1] // block_k:
    raise ValueError("Invalid block shapes")
  pid_m = pl.program_id(axis=0)
  pid_n = pl.program_id(axis=1)
  idx_m = pid_m * block_m + jnp.arange(block_m)
  idx_n = pid_n * block_n + jnp.arange(block_n)
  idx_m = pl.max_contiguous(pl.multiple_of(idx_m, block_m), block_m)
  idx_n = pl.max_contiguous(pl.multiple_of(idx_n, block_n), block_n)

  acc_i = acc_f = acc_g = acc_o = jnp.zeros((block_m, block_n), jnp.float32)

  def body(k, acc_refs):
    accs = [acc_ref[:] for acc_ref in acc_refs]
    idx_k = k * block_k + jnp.arange(block_k)
    idx_h = k * block_h + jnp.arange(block_h)
    ws = [pl.load(w_ref, (idx_k, idx_n))
          for w_ref in [w_ii_ref, w_if_ref, w_ig_ref, w_io_ref]]
    us = [pl.load(u_ref, (idx_h, idx_n))
          for u_ref in [u_hi_ref, u_hf_ref, u_hg_ref, u_ho_ref]]
    x = pl.load(input_ref, (idx_m, idx_k))
    xs = [pl.dot(x, w) for w in ws]
    h = pl.load(h_prev_ref, (idx_m, idx_h))
    hs = [pl.dot(h, u) for u in us]
    accs = [acc + x + h for acc, x, h in zip(accs, xs, hs)]
    for acc_ref, acc in zip(acc_refs, accs):
      acc_ref[:] = acc
  num_k_blocks = input_ref.shape[1] // block_k
  accs = for_loop.for_loop(num_k_blocks, body, [acc_i, acc_f, acc_o, acc_g])
  bs = [pl.load(b_ref, (idx_n,))
        for b_ref in [b_hi_ref, b_hf_ref, b_hg_ref, b_ho_ref]]
  acc_i, acc_f, acc_g, acc_o = [acc + b for acc, b in zip(accs, bs)]
  i_gate, f_gate, o_gate = (
      jax.nn.sigmoid(acc_i), jax.nn.sigmoid(acc_f), jax.nn.sigmoid(acc_o))
  cell = jnp.tanh(acc_g)
  c_prev = pl.load(c_prev_ref, (idx_m, idx_n))
  c_next = f_gate * c_prev + i_gate * cell
  h_next = (o_gate * jnp.tanh(c_next))
  pl.store(output_ref, (idx_m, idx_n),
           h_next.astype(output_ref.dtype))
  pl.store(c_next_ref, (idx_m, idx_n),
           c_next.astype(c_next_ref.dtype))


@functools.partial(jax.jit, static_argnames=["block_batch", "block_features",
                                             "block_hidden",
                                             "num_stages", "num_warps",
                                             "debug"])
def lstm_cell(weights, x, h, c, *, block_batch: int, block_features: int,
              block_hidden: int, num_warps: int,
              num_stages: int, debug: bool = False):
  ((w_ii, u_hi, b_hi), (w_if, u_hf, b_hf),
   (w_ig, u_hg, b_hg), (w_io, u_ho, b_ho)) = weights
  batch_size, num_features = x.shape
  hidden_size = h.shape[1]
  num_feature_blocks = jt.cdiv(num_features, block_features)
  block_h = jt.cdiv(hidden_size, num_feature_blocks)

  grid = (jt.cdiv(batch_size, block_batch), jt.cdiv(hidden_size, block_hidden))
  out_shapes = (
      jax.ShapeDtypeStruct((batch_size, hidden_size), x.dtype),
      jax.ShapeDtypeStruct((batch_size, hidden_size), x.dtype),
  )
  kernel = functools.partial(lstm_kernel, block_m=block_batch,
                             block_n=block_hidden,
                             block_k=block_features,
                             block_h=block_h)
  y, c = pl.pallas_call(kernel, grid=grid,
                        out_shape=out_shapes,
                        interpret=False,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        name="lstm_cell",
                        debug=debug)(w_ii, u_hi, b_hi, w_if, u_hf, b_hf,
                                     w_ig, u_hg, b_hg, w_io, u_ho, b_ho,
                                     x, h, c)
  return y, c

@jax.jit
def lstm_cell_reference(weights, x, h, c):
  ((w_ii, u_hi, b_hi), (w_if, u_hf, b_hf),
   (w_ig, u_hg, b_hg), (w_io, u_ho, b_ho)) = weights
  ws = [w_ii, w_if, w_ig, w_io]
  us = [u_hi, u_hf, u_hg, u_ho]
  bs = [b_hi, b_hf, b_hg, b_ho]
  xs = [jnp.dot(x, w) for w in ws]
  hs = [jnp.dot(h, u) for u in us]
  accs = [x + h for x, h in zip(xs, hs)]
  acc_i, acc_f, acc_g, acc_o = [acc + b[None] for acc, b in zip(accs, bs)]
  i_gate, f_gate, o_gate = (
      jax.nn.sigmoid(acc_i), jax.nn.sigmoid(acc_f), jax.nn.sigmoid(acc_o))
  cell = jnp.tanh(acc_g)
  c = f_gate * c + i_gate * cell
  y = o_gate * jnp.tanh(c)
  return y, c

@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _init_weights(key, feature_size, hidden_size, dtype):
  k1, k2, k3 = random.split(key, 3)
  w = random.normal(k1, (feature_size, hidden_size), dtype)
  u = random.normal(k2, (hidden_size, hidden_size), dtype)
  b = random.normal(k3, (hidden_size,), dtype)
  return w, u, b

def make_lstm(kernel):
  @jax.jit
  def lstm(weights, xs, c):
    h = jnp.zeros_like(c)
    def body(carry, x):
      h, c = carry
      h, c = kernel(weights, x, h, c)
      return (h, c), h
    (_, c), ys = jax.lax.scan(body, (h, c), xs)
    return ys, c
  return lstm

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--feature_size', type=int, default=512)
  parser.add_argument('--hidden_size', type=int, default=256)
  parser.add_argument('--block_batch', type=int, default=16)
  parser.add_argument('--block_features', type=int, default=32)
  parser.add_argument('--block_hidden', type=int, default=16)
  parser.add_argument('--num_warps', type=int, default=4)
  parser.add_argument('--num_stages', type=int, default=3)
  parser.add_argument('--seq_len', type=int, default=500)
  parser.add_argument('--n_trials', type=int, default=1000)


  args = parser.parse_args()
  x_key, h_key, c_key, weights_key, xs_key = random.split(random.PRNGKey(0), 5)
  dtype = jnp.float16

  batch_size = args.batch_size
  feature_size = args.feature_size
  hidden_size = args.hidden_size

  block_batch = args.block_batch
  block_features = args.block_features
  block_hidden = args.block_hidden
  num_warps = args.num_warps
  num_stages = args.num_stages

  weights = [_init_weights(k, feature_size, hidden_size, dtype)
             for k in random.split(weights_key, 4)]
  x = random.normal(x_key, (batch_size, feature_size), dtype)
  h = random.normal(h_key, (batch_size, hidden_size), dtype)
  c = random.normal(c_key, (batch_size, hidden_size), dtype)
  lstm_cell = jax.jit(functools.partial(lstm_cell,
                                        block_batch=block_batch,
                                        block_hidden=block_hidden,
                                        block_features=block_features,
                                        num_warps=num_warps,
                                        num_stages=num_stages))
  y, c_next = jax.block_until_ready(lstm_cell(weights, x, h, c))
  y_ref, c_next_ref = lstm_cell_reference(weights, x, h, c)
  np.testing.assert_allclose(y, y_ref, atol=0.05, rtol=0.05)
  np.testing.assert_allclose(c_next, c_next_ref, atol=0.05, rtol=0.05)

  if args.n_trials > 0:
    seq_len = args.seq_len
    xs = random.normal(xs_key, (seq_len, batch_size, feature_size), dtype)
    lstm = make_lstm(lstm_cell)
    lstm_reference = make_lstm(lstm_cell_reference)

    jax.block_until_ready(lstm(weights, xs, c))
    jax.block_until_ready(lstm_reference(weights, xs, c))

    print("Starting benchmark...")
    n_trials = args.n_trials
    xla_duration = timeit.timeit(lambda: jax.block_until_ready(
      lstm_reference(weights, xs, c)), number=n_trials)
    print(f"XLA: {xla_duration / n_trials * 1000:.2f}ms")
    triton_duration = timeit.timeit(lambda: jax.block_until_ready(
      lstm(weights, xs, c)), number=n_trials)
    print(f"Triton: {triton_duration / n_trials * 1000:.4f}ms")
    print(f"Triton speedup: {xla_duration / triton_duration:.2f}")
