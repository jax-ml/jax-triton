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
import timeit

from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import for_loop
import numpy as np

import jax_triton as jt
from jax_triton import pallas as pl

def layer_norm_forward_kernel(
    x_ref, weight_ref, bias_ref, # Input arrays
    o_ref, mean_ref=None, rstd_ref=None, # Output arrays
    *, eps: float, block_size: int):
  row_idx = pl.program_id(axis=0)

  def mean_body(i, acc_ref):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < x_ref.shape[1]
    a = pl.load(x_ref, (row_idx, col_idx), mask=mask, other=0.,
                eviction_policy="evict_last").astype(jnp.float32)
    acc_ref[:] += a
  mean = for_loop(jt.cdiv(x_ref.shape[1], block_size), mean_body,
                  jnp.zeros(block_size)).mean(axis=0)

  def var_body(i, acc_ref):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < x_ref.shape[1]
    a = pl.load(x_ref, (row_idx, col_idx), mask=mask, other=0.,
                eviction_policy="evict_last").astype(jnp.float32)
    a = jnp.where(mask, a - mean, 0.)
    acc_ref[:] += a * a
  var = for_loop(jt.cdiv(x_ref.shape[1], block_size), var_body,
                 jnp.zeros(block_size)).mean()
  rstd = 1 / jnp.sqrt(var + eps)
  if mean_ref is not None:
    mean_ref[row_idx] = mean
  if rstd_ref is not None:
    rstd_ref[row_idx] = rstd

  def body(i, _):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < x_ref.shape[1]
    weight = pl.load(weight_ref, (col_idx,), mask=mask)
    bias = pl.load(bias_ref, (col_idx,), mask=mask)
    x = pl.load(x_ref, (row_idx, col_idx), mask=mask, other=0.,
                eviction_policy="evict_first").astype(jnp.float32)
    out = (x - mean) * rstd * weight + bias
    pl.store(o_ref, (row_idx, col_idx), out.astype(o_ref.dtype), mask=mask)
  for_loop(jt.cdiv(x_ref.shape[1], block_size), body, ())


@functools.partial(jax.jit, static_argnames=["num_warps", "num_stages",
                                             "num_stages", "eps", "interpret"])
def layer_norm(
    x, weight, bias,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = 3,
    eps: float = 1e-5,
    interpret: bool = False):
  m, n = x.shape
  # Triton heuristics
  # Less than 64KB per feature: enqueue fused kernel
  max_fused_size = 65536 // x.dtype.itemsize
  block_size = min(max_fused_size, jt.next_power_of_2(n))
  block_size = min(max(block_size, 128), 4096)
  num_warps = min(max(block_size // 256, 1), 8)

  grid = m  # one thread per row

  kernel = functools.partial(layer_norm_forward_kernel, eps=eps,
                             block_size=block_size)
  out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
  return pl.pallas_call(kernel, num_warps=num_warps, num_stages=num_stages,
                        grid=grid, out_shape=out_shape, debug=False,
                        interpret=interpret)(x, weight, bias)


@functools.partial(jax.jit, static_argnames=["eps"])
def layer_norm_reference(x, weight, bias, *, eps: float = 1e-5):
  mean = jnp.mean(x, axis=1)
  mean2 = jnp.mean(jnp.square(x), axis=1)
  var = jnp.maximum(0., mean2 - jnp.square(mean))
  y = x - mean[:, None]
  mul = lax.rsqrt(var + eps)
  return y * mul[:, None] * weight[None] + bias[None]


if __name__ == "__main__":
  dtype = jnp.float32
  m, n = 4096, 4096

  weight_key, bias_key, x_key = jax.random.split(jax.random.PRNGKey(0), 3)
  weight = jax.random.normal(weight_key, (n,), dtype=dtype)
  bias = jax.random.normal(bias_key, (n,), dtype=dtype)
  x = jax.random.normal(x_key, (m, n), dtype=dtype)

  out = layer_norm(x, weight, bias)
  out_ref = layer_norm_reference(x, weight, bias)
  np.testing.assert_allclose(out, out_ref, atol=0.03, rtol=0.03)

  n_trials = 1000
  duration = timeit.timeit(lambda: layer_norm(x, weight, bias).block_until_ready(),
                           number=n_trials)
  print(f"Fused Layer Norm: {duration / n_trials * 1000:.2f}ms")
  duration = timeit.timeit(lambda: layer_norm_reference(x, weight, bias).block_until_ready(),
                           number=n_trials)
  print(f"Reference Layer Norm: {duration / n_trials * 1000:.2f}ms")
