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

import timeit

import jax
import jax.numpy as jnp
import numpy as np

from jax_triton.pallas.ops import layer_norm

if __name__ == "__main__":
  dtype = jnp.float32
  b, m, n = 32, 4096, 8196

  weight_key, bias_key, x_key = jax.random.split(jax.random.PRNGKey(0), 3)
  weight = jax.random.normal(weight_key, (n,), dtype=dtype)
  bias = jax.random.normal(bias_key, (n,), dtype=dtype)
  x = jax.random.normal(x_key, (b, m, n), dtype=dtype)

  out = layer_norm.layer_norm(x, weight, bias, interpret=True)
  out_ref = layer_norm.layer_norm_reference(x, weight, bias)
  np.testing.assert_allclose(out, out_ref, rtol=1e-6, atol=1e-6)

  def f(x, w, b):
      return layer_norm.layer_norm(x, w, b).sum()

  def f_ref(x, w, b):
      return layer_norm.layer_norm_reference(x, w, b).sum()

  f_grad = jax.jit(jax.grad(f, argnums=(0, 1, 2)))
  f_ref_grad = jax.jit(jax.grad(f_ref, argnums=(0, 1, 2)))

  dx, dw, db = f_grad(x, weight, bias)
  dx_ref, dw_ref, db_ref = f_ref_grad(x, weight, bias)
  np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(dw, dw_ref, rtol=1e-2, atol=1e-2)
  np.testing.assert_allclose(db, db_ref, rtol=1e-2, atol=1e-2)

  n_trials = 1000
  duration = timeit.timeit(lambda: layer_norm.layer_norm(x, weight, bias).block_until_ready(),
                           number=n_trials)
  print(f"Fused Layer Norm: {duration / n_trials * 1000:.2f}ms")
  duration = timeit.timeit(lambda: layer_norm.layer_norm_reference(x, weight, bias).block_until_ready(),
                          number=n_trials)
  print(f"Reference Layer Norm: {duration / n_trials * 1000:.2f}ms")
  duration = timeit.timeit(lambda: jax.block_until_ready(f_grad(x, weight, bias)),
                          number=n_trials)
  print(f"Fused Layer Norm Gradient: {duration / n_trials * 1000:.2f}ms")
  duration = timeit.timeit(lambda: jax.block_until_ready(f_ref_grad(x, weight, bias)),
                          number=n_trials)
  print(f"Reference Layer Norm Gradient: {duration / n_trials * 1000:.2f}ms")
