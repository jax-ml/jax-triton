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

import jax
import jax.numpy as jnp
import numpy as np

from jax_triton.pallas.ops import attention

if __name__ == "__main__":
  dtype = jnp.float16
  batch, seq_len, n_heads, head_dim = 4, 1024, 48, 64
  shape = (batch, seq_len, n_heads, head_dim)

  q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(q_key, shape, dtype=dtype)
  k = jax.random.normal(k_key, shape, dtype=dtype)
  v = jax.random.normal(v_key, shape, dtype=dtype)


  o_ref = attention.mha_reference(q, k, v).block_until_ready()

  mha = jax.jit(attention.mha)
  o = mha(q, k, v).block_until_ready()

  mha_interpret = functools.partial(attention.mha, interpret=True)
  o_int = mha_interpret(q, k, v).block_until_ready()

  np.testing.assert_allclose(o, o_int, atol=0.03, rtol=0.03)
  np.testing.assert_allclose(o_int, o_ref, atol=0.05, rtol=0.05)
  np.testing.assert_allclose(o, o_ref, atol=0.05, rtol=0.05)

  n_trials = 1000
  duration = timeit.timeit(lambda: mha(q, k, v).block_until_ready(),
                           number=n_trials)
  print(f"Fused Attention: {duration / n_trials * 1000:.2f}ms")
  duration = timeit.timeit(lambda: attention.mha_reference(q, k, v).block_until_ready(),
                           number=n_trials)
  print(f"Reference Attention: {duration / n_trials * 1000:.2f}ms")
