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

import functools
import timeit

import jax
import jax.numpy as jnp
import numpy as np

from jax_triton.pallas.ops import attention

def main(unused_argv):
  dtype = jnp.float16
  batch, seq_len, n_heads, head_dim = 1, 8192, 24, 128
  shape = (batch, seq_len, n_heads, head_dim)
  causal = True

  q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(q_key, shape, dtype=dtype)
  k = jax.random.normal(k_key, shape, dtype=dtype)
  v = jax.random.normal(v_key, shape, dtype=dtype)


  print("FWD")
  mha = jax.jit(functools.partial(attention.mha, backward_pass_impl="triton_split",
                                  causal=causal))
  o = mha(q, k, v).block_until_ready()
  mha_ref = jax.checkpoint(
    functools.partial(attention.mha_reference, causal=causal))
  print("FWD REF")
  o_ref = mha_ref(q, k, v).block_until_ready()

  print("DONE!")

  # mha_interpret = functools.partial(attention.mha, interpret=True)
  # o_int = mha_interpret(q, k, v).block_until_ready()

  # np.testing.assert_allclose(o, o_int, atol=0.03, rtol=0.03)
  # np.testing.assert_allclose(o_int, o_ref, atol=0.05, rtol=0.05)
  np.testing.assert_allclose(o, o_ref, atol=2e-2, rtol=2e-2)
  n_trials = 10
  duration = timeit.timeit(lambda: jax.block_until_ready(mha(q, k, v)),
                           number=n_trials)
  print(f"Fused Attention: {duration / n_trials * 1000:.2f}ms")
  duration = timeit.timeit(lambda:
                           jax.block_until_ready(mha_ref(q, k, v)),
                           number=n_trials)
  print(f"Reference Attention: {duration / n_trials * 1000:.2f}ms")

  o, mha_vjp = jax.vjp(mha, q, k, v)
  o_ref, mha_vjp_ref = jax.vjp(mha_ref, q, k, v)

  def fwd_bwd(impl, *args):
    o, f_vjp = jax.vjp(impl, *args)
    return f_vjp(o)
  fwd_bwd = jax.jit(fwd_bwd, static_argnums=(0,))
  mha_vjp = functools.partial(fwd_bwd, mha)
  mha_vjp_ref = functools.partial(fwd_bwd, mha_ref)

  print("BWD")
  jax.block_until_ready(mha_vjp(q, k, v))
  print("BWD REF")
  jax.block_until_ready(mha_vjp_ref(q, k, v))
  print("DONE!")

  n_trials = 100
  duration = timeit.timeit(lambda: jax.block_until_ready(mha_vjp(q, k, v)),
                           number=n_trials)
  print(f"Fused Attention: {duration / n_trials * 1000:.2f}ms")
  duration = timeit.timeit(lambda: jax.block_until_ready(mha_vjp_ref(q, k, v)),
                           number=n_trials)
  print(f"Reference Attention: {duration / n_trials * 1000:.2f}ms")

if __name__ == "__main__":
  from absl import app
  app.run(main)
