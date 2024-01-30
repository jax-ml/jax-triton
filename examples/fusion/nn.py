# Copyright 2024 The jax_triton Authors.
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

import jax
import jax.numpy as jnp
from jax import random
import timeit

from jax_triton.experimental import fusion as ji

dtype = jnp.float16

def dense(features, weights, activation=jax.nn.gelu):
  kernel, bias = weights
  return activation(features.dot(kernel) + bias[None])

def init_layer(key, in_size, out_size):
  k1, k2 = random.split(key)
  return (random.normal(k1, [in_size, out_size], dtype),
          random.normal(k2, [out_size], dtype))

batch_size = 4096
hidden_size = 512
input_size = 512

n_layers = 4

def apply(weights, x):
  for weight in weights[:-1]:
    x = dense(x, weight)
  return dense(x, weights[-1], activation=lambda x: x)

def main(unused_argv):
  keys = random.split(random.PRNGKey(0), n_layers)
  keys_iter = iter(keys)

  weights = [
      init_layer(next(keys_iter), input_size, hidden_size),
      ] + [
          init_layer(next(keys_iter), hidden_size, hidden_size)
          for _ in range(n_layers - 1)
      ]

  mlp = functools.partial(apply, weights)

  x = jnp.ones((batch_size, input_size), dtype=dtype)

  xla_jit_net = jax.jit(mlp)
  ji_jit_net = jax.jit(ji.jit(mlp, debug=True))

  ji_jit_net(x).block_until_ready()
  xla_jit_net(x).block_until_ready()

  n_trials = 5000

  t = timeit.timeit(lambda: ji_jit_net(x).block_until_ready(), number=n_trials)
  print(f"jax-inductor: {t:.4}ms")

  t = timeit.timeit(lambda: xla_jit_net(x).block_until_ready(), number=n_trials)
  print(f"XLA: {t:.4}ms")

if __name__ == "__main__":
  from absl import app
  app.run(main)
