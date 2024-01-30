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
import rich.console
import rich.table
import timeit
import numpy as np

import jax
import jax.numpy as jnp
from jax import random

from jax_triton.experimental import fusion as ji

leaky_relu = lambda x: jnp.where(x >= 0., x, 0.01 * x)

def dense_activation(act, x, w, b):
  return act(jnp.matmul(x, w) + b)

dense = functools.partial(dense_activation, lambda x: x)
dense_leaky_relu = functools.partial(dense_activation, leaky_relu)
dense_gelu = functools.partial(dense_activation, jax.nn.gelu)

FUNCTIONS = [
    ("jax-inductor", "none", ji.jit(dense)),
    ("jax-inductor", "leaky_relu", ji.jit(dense_leaky_relu)),
    ("jax-inductor", "gelu", ji.jit(dense_gelu)),
    ("XLA", "none", jax.jit(dense)),
    ("XLA", "leaky_relu", jax.jit(dense_leaky_relu)),
    ("XLA", "gelu", jax.jit(dense_gelu)),
    ]

SIZES = [
    (256, 128),
    (512, 256),
    (1024, 512),
    (2048, 1024),
    ]

n_trials = 20000

def main(unused_argv):
  console = rich.console.Console()
  for b, d in SIZES:
    table = rich.table.Table(title=f"({b}, {d}) x ({d}, {d})", )
    table.add_column("Codegen")
    table.add_column("Activation")
    table.add_column("Average time (ms)")

    k1, k2, k3 = random.split(random.PRNGKey(0), 3)
    x = random.normal(k1, (b, d), dtype=jnp.float16).block_until_ready()
    w = random.normal(k2, (d, d), dtype=jnp.float16).block_until_ready()
    b = random.normal(k3, (d,), dtype=jnp.float16).block_until_ready()

    for func_name, act_name, func in FUNCTIONS:
      for _ in range(10):
        func(x, w, b).block_until_ready()
      times = timeit.Timer(
          lambda: func(x, w, b).block_until_ready()).repeat(
              number=n_trials, repeat=5)
      table.add_row(func_name, act_name, f"{np.min(times) / n_trials * 1000:.4}")
    console.print(table)

if __name__ == "__main__":
  from absl import app
  app.run(main)
