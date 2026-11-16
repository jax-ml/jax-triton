# Copyright 2026 The jax_triton Authors.
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

"""Tests for jax_triton examples."""

import importlib.util
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import config
from jax import random
import jax.numpy as jnp
import numpy as np

try:
  # pylint: disable-next=g-import-not-at-top
  from jax_triton.examples import fused_attention
except ImportError:
  _repo_root = pathlib.Path(__file__).resolve().parents[1]
  _spec = importlib.util.spec_from_file_location(
      "examples.fused_attention",
      _repo_root / "examples" / "fused_attention.py",
  )
  assert _spec is not None and _spec.loader is not None
  fused_attention = importlib.util.module_from_spec(_spec)
  _spec.loader.exec_module(fused_attention)


config.parse_flags_with_absl()


def _ref_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    sm_scale: float | None = None,
    causal: bool = False,
) -> jax.Array:
  # fused_attention uses (B, H, S, D); dot_product_attention uses (B, S, H, D).
  q_bshd = jnp.transpose(q, (0, 2, 1, 3))
  k_bshd = jnp.transpose(k, (0, 2, 1, 3))
  v_bshd = jnp.transpose(v, (0, 2, 1, 3))
  out_bshd = jax.nn.dot_product_attention(
      q_bshd, k_bshd, v_bshd, scale=sm_scale, is_causal=causal
  )
  return jnp.transpose(out_bshd, (0, 2, 1, 3))


class ExamplesTest(parameterized.TestCase):

  @parameterized.product(
      shape=[
          (2, 3, 256, 64),
          (2, 4, 512, 128),
          (1, 3, 200, 64),
      ],
      causal=[False, True],
      dtype=[jnp.float16, jnp.bfloat16],
  )
  def test_fused_attention(self, shape, causal, dtype):
    q_key, k_key, v_key = random.split(random.key(0), 3)
    q = random.normal(q_key, shape, dtype=dtype)
    k = random.normal(k_key, shape, dtype=dtype)
    v = random.normal(v_key, shape, dtype=dtype)

    if causal:
      out = fused_attention.fused_attention(q, k, v, causal=True)
    else:
      out = fused_attention.fused_attention(q, k, v)
    ref = _ref_attention(q, k, v, causal=causal)

    self.assertEqual(out.shape, shape)
    self.assertEqual(out.dtype, dtype)
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-2
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.asarray(ref, dtype=np.float32),
        atol=tol,
        rtol=tol,
    )

  def test_fused_attention_custom_scale_and_jit(self):
    shape = (2, 3, 256, 64)
    q_key, k_key, v_key = random.split(random.key(1), 3)
    q = random.normal(q_key, shape, dtype=jnp.float16)
    k = random.normal(k_key, shape, dtype=jnp.float16)
    v = random.normal(v_key, shape, dtype=jnp.float16)

    sm_scale = 0.25
    jitted_fn = jax.jit(
        lambda q_, k_, v_: fused_attention.fused_attention(
            q_, k_, v_, sm_scale=sm_scale
        )
    )
    out = jitted_fn(q, k, v)
    ref = _ref_attention(q, k, v, sm_scale=sm_scale)

    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.asarray(ref, dtype=np.float32),
        atol=1e-2,
        rtol=1e-2,
    )

  def test_fused_attention_main(self):
    fused_attention.main(None)


if __name__ == "__main__":
  absltest.main()
