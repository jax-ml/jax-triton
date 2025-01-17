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

from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
from jax_triton import ops
import numpy as np


class OpsTest(parameterized.TestCase):

  def test_softmax(self):
    x = jax.random.normal(random.PRNGKey(0), (1024, 125), dtype=jnp.float32)

    np.testing.assert_allclose(
        ops.softmax(x, axis=-1), jax.nn.softmax(x, axis=-1),
        atol=1e-7,
        rtol=1e-6
    )


if __name__ == "__main__":
  parameterized.absltest.main()
