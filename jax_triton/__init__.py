# Copyright 2025 The jax_triton Authors.
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

"""Library for JAX-Triton integrations."""

__all__ = [
    "utils",
    "triton_call",
    "cdiv",
    "next_power_of_2",
    "strides_from_shape",
    "__version__",
    "__version_info__",
]

from jax_triton import utils
from jax_triton.triton_lib import triton_call
from jax.experimental.pallas import cdiv
from jax.experimental.pallas import next_power_of_2
from jax.experimental.pallas import strides_from_shape
from jax_triton.version import __version__
from jax_triton.version import __version_info__
