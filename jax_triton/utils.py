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

"""Contains utilities for writing and calling Triton functions."""


__all__ = ["cdiv", "strides_from_shape", "next_power_of_2"]


from jax.experimental.pallas import cdiv
from jax.experimental.pallas import strides_from_shape
from jax.experimental.pallas import next_power_of_2
