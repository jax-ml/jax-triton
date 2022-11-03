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

"""Module for pallas, a jaxpr "dialect" for Triton."""
from jax_triton.pallas.pallas_call import pallas_call
from jax_triton.pallas.primitives import atomic_add
from jax_triton.pallas.primitives import atomic_and
from jax_triton.pallas.primitives import atomic_max
from jax_triton.pallas.primitives import atomic_min
from jax_triton.pallas.primitives import atomic_or
from jax_triton.pallas.primitives import atomic_xchg
from jax_triton.pallas.primitives import atomic_xor
from jax_triton.pallas.primitives import dot
from jax_triton.pallas.primitives import load
from jax_triton.pallas.primitives import max_contiguous
from jax_triton.pallas.primitives import multiple_of
from jax_triton.pallas.primitives import program_id
from jax_triton.pallas.primitives import store
from jax_triton.pallas.primitives import swap
