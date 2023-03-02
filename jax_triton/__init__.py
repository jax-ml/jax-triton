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

"""Library for JAX-Triton integrations."""
from jax_triton.utils import cdiv
from jax_triton.utils import next_power_of_2
from jax_triton.utils import strides_from_shape
from jax_triton.triton_lib import triton_call
from jax_triton.triton_lib import triton_kernel_call_lib
from jax_triton.version import __version__
from jax_triton.version import __version_info__
from jax_triton import pallas

get_compute_capability = triton_kernel_call_lib.get_compute_capability
