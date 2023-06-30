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

"""Pallas utility functions."""
from jax import lax


def when(condition):
  def _wrapped(f):
    if isinstance(condition, bool):
      if condition:
        f()
    else:
      lax.cond(condition, f, lambda: None)
  return _wrapped
