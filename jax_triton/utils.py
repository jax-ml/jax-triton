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

"""Contains utilities for writing and calling Triton functions."""
from __future__ import annotations
import math

from typing import Any, Callable, Dict, Tuple, Union

import numpy as np

Grid = Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
GridOrLambda = Union[Grid, Callable[[Dict[str, Any]], Grid]]


def normalize_grid(grid: GridOrLambda, metaparams) -> Tuple[int, int, int]:
  if callable(grid):
    grid = grid(metaparams)
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))


def avals_to_layouts(avals):
  return [list(reversed(range(aval.ndim))) for aval in avals]


def cdiv(a: int, b: int) -> int:
  return (a + b - 1) // b


def strides_from_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
  size = np.prod(shape)
  strides = []
  for s in shape:
    size = size // s
    strides.append(int(size))
  return tuple(strides)


def next_power_of_2(x: int) -> int:
  if x == 0:
    return 1
  return 2 ** math.ceil(math.log2(x))
