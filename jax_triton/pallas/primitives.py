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

"""Module for pallas-specific JAX primitives and functions."""
from __future__ import annotations
import dataclasses
import enum
import functools

from typing import Any, List, Optional, Tuple, Union

import jax
from jax import core as jax_core
from jax import lax
from jax import tree_util
from jax._src import ad_util
from jax._src import state
from jax._src.util import (safe_map, safe_zip, split_list, merge_lists,
                           partition_list)
from jax._src.state import primitives as state_primitives
from jax._src.typing import Array
from jax.interpreters import ad
import jax.numpy as jnp
import numpy as np

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

def _process_idx(idx, ref_shape):
  if any(isinstance(i, slice) and i != slice(None) for i in idx):
    raise NotImplementedError("Non-`slice(None)` slices not supported yet.")
  if len(idx) != len(ref_shape):
    raise ValueError("Must provide indexer for each dimension of `Ref`.")
  is_int_indexing = [isinstance(i, (jnp.ndarray, int)) for i in idx]
  other_indexers, int_indexers = partition_list(is_int_indexing, idx)
  int_indexers = [np.array(i, np.int32) if isinstance(i, int) else i for i in
                  int_indexers]
  indexer_shapes = [jnp.shape(i) for i in int_indexers]
  bcast_shape = tuple(s for i in indexer_shapes for s in i)
  idx_iter = iter(range(len(bcast_shape)))
  int_indexers = [
      lax.broadcast_in_dim(i, bcast_shape, tuple(next(idx_iter) for _ in
                                                 range(len(i.shape))))
      for i in int_indexers
  ]
  return merge_lists(is_int_indexing, other_indexers, int_indexers)

program_id_p = jax_core.Primitive("program_id")

def program_id(axis):
  return program_id_p.bind(axis=axis)

def _program_id_abstract_eval(**_):
  return jax_core.ShapedArray((), jnp.int32)
program_id_p.def_abstract_eval(_program_id_abstract_eval)

class AtomicOpType(enum.Enum):
  XCHG = "xchg"
  ADD = "add"
  MAX = "max"
  MIN = "min"
  AND = "and"
  OR = "or"
  XOR = "xor"

atomic_rmw_p = jax_core.Primitive("atomic_rmw")

def _atomic_abstract_eval(ref_aval, val_aval, *all_avals,
                          args_tree, atomic_type: AtomicOpType,
                          **_: Any):
  if ref_aval.dtype == jnp.dtype("float16") and atomic_type != AtomicOpType.ADD:
    raise ValueError(f"`atomic_{atomic_type.value}` does not support f16.")
  if ref_aval.dtype in {jnp.dtype("bool"), jnp.dtype("int8"),
                        jnp.dtype("int16"), jnp.bfloat16}:
    raise ValueError(f"`atomic_{atomic_type.value}` does not support {ref_aval.dtype}.")
  return _swap_abstract_eval(ref_aval, val_aval, *all_avals,
                             args_tree=args_tree)
atomic_rmw_p.def_effectful_abstract_eval(_atomic_abstract_eval)

def atomic_rmw(x_ref, idx, val, *, mask: Optional[Any] = None, atomic_type: AtomicOpType):
  idx = NDIndexer.from_indices_shape(idx, x_ref.shape)
  args = (idx,)
  if mask is not None:
    args = (*args, mask)
  flat_args, args_tree = tree_util.tree_flatten(args)
  return atomic_rmw_p.bind(x_ref, val, *flat_args, args_tree=args_tree,
                           atomic_type=atomic_type, masked=mask is not None)

atomic_xchg = functools.partial(atomic_rmw, atomic_type=AtomicOpType.XCHG)
atomic_add = functools.partial(atomic_rmw, atomic_type=AtomicOpType.ADD)
atomic_max = functools.partial(atomic_rmw, atomic_type=AtomicOpType.MAX)
atomic_min = functools.partial(atomic_rmw, atomic_type=AtomicOpType.MIN)
atomic_and = functools.partial(atomic_rmw, atomic_type=AtomicOpType.AND)
atomic_or = functools.partial(atomic_rmw, atomic_type=AtomicOpType.OR)
atomic_xor = functools.partial(atomic_rmw, atomic_type=AtomicOpType.XOR)

max_contiguous_p = jax_core.Primitive("max_contiguous")

def max_contiguous(x, values):
  if not isinstance(values, list):
    values = [values]
  return max_contiguous_p.bind(x, values=values)

def _max_contiguous_abstract_eval(aval, **_):
  return aval
max_contiguous_p.def_abstract_eval(_max_contiguous_abstract_eval)

multiple_of_p = jax_core.Primitive("multiple_of")

def multiple_of(x, values):
  if not isinstance(values, list):
    values = [values]
  return multiple_of_p.bind(x, values=values)

def _multiple_of_abstract_eval(aval, **_):
  return aval
multiple_of_p.def_abstract_eval(_multiple_of_abstract_eval)

@tree_util.register_pytree_node_class
@dataclasses.dataclass
class Slice:
  start: Any
  size: int

  def tree_flatten(self):
    if isinstance(self.start, int):
      return (), (True, self.start, self.size)
    return (self.start,), (False, self.size)

  @classmethod
  def tree_unflatten(cls, data, xs):
    if data[0]:
      return Slice(data[1], data[2])
    return Slice(xs[0], data[1])

  @classmethod
  def from_slice(cls, slc: slice, size: int) -> Slice:
    start, stop = slc.start, slc.stop
    start = 0 if start is None else start
    stop = size if stop is None else stop
    return Slice(start, stop - start)

def dslice(start: Optional[Union[int, Array]], stop: Optional[int] = None):
  if start is None:
    return slice(None)
  if stop is None:
    if not isinstance(start, int):
      raise ValueError("Non-static `dslice`")
    return Slice(0, start)
  return Slice(start, stop)

@tree_util.register_pytree_node_class
@dataclasses.dataclass
class NDIndexer:
  indices: Tuple[Union[int, Slice, Array]]
  shape: Tuple[int, ...]
  int_indexer_shape: Tuple[int, ...]

  def __post_init__(self):
    if len(self.indices) != len(self.shape):
      raise ValueError("`indices` must be the same length as `Ref` shape.")

  def tree_flatten(self):
    indexed_dims = [not isinstance(idx, slice) for idx in self.indices]
    slice_idx, non_slice_idx = partition_list(indexed_dims, self.indices)
    flat_idx, idx_tree = tree_util.tree_flatten(non_slice_idx)
    return flat_idx, (slice_idx, idx_tree, indexed_dims, self.shape,
                      self.int_indexer_shape)

  @classmethod
  def tree_unflatten(cls, data, flat_idx):
    slice_idx, idx_tree, indexed_dims, shape, int_indexer_shape = data
    non_slice_idx = tree_util.tree_unflatten(idx_tree, flat_idx)
    indices = merge_lists(indexed_dims, slice_idx, non_slice_idx)
    return NDIndexer(tuple(indices), shape, int_indexer_shape)

  @classmethod
  def from_indices_shape(cls, indices, shape) -> NDIndexer:
    indices = tuple(Slice.from_slice(i, s) if isinstance(i, slice)
                    else i for i, s in zip(indices, shape))
    if any(isinstance(i, slice) and i != slice(None) for i in indices):
      raise NotImplementedError("Non-`slice(None)` slices not supported yet.")
    if len(indices) != len(shape):
      raise ValueError("Must provide indexer for each dimension of `Ref`.")
    is_int_indexing = [isinstance(i, (Array, int)) for i in indices]
    other_indexers, int_indexers = partition_list(is_int_indexing, indices)
    int_indexers = [np.array(i, np.int32) if isinstance(i, int) else i for i in
                    int_indexers]
    indexer_shapes = [i.shape for i in int_indexers]
    bcast_shape = tuple(s for i in indexer_shapes for s in i)
    idx_iter = iter(range(len(bcast_shape)))
    int_indexers = [
        lax.broadcast_in_dim(i, bcast_shape, tuple(next(idx_iter) for _ in
                                                   range(len(i.shape))))
        for i in int_indexers
    ]
    indices = merge_lists(is_int_indexing, other_indexers, int_indexers)
    return NDIndexer(tuple(indices), shape, bcast_shape)

  def get_indexer_shape(self) -> Tuple[int, ...]:
    is_int_indexing = [not isinstance(i, Slice) for i in self.indices]
    other_indexers, _ = partition_list(is_int_indexing, self.indices)
    other_shape = [s.size for s in other_indexers]
    return tuple((*self.int_indexer_shape, *other_shape))

load_p = jax_core.Primitive('masked_load')

def _load_abstract_eval(ref_aval, *all_avals, args_tree,
                        **params: Any):
  idx_aval, *_ = tree_util.tree_unflatten(args_tree, all_avals)
  return (jax_core.ShapedArray(idx_aval.get_indexer_shape(), ref_aval.dtype),
          {state.ReadEffect(ref_aval)})
load_p.def_effectful_abstract_eval(_load_abstract_eval)

swap_p = jax_core.Primitive('masked_swap')

def _swap_abstract_eval(ref_aval, val_aval, *all_avals, args_tree,
                        **_: Any):
  idx_aval, *_ = tree_util.tree_unflatten(args_tree, all_avals)
  expected_output_shape = idx_aval.get_indexer_shape()
  if expected_output_shape != val_aval.shape:
    raise ValueError("Invalid shape for `swap`. "
                     f"Ref shape: {ref_aval.shape}. "
                     f"Value shape: {val_aval.shape}. "
                     f"Indices: {idx_aval}. ")
  if ref_aval.dtype != val_aval.dtype:
    raise ValueError("Invalid dtype for `swap`. "
                     f"Ref dtype: {ref_aval.dtype}. "
                     f"Value shape: {val_aval.dtype}. ")
  return (jax_core.ShapedArray(expected_output_shape, ref_aval.dtype),
          {state.WriteEffect(ref_aval)})
swap_p.def_effectful_abstract_eval(_swap_abstract_eval)

def load(x_ref, idx, *, mask=None, other=None, cache_modifier="",
         eviction_policy="", volatile=False):
  idx = NDIndexer.from_indices_shape(idx, x_ref.shape)
  args = (idx,)
  if mask is not None:
    args = (*args, mask)
  if other is not None:
    assert mask is not None
    args = (*args, other)
  flat_args, args_tree = tree_util.tree_flatten(args)
  return load_p.bind(x_ref, *flat_args, masked=mask is not None, cache_modifier=cache_modifier,
                     eviction_policy=eviction_policy, is_volatile=volatile,
                     args_tree=args_tree)

def swap(x_ref, idx, val, *, mask=None, eviction_policy="") -> Any:
  idx = NDIndexer.from_indices_shape(idx, x_ref.shape)
  args = (idx,)
  if mask is not None:
    args = (*args, mask)
  flat_args, args_tree = tree_util.tree_flatten(args)
  return swap_p.bind(x_ref, val, *flat_args, masked=mask is not None,
                     eviction_policy=eviction_policy, args_tree=args_tree)

def store(x_ref, idx, val, *, mask=None, eviction_policy="") -> None:
  _ = swap(x_ref, idx, val, mask=mask, eviction_policy=eviction_policy)

def dot(a, b, trans_a=False, trans_b=False, allow_tf32=True):
  rhs_contract_dim = int(trans_b)
  lhs_contract_dim = int(not trans_a)
  return jax.lax.dot_general(
      a, b, dimension_numbers=(((lhs_contract_dim,), (rhs_contract_dim,)), ((), ())),
      precision=lax.Precision.HIGH if allow_tf32 else lax.Precision.HIGHEST,
      preferred_element_type=None)
