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

from typing import Any, List, Tuple

import jax
from jax import core as jax_core
from jax import lax
from jax._src import ad_util
from jax._src.util import (safe_map, safe_zip, split_list, merge_lists,
                           partition_list)
from jax._src.state import primitives as state_primitives
from jax.interpreters import ad
import jax.numpy as jnp
import numpy as np

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

program_id_p = jax_core.Primitive("program_id")

def program_id(axis):
  return program_id_p.bind(axis=axis)

def _program_id_abstract_eval(**_):
  return jax_core.ShapedArray((), jnp.int32)
program_id_p.def_abstract_eval(_program_id_abstract_eval)

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

load_p = jax_core.Primitive('masked_load')

def _load_abstract_eval(ref_aval, *all_avals, indexed_dims: Tuple[bool],
                        **params: Any):
  idx_avals, _ = split_list(all_avals, [sum(indexed_dims)])
  return state_primitives._get_abstract_eval(
      ref_aval, *idx_avals, indexed_dims=indexed_dims)
load_p.def_effectful_abstract_eval(_load_abstract_eval)

def _load_jvp(primals: List[Any], tangents: List[Any], *, indexed_dims, **params):
  ref_primal, *idx_and_rest_primals = primals
  ref_tangent, *idx_and_rest_tangents = tangents
  idx_primals, _ = split_list(idx_and_rest_primals, [sum(indexed_dims)])
  _, rest_tangents = split_list(idx_and_rest_tangents, [sum(indexed_dims)])
  return (load_p.bind(ref_primal, *idx_and_rest_primals,
                      indexed_dims=indexed_dims, **params),
          load_p.bind(ref_tangent, *idx_primals, *rest_tangents,
                      indexed_dims=indexed_dims, **params))
ad.primitive_jvps[load_p] = _load_jvp

def _load_transpose(g, ref, *idx_and_rest, **params):
  raise NotImplementedError
ad.primitive_transposes[load_p] = _load_transpose

swap_p = jax_core.Primitive('masked_swap')

def _swap_abstract_eval(ref_aval, val_aval, *all_avals, indexed_dims: Tuple[bool],
                        **_: Any):
  idx_avals, _ = split_list(all_avals, [sum(indexed_dims)])
  return state_primitives._swap_abstract_eval(
      ref_aval, val_aval, *idx_avals, indexed_dims=indexed_dims)
swap_p.def_effectful_abstract_eval(_swap_abstract_eval)

def _swap_jvp(primals: List[Any], tangents: List[Any], *, indexed_dims, **params):
  ref_primal, val_primal, *idx_and_rest_primals = primals
  ref_tangent, val_tangent, *idx_and_rest_tangents = tangents
  idx_primals, _ = split_list(idx_and_rest_primals, [sum(indexed_dims)])
  _, rest_tangents = split_list(idx_and_rest_tangents, [sum(indexed_dims)])
  val_tangent = ad_util.instantiate(val_tangent)
  return (swap_p.bind(ref_primal, val_primal, *idx_and_rest_primals,
                      indexed_dims=indexed_dims, **params),
          swap_p.bind(ref_tangent, val_tangent, *idx_primals, *rest_tangents,
                      indexed_dims=indexed_dims, **params))
ad.primitive_jvps[swap_p] = _swap_jvp

def _swap_transpose(g, ref, *idx_and_rest, **params):
  raise NotImplementedError
ad.primitive_transposes[swap_p] = _swap_transpose

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

def load(x_ref, idx, *, mask=None, other=None, cache_modifier="",
         eviction_poicy="", volatile=False):
  idx = _process_idx(idx, x_ref.shape)
  idx, indexed_dims = state_primitives._unpack_idx(idx, x_ref.ndim)
  args = idx
  if mask is not None:
    args = (*args, mask)
  if other is not None:
    assert mask is not None
    args = (*args, other)
  return load_p.bind(x_ref, *args, masked=mask is not None, cache_modifier=cache_modifier,
                     eviction_policy=eviction_poicy, is_volatile=volatile,
                     indexed_dims=indexed_dims)


def swap(x_ref, idx, val, *, mask=None, eviction_policy="") -> Any:
  idx = _process_idx(idx, x_ref.shape)
  idx, indexed_dims = state_primitives._unpack_idx(idx, x_ref.ndim)
  args = idx
  if mask is not None:
    args = (*args, mask)
  return swap_p.bind(x_ref, val, *args, masked=mask is not None,
                     eviction_policy=eviction_policy, indexed_dims=indexed_dims)

def store(x_ref, idx, val, *, mask=None, eviction_policy="") -> None:
  _ = swap(x_ref, idx, val, mask=mask, eviction_policy=eviction_policy)

def dot(a, b, trans_a=False, trans_b=False, allow_tf32=True):
  rhs_contract_dim = int(trans_b)
  lhs_contract_dim = int(not trans_a)
  return jax.lax.dot_general(
      a, b, dimension_numbers=(((lhs_contract_dim,), (rhs_contract_dim,)), ((), ())),
      precision=lax.Precision.HIGH if allow_tf32 else lax.Precision.HIGHEST,
      preferred_element_type=None)
