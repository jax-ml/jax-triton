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

"""Module for calling pallas functions from JAX."""
from functools import partial

from typing import Dict, Tuple

import jax
from jax import api_util
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import mlir
from jax.interpreters import xla
from jax._src import ad_util
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
from jax._src import state
from jax._src.util import (
    split_list, safe_map, safe_zip)
from jax._src.lax.control_flow import for_loop

from triton._C.libtriton import triton as tc

from jax_triton.triton_call import emit_triton_kernel_call, avals_to_layouts
from jax_triton.pallas import lowering

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

pallas_call_p = jax_core.Primitive('pallas_call')
pallas_call_p.multiple_results = True

pallas_call_p.def_impl(partial(xla.apply_primitive, pallas_call_p))

def _pallas_call_abstract_eval(*avals, out_shapes, **_):
  return map(lambda x: jax_core.ShapedArray(x.shape, x.dtype), out_shapes)
pallas_call_p.def_abstract_eval(_pallas_call_abstract_eval)

def _pallas_call_jvp_rule(primals, tangents, *, jaxpr, name, which_linear,
    input_output_aliases: Tuple[Tuple[int, int], ...],
    out_shapes, grid, debug, **metaparams):
  if input_output_aliases:
    raise NotImplementedError("JVP with aliasing not supported.")
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  tangents = [ad.instantiate_zeros(t) if inst else t
              for t, inst in zip(tangents, nonzero_tangents)]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  nonzero_tangents_with_outputs = nonzero_tangents + [True] * len(out_shapes)
  closed_jaxpr = jax_core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, nonzero_tangents_with_outputs, [])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  jvp_which_linear = which_linear + (True,) * len(tangents)
  jvp_outshapes = (*out_shapes, *out_shapes)
  invars, outvars = [], []
  inputs = []
  for i in range(len(jvp_jaxpr.invars)):
    if i % 2 == 0:
      # Primal
      invars.append(jvp_jaxpr.invars[i])
    else:
      # Tangent
      outvars.append(jvp_jaxpr.invars[i])
  for i in range(len(primals)):
    inputs.append(primals[i])
  jvp_jaxpr = jvp_jaxpr.replace(invars=[*invars, *outvars])
  out_flat = pallas_call_p.bind(*primals, *tangents, jaxpr=jvp_jaxpr,
                                name=f"{name}_jvp",
                                out_shapes=jvp_outshapes,
                                which_linear=jvp_which_linear,
                                grid=grid, debug=debug, **metaparams)
  # `out_flat` includes constant inputs into the `for_loop` which are converted
  # into outputs as well. We don't care about these in AD so we throw them out.
  out_primals, out_tangents = split_list(out_flat, [len(out_flat) // 2])
  return out_primals, out_tangents
ad.primitive_jvps[pallas_call_p] = _pallas_call_jvp_rule


def pallas_call_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, jaxpr, name,
                         out_shapes, which_linear, grid, num_warps, num_stages,
                         debug: bool,
                         input_output_aliases: Tuple[Tuple[int, int], ...],
                         **metaparams):
  del which_linear
  lowering_result = lowering.lower_jaxpr_to_triton_module(jaxpr, name)
  if debug:
    lowering_result.module.print()
  backend = tc.runtime.backend.CUDA
  device = 0
  name, asm, shared_mem = tc.code_gen.compile_ttir(backend, lowering_result.module, device,
      num_warps, num_stages, {}, 0)
  out_type = ir.TupleType.get_tuple([
      ir.RankedTensorType.get(out_shape.shape, mlir.dtype_to_ir_type(out_shape.dtype))
      for out_shape in ctx.avals_out])
  i32_type = ir.IntegerType.get_signless(32)
  descriptor, keepalive = emit_triton_kernel_call(
      ctx, name, asm, shared_mem, num_warps=num_warps, grid=grid,
      metaparams=metaparams, dump_binary_path=None)
  ctx.module_context.add_keepalive(keepalive)
  output_operand_aliases = ir.ArrayAttr.get([
          mhlo.OutputOperandAlias.get(
              output_tuple_indices=[output],
              operand_index=input,
              operand_tuple_indices=[])
          for input, output in input_output_aliases
      ])
  out = mhlo.CustomCallOp(
            [out_type], in_nodes,
            call_target_name=ir.StringAttr.get("triton_kernel_call"),
            has_side_effect=ir.BoolAttr.get(False),
            backend_config=ir.StringAttr.get(descriptor),
            api_version=ir.IntegerAttr.get(i32_type, 1),
            called_computations=ir.ArrayAttr.get([]),
            operand_layouts=avals_to_layouts(ctx.avals_in),
            result_layouts=avals_to_layouts(ctx.avals_out),
            output_operand_aliases=output_operand_aliases)
  results = [mhlo.GetTupleElementOp(out, mlir.i32_attr(i)).result
             for i in range(len(out_shapes))]
  return results
mlir.register_lowering(pallas_call_p, pallas_call_lowering)

def pallas_call(f, out_shape, grid, *, debug: bool = False,
                num_warps: int = 4, num_stages: int = 3,
                input_output_aliases: Dict[int, int] = {},
                **metaparams):
  if isinstance(grid, int):
    grid = (grid,)
  name = f.__name__ if hasattr(f, "__name__") and f.__name__ else"func"
  flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)
  flat_out_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in
                     flat_out_shapes]
  def wrapped(*args, **kwargs):
    flat_args, _ = tree_util.tree_flatten((args, kwargs))
    in_tree = tree_util.tree_structure((*args, *flat_out_shapes))
    flat_avals = map(
        jax_core.raise_to_shaped, map(jax_core.get_aval, flat_args))
    flat_output_avals = [jax_core.ShapedArray(out_shape.shape, out_shape.dtype)
        for out_shape in flat_out_shapes]
    ptr_avals = [state.ShapedArrayRef(aval.shape, aval.dtype) for aval in flat_avals]
    out_ptr_avals = [state.ShapedArrayRef(aval.shape, aval.dtype) for aval in flat_output_avals]
    flat_fun, _ = api_util.flatten_fun_nokwargs(lu.wrap_init(f), in_tree)
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, [*ptr_avals, *out_ptr_avals])
    jaxpr = for_loop._hoist_consts_to_refs(jaxpr)
    if debug:
      print(jaxpr)

    which_linear = (False,) * len(flat_args)
    out_flat = pallas_call_p.bind(
        *consts, *flat_args, jaxpr=jaxpr, name=name, which_linear=which_linear,
        out_shapes=tuple(flat_out_shapes), grid=grid, debug=debug,
        num_warps=num_warps,
        input_output_aliases=tuple(input_output_aliases.items()),
        num_stages=num_stages, **metaparams)
    return tree_util.tree_unflatten(out_tree, out_flat)
  return wrapped
