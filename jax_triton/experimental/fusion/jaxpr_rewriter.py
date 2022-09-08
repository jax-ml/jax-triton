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

"""Contains utilities for rewriting jaxprs."""
from __future__ import annotations

import abc
import dataclasses
import itertools as it

from typing import Any, Callable, Dict, List, Set, Tuple, Union

from jax import core as jax_core
import jax.numpy as jnp

from oryx.experimental.matching import matcher
from oryx.experimental.matching import jax_rewrite as jr

Expr = matcher.Expr
Bindings = matcher.Bindings
Continuation = matcher.Continuation
Success = matcher.Success

class Node(matcher.Pattern, metaclass=abc.ABCMeta):

  @abc.abstractproperty
  def parents(self) -> List[Node]:
    ...


  @abc.abstractmethod
  def set_parent(self, node, new_node):
    ...

  @abc.abstractmethod
  def map_parents(self, fn: Callable[[Node], Node]) -> Node:
    ...

@dataclasses.dataclass(eq=False)
class Eqn(Node):
  primitive: jax_core.Primitive
  params: jr.Params
  invars: List[Node]
  shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]
  dtype: Union[jnp.dtype, List[jnp.dtype]]

  @property
  def parents(self):
    return self.invars

  def set_parent(self, node, new_node):
    invar_idx = self.invars.index(node)
    self.invars[invar_idx] = new_node

  def map_parents(self, fn):
    return Eqn(self.primitive, self.params, list(map(fn, self.invars)),
               self.shape, self.dtype)

  def match(self, expr, bindings, succeed):
    if not isinstance(expr, Eqn):
      return
    yield from matcher.matcher((self.primitive, self.params, self.invars,
      self.shape, self.dtype))(
        (expr.primitive, expr.params, expr.invars, expr.shape, expr.dtype),
        bindings, succeed)

@dataclasses.dataclass(frozen=True, eq=False)
class JaxprVar(Node):
  shape: Tuple[int, ...]
  dtype: jnp.dtype

  def match(self, expr, bindings, succeed):
    if expr is self:
      yield from succeed(bindings)

  @property
  def parents(self):
    return []

  def set_parent(self, node, new_node):
    raise NotImplementedError

  def map_parents(self, fn):
    return self

  @classmethod
  def from_var(cls, var: jax_core.Var) -> JaxprVar:
    return JaxprVar(var.aval.shape, var.aval.dtype)

@dataclasses.dataclass(eq=False, frozen=True)
class Literal(Node):
  value: Any
  dtype: jnp.dtype

  @property
  def parents(self):
    return []

  def map_parents(self, fn, visited):
    return self

  def set_parent(self, node, new_node):
    raise NotImplementedError

  @property
  def shape(self):
    return ()

  def match(self, expr, bindings, succeed):
    if not isinstance(expr, Literal):
      return []
    yield from matcher.matcher((self.value, self.dtype))((expr.value,
      expr.dtype), bindings, succeed)

  @classmethod
  def from_literal(cls, var: jax_core.Literal) -> Literal:
    return Literal(var.val, var.aval.dtype)


@dataclasses.dataclass(eq=False)
class Part(Node):
  index: int
  shape: Tuple[int, ...]
  dtype: jnp.dtype
  parent: Node

  def match(self, expr, bindings, succeed):
    if not isinstance(expr, Part):
      return []
    yield from matcher.matcher((self.index, self.shape, self.dtype, self.parent))((
      expr.index, expr.shape, expr.dtype, expr.parent), bindings, succeed)

  def set_parent(self, _, new_node):
    self.parent = new_node

  @property
  def parents(self):
    return [self.parent]

  def map_parents(self, fn):
    return Part(self.index, self.shape, self.dtype, fn(self.parent))

@dataclasses.dataclass(eq=True)
class JaxprGraph(matcher.Pattern):
  constvars: List[Node]
  invars: List[Node]
  outvars: List[Node]

  def get_nodes(self):
    nodes = set(self.outvars)
    queue = list(self.outvars)
    while queue:
      node = queue.pop(0)
      nodes.add(node)
      for p in node.parents:
        queue.append(p)
    return nodes

  def get_children(self, node) -> List[Node]:
    nodes = self.get_nodes()
    return [n for n in nodes if node in n.parents]

  def rewrite_subgraph(self, pattern, handler) -> bool:
    queue = list(self.outvars)
    while queue:
      node = queue.pop(0)
      assert isinstance(node, Node)
      try:
        match = matcher.match(pattern, node)
        new_node = handler(**match)
        if node in self.outvars:
          i = self.outvars.index(node)
          self.outvars[i] = new_node
        elif isinstance(node, Eqn):
          children = self.get_children(node)
          assert children
          for c in children:
            c.set_parent(node, new_node)
        else:
          raise NotImplementedError
        return True
      except matcher.MatchError:
        queue.extend(node.parents)
        for p in node.parents:
          if isinstance(p, Eqn):
            assert self.get_children(p)
    return False

  @classmethod
  def from_jaxpr(cls, jaxpr: jax_core.Jaxpr) -> JaxprGraph:
    var_mapping = {}
    for var in it.chain(jaxpr.constvars, jaxpr.invars):
      node = JaxprVar.from_var(var)
      var_mapping[var] = node
    for eqn in jaxpr.eqns:
      invars = []
      for invar in eqn.invars:
        if isinstance(invar, jax_core.Literal):
          node = Literal.from_literal(invar)
        else:
          node = var_mapping[invar]
        invars.append(node)
      if eqn.primitive.multiple_results:
        node = Eqn(eqn.primitive, jr.Params(eqn.params), invars,
                   [o.aval.shape for o in eqn.outvars],
                   [o.aval.dtype for o in eqn.outvars])
        for i, outvar in enumerate(eqn.outvars):
          part = Part(i, outvar.aval.shape, outvar.aval.dtype, node)
          var_mapping[outvar] = part
      else:
        node = Eqn(eqn.primitive, jr.Params(eqn.params), invars,
                   eqn.outvars[0].aval.shape, eqn.outvars[0].aval.dtype)
        var_mapping[eqn.outvars[0]] = node
    constvars = [var_mapping[constvar] for constvar in jaxpr.constvars]
    invars = [var_mapping[invar] for invar in jaxpr.invars]
    outvars = [var_mapping[outvar] for outvar in jaxpr.outvars]
    return JaxprGraph(constvars, invars, outvars)

  def to_jaxpr(self) -> jax_core.Jaxpr:
    gen = jax_core.gensym()
    eqns = []
    sorted_nodes = self.toposort()
    env = {}
    for var in it.chain(self.invars, self.constvars):
      env[var] = gen(jax_core.ShapedArray(var.shape, var.dtype))
    incomplete_eqns = {}
    for node in sorted_nodes:
      if isinstance(node, Literal):
        continue
      elif isinstance(node, JaxprVar):
        assert node in env
        continue
      elif isinstance(node, Eqn):
        invars = []
        for n in node.invars:
          if isinstance(n, Literal):
            invars.append(jax_core.Literal(n.value, jax_core.ShapedArray((),
              n.dtype)))
          else:
            invars.append(env[n])
        jaxpr_eqn = jax_core.JaxprEqn(invars, [], node.primitive,
            dict(node.params), jax_core.no_effects, None)
        if node.primitive.multiple_results:
          incomplete_eqns[node] = jaxpr_eqn
        else:
          outvar = gen(jax_core.ShapedArray(node.shape, node.dtype))
          env[node] = outvar
          jaxpr_eqn = jaxpr_eqn.replace(outvars=[outvar])
          incomplete_eqns[node] = jaxpr_eqn
      elif isinstance(node, Part):
        eqn = node.parent
        incomplete_eqn = incomplete_eqns[eqn]
        outvars = list(incomplete_eqn.outvars)
        if len(outvars) <= node.index:
          outvars = outvars + [None] * (node.index - len(outvars) + 1)
        outvar = gen(jax_core.ShapedArray(node.shape, node.dtype))
        outvars[node.index] = outvar
        env[node] = outvar
        incomplete_eqns[eqn] = incomplete_eqn.replace(outvars=outvars)
    eqns = list(incomplete_eqns.values())
    constvars = [env[n] for n in self.constvars]
    invars = [env[n] for n in self.invars]
    outvars = [env[n] for n in self.outvars]
    return jax_core.Jaxpr(constvars, invars, outvars, eqns, jax_core.no_effects)

  def toposort(self) -> List[Node]:
    node_stack = list(self.outvars)
    child_counts = {}
    while node_stack:
      node = node_stack.pop()
      if node in child_counts:
        child_counts[node] += 1
      else:
        child_counts[node] = 1
        node_stack.extend(node.parents)
    for node in self.outvars:
      child_counts[node] -= 1
    childless_nodes = [node for node in self.outvars if child_counts[node] == 0]
    sorted_nodes = []
    while childless_nodes:
      node = childless_nodes.pop()
      sorted_nodes.append(node)
      for parent in node.parents:
        if child_counts[parent] == 1:
          childless_nodes.append(parent)
        else:
          child_counts[parent] -= 1
    return list(reversed(sorted_nodes))
