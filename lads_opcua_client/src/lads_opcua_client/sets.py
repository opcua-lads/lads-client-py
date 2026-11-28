"""Sets of LADS objects (LADSSet, ComponentSet, VariableSet)."""

from __future__ import annotations

import asyncio
from asyncua import Node
from typing import Self, TYPE_CHECKING, Type
from .log import _logger
from .node import LADSNode, promote_to
from .util import is_of_type, unique_name_delimiter
from .variables import NodeVersionVariable, get_properties_and_variables

if TYPE_CHECKING:
    from .server import Server
    from .variables import BaseVariable


# MARK: LADSSet
class LADSSet(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(LADSSet, node, server.SetType, server)

    async def init(self, server: Server):
        self.node_version: NodeVersionVariable = None
        await super().init(server)
        try:
            # node_version variable is optional
            node_version = await self.get_child("NodeVersion")
            self.node_version = await NodeVersionVariable.promote(node_version, server)
            self.node_version.set = self
        except Exception as error:
             _logger.warning(error)
        self.children = await self.get_child_objects()

    async def promote_children(self, child_class: Type, child_type: Node, set_type: Node):
        if self.children is None:
            return
        if set_type is not None:
            assert(await is_of_type(self.server, self, set_type))
        self.child_class = child_class
        self.child_type = child_type
        self.children = await asyncio.gather(*(self.promote_child(child) for child in self.children))
        self.children.sort(key = lambda child: child.display_name)

    async def promote_child(self, child: Node) -> LADSNode:
        return await promote_to(self.child_class, child, self.child_type, self.server)

    @property
    def variables(self) ->list[BaseVariable]:
        return [] if self.node_version is None else [self.node_version]

    def node_version_changed(self):
        _logger.debug(f"NodeVersion of {self.display_name} changed.")
        self.call_async(self.update_children())

    async def update_children(self):
        current_nodes = await self.get_child_objects(self)
        current_node_ids = set(map(lambda node: node.nodeid, current_nodes))
        previous_nodes = self.children
        previous_node_ids = set(map(lambda node: node.nodeid, previous_nodes))
        new_node_ids = current_node_ids.difference(previous_node_ids)
        deleted_node_ids = previous_node_ids.difference(current_node_ids)
        if len(new_node_ids) > 0:
            for node_id in new_node_ids:
                nodes = list(filter(lambda node: node.nodeid == node_id, current_nodes))
                assert(len(nodes) == 1)
                node = await self.promote_child(nodes[0])
                self.children.append(node)
        if len(deleted_node_ids) > 0:
            for node_id in deleted_node_ids:
                nodes = list(filter(lambda node: node.nodeid == node_id, previous_nodes))
                assert(len(nodes) == 1)
                node = nodes[0]
                self.children.remove(node)


# MARK: ComponentSet
class ComponentSet(LADSSet):
    # since the Machinery type Components is not derived from LADS.SetType we need a different type check
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(ComponentSet, node, server.ComponentSetType, server)


# MARK: VariableSet
class VariableSet(LADSSet):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(VariableSet, node, server.VariableSetType, server)

    async def init(self, server: Server):
        await super().init(server)
        await self.update_children()

    async def update_children(self):
        self._variables: list[BaseVariable] = []
        await self.collect_variables(self)

    async def collect_variables(self, lads_node: LADSNode, path: str = ""):
        # collect variables and properties of current node
        variables = await get_properties_and_variables(lads_node)
        for variable in variables:
            if not "NodeVersion" in variable.display_name:
                variable.alternate_display_name = unique_name_delimiter.join([path, variable.display_name])
                self._variables.append(variable)
        # recurse objects if any
        nodes = await self.get_child_objects(lads_node)
        for node in nodes:
            parent = await LADSNode.promote(node, self.server)
            await self.collect_variables(parent, unique_name_delimiter.join([path, parent.display_name]))

    @property
    def variables(self) -> list[BaseVariable]:
        return self._variables
