"""LADSNode: base class of all nodes in the LADS OPC UA model, and the promotion of generic nodes."""

from __future__ import annotations

import asyncio
from asyncua import Node, ua
from asyncua.common.ua_utils import is_subtype
from typing import Any, List, Self, TYPE_CHECKING, Type
from .ids import SubscriptionLevel
from .log import _logger
from .util import remove_none

if TYPE_CHECKING:
    from .server import Server
    from .variables import BaseVariable


AFOSupport = True


if AFOSupport:
    from .afo import DictionaryEntry, get_entry


# MARK: LADSNode
class LADSNode(Node):
    """
    Represents a node in the LADS OPC UA model.

    Attributes:
        server: Server - the server object.
        browse_name: ua.QualifiedName - the browse name of the node.
        display_name: str - the display name of the node.
        description: str - the description of the node.
        unique_name: str - the unique name of the node.
        variables: list[BaseVariable] - the variables of the node.
        subscribed_variables: list[BaseVariable] - the subscribed variables of the node.
        permanent_subscribed_variables: list[BaseVariable] - the permanent subscribed variables of the node.
        temporary_subscribed_variables: list[BaseVariable] - the temporary subscribed variables of the node.

    Methods:
        promote(cls, node: Node, server: Server) -> Self - promote a asyncua node to a LADSNode.
        variable_named(self, name: str) -> BaseVariable - get a variable by name.
        update_variables_async(self) - update the variables of the node asynchronously.
        update_variables(self) - update the variables of the node.
        call_async(self, func) - call a function asynchronously.
        get_child_or_none(self, name: ua.QualifiedName) -> Node - get a child node or None by name.
        get_di_child(self, name: str) -> Node - get a DI child node by name.
        get_di_variable(self, name: str) -> BaseVariable - get a DI variable by name.
        get_machinery_child(self, name: str) -> Node - get a machinery child node by name.
        get_machinery_variable(self, name: str) -> BaseVariable - get a machinery variable by name.
        get_lads_child(self, name: str) -> Node - get a LADS child node by name.
        get_lads_variable(self, name: str) -> BaseVariable - get a LADS variable by name.
        get_child_objects(self, parent: Node = None) -> list[Node] - get all child objects of a node or parent node.
        call_lads_method(self, name: str, *args: Any) -> ua.StatusCode - call a LADS method by name.
    """

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(LADSNode, node, server.BaseObjectType, server)

    def __str__(self):
        return f"{self.__class__.__name__}({self.display_name})"

    async def init(self, server: Server):
        """
        Initialize the LADSNode.

        Args:
            server: Server - the server object.
        """

        self.server: Server = server
        (self.browse_name, self._display_name, self.description, self.dictionary_entries)  = await asyncio.gather(
            self.read_browse_name(),
            self.read_display_name(),
            self.read_description(),
            self.read_dictionary_entries()
        )
        if AFOSupport:
            self._dictionary_entry_objects = None
            self._dcitionary_entries_as_markdown = None
        msg = f"Initializing {self.__class__.__name__}({self.display_name})"
        _logger.info(msg)

    async def finalize_init(self):
        pass

    ###################################################
    # Work around for buggy _to_nodeid() implementation
    # Issue number: TBD
    async def get_references(
        self,
        refs: int = ua.ObjectIds.References,
        direction: ua.BrowseDirection = ua.BrowseDirection.Both,
        nodeclassmask: ua.NodeClass = ua.NodeClass.Unspecified,
        includesubtypes: bool = True,
        result_mask: ua.BrowseResultMask = ua.BrowseResultMask.All
    ) -> List[ua.ReferenceDescription]:
        """
        returns references of the node based on specific filter defined with:

        refs = ObjectId of the Reference
        direction = Browse direction for references
        nodeclassmask = filter nodes based on specific class
        includesubtypes = If true subtypes of the reference (ref) are also included
        result_mask = define what results information are requested
        """
        def _to_nodeid(nodeid: int):
            if nodeid <= 255:
                return ua.TwoByteNodeId(nodeid)
            elif nodeid <= 65535:
                return ua.FourByteNodeId(nodeid)
            else:
                return ua.NumericNodeId(nodeid)

        desc = ua.BrowseDescription()
        desc.BrowseDirection = direction
        desc.ReferenceTypeId = _to_nodeid(refs)
        desc.IncludeSubtypes = includesubtypes
        desc.NodeClassMask = nodeclassmask
        desc.ResultMask = result_mask
        desc.NodeId = self.nodeid
        params = ua.BrowseParameters()
        params.View.Timestamp = ua.get_win_epoch()
        params.NodesToBrowse.append(desc)
        params.RequestedMaxReferencesPerNode = 0
        results = await self.session.browse(params)
        references = await self._browse_next(results)
        return references


    @property
    def display_name(self) -> str:
        if self._display_name is not None:
            return self._display_name.Text
        else:
            return self.browse_name.Name

    @property
    def unique_name(self) -> str:
        return self.display_name

    @property
    def variables(self) ->list[BaseVariable]:
        return []

    @property
    def subscribed_variables(self) ->list[BaseVariable]:
        return list(filter(lambda variable: variable.subscription_level > SubscriptionLevel.Never, self.variables))

    @property
    def permanent_subscribed_variables(self) ->list[BaseVariable]:
        return list(filter(lambda variable: variable.subscription_level == SubscriptionLevel.Permanent, self.variables))

    @property
    def temporary_subscribed_variables(self) ->list[BaseVariable]:
        return list(filter(lambda variable: variable.subscription_level == SubscriptionLevel.Temporary, self.variables))

    def variable_named(self, name: str) -> BaseVariable:
        for variable in self.variables:
            if name == variable.browse_name.Name:
                return variable
        return None

    async def update_variables_async(self):
        variables = remove_none(self.variables)
        await asyncio.gather(*(variable.update_value() for variable in variables))

    def update_variables(self):
        self.call_async(self.update_variables_async())

    def call_async(self, func):
        self.server.call_async_queue.put(func)

    async def get_child_or_none(self, name : ua.QualifiedName) -> Node:
        try:
            return await self.get_child(name)
        except:
            return None

    async def get_di_child(self, name : str) -> Node:
        return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_DI))

    async def get_di_variable(self, name : str) -> BaseVariable:
        from .variables import BaseVariable  # lazy: avoids a circular import
        return await BaseVariable.promote(await self.get_di_child(name), self.server)

    async def get_amb_child(self, name : str) -> Node:
        return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_AMB))

    async def get_amb_variable(self, name : str) -> BaseVariable:
        from .variables import BaseVariable  # lazy: avoids a circular import
        return await BaseVariable.promote(await self.get_amb_child(name), self.server)

    async def get_machinery_child(self, name : str) -> Node:
        return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_Machinery))

    async def get_machinery_variable(self, name : str) -> BaseVariable:
        from .variables import BaseVariable  # lazy: avoids a circular import
        return await BaseVariable.promote(await self.get_machinery_child(name), self.server)

    async def get_lads_child(self, name : str) -> Node:
        return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_LADS))

    async def get_lads_variable(self, name : str) -> BaseVariable:
        from .variables import BaseVariable  # lazy: avoids a circular import
        return await BaseVariable.promote(await self.get_lads_child(name), self.server)

    async def get_child_objects(self, parent: Node = None) -> list[Node]:
        if parent is None: parent = self
        # search for HasChild and Organizes references
        (has_child_objects, organizes_objects) = await asyncio.gather(
            parent.get_children(refs = ua.ObjectIds.Aggregates, nodeclassmask = ua.NodeClass.Object),
            parent.get_children(refs = ua.ObjectIds.Organizes, nodeclassmask = ua.NodeClass.Object)
        )
        # reduce results to set
        child_objects = set(has_child_objects)
        child_objects.update(organizes_objects)
        return list(child_objects)


    async def call_namespace_method(self, name: str, ns: int, *args: Any) -> ua.StatusCode:
        try:
            converted_args = tuple(
                ua.Variant([], VariantType=ua.VariantType.ExtensionObject)
                if isinstance(arg, list) and len(arg) == 0
                else arg
                for arg in args
            )
            _logger.debug(f"Call method {name} with args {converted_args}")
            return await self.call_method(
                ua.QualifiedName(name, ns),
                *converted_args,
            )
        except Exception as error:
            _logger.error(error)
            return ua.StatusCodes.BadNotImplemented

    async def call_lads_method(self, name: str, *args: Any) -> ua.StatusCode:
        return await self.call_namespace_method(name, self.server.ns_LADS, *args)

    async def call_di_method(self, name: str, *args: Any) -> ua.StatusCode:
        return await self.call_namespace_method(name, self.server.ns_DI, *args)

    if AFOSupport:
        @property
        def dictionary_entry_objects(self) -> list[DictionaryEntry]:
            if self._dictionary_entry_objects is None:
                self._dictionary_entry_objects: list[DictionaryEntry] = list(filter(lambda entry: entry is not None, map(lambda dictionary_entry: get_entry(dictionary_entry), self.dictionary_entries)))

            return self._dictionary_entry_objects

    @property
    def dictionary_entries_as_markdown(self) -> str:
        if AFOSupport:
            if self._dcitionary_entries_as_markdown is None:
                definitions: list[str] = []
                for entry in self.dictionary_entry_objects:
                    if entry is not None:
                        markdown = f"**{entry.prefLabel}**  \r\n{entry.definition}  "
                    definitions.append(markdown)
                self._dcitionary_entries_as_markdown = "\n\r".join(definitions)
            return self._dcitionary_entries_as_markdown
        else:
            return ""

    async def read_dictionary_entries(self) -> list[str]:
        if AFOSupport:
            try:
                nodes = await self.get_referenced_nodes(refs = ua.ObjectIds.HasDictionaryEntry, direction = ua.BrowseDirection.Forward, nodeclassmask = ua.NodeClass.Object)
                if  len(nodes) == 0:
                    return []
                names = list(map(lambda node: node.nodeid.Identifier, nodes))
                return names
            except Exception as err:
                print(f"Exception when reading dictionary entries {err}")
                return []
        else:
            return []


#MARK: Promotion of generic OPC UA nodes to specfic LADS objects
async def promote_to(cls: Type, node: Node, super_type_node: Node, server: Server) -> LADSNode:
    if node is None: return None
    node_class = await node.read_node_class()
    if node_class != ua.NodeClass.Method:
        type = await node.read_type_definition()
        type_node = server.client.get_node(type)
        result = await is_subtype(type_node, super_type_node.nodeid)
        assert result, f"node {node.nodeid} is expexted to be of type {type_node.nodeid}"
    node.__class__ = cls
    # promoted_node : cls = node
    promoted_node = node
    await promoted_node.init(server)
    return promoted_node
