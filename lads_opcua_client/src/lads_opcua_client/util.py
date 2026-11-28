"""Small helper functions used across the LADS OPC UA client library."""

from __future__ import annotations

import datetime as dt
from asyncua import Node, ua
from asyncua.common.ua_utils import get_node_supertypes, is_subtype
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import Server


async def get_parent_nodes(server: Server, node: Node, root_node: Node = None) -> list[Node]:
    if node is None:
        return[]
    parent = await node.get_parent()
    if parent == root_node:
        return [parent]
    else:
        return [parent] + await get_parent_nodes(server, parent, root_node)


async def browse_types(server: Server, node: Node) -> list[Node]:
    type_node_id =  await node.read_type_definition()
    type_node = server.get_node(type_node_id)
    return await get_node_supertypes(type_node, includeitself=True)


async def is_of_type(server: Server, node: Node, super_type_node: Node) -> bool:
    type_node_id = await node.read_type_definition()
    type_node = server.client.get_node(type_node_id)
    result = await is_subtype(type_node, super_type_node.nodeid)
    return result


unique_name_delimiter = "/"


def variant_value_to_str(variant: ua.Variant) -> str:
    if variant is None:
        return "unknown"
    value = variant.Value
    if isinstance(value,ua.LocalizedText):
        return value.Text if value.Text is not None else ""
    elif isinstance(value, ua.QualifiedName):
        return  value.Name
    elif isinstance(value, dt.datetime):
        return  value.strftime("%d.%m.%Y %H:%M:%S")
    else:
        s = str(value)
        if s.startswith("NameNodeIdDataType"):
            try:
                name: ua.LocalizedText = value.Name
                node_id: ua.NodeId = value.NodeId
                return name.Text
            except:
                return s
        else:
            return s


def duration_to_str(ms: float, ms_digits: int = 3) -> str:
    if ms is None:
        return ""
    if not 0 <= ms_digits <= 3:
        raise ValueError("ms_digits must be between 0 and 3")
    factor = 10 ** (3 - ms_digits)
    total_ms = int(round(ms / factor) * factor)

    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)

    result = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    if ms_digits:
        fraction = millis // factor
        result += f",{fraction:0{ms_digits}d}"

    return result


def remove_none(nodes: list[Node]) -> list[Node]:
    return list(filter(lambda node: node is not None, nodes))
