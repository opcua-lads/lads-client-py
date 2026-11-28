"""Compliance documents of a device."""

from __future__ import annotations

import asyncio
from asyncua import Node, ua
from asyncua.client.ua_file import UaFile
from typing import Self, TYPE_CHECKING
from .log import _logger
from .node import LADSNode, promote_to
from .program import Result
from .sets import LADSSet

if TYPE_CHECKING:
    from .server import Server


# MARK: class ComplianceDocumentSet(LADSSet):
class ComplianceDocumentSet(LADSSet):
    # since the Machinery type Components is not derived from LADS.SetType we need a different type check
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(ComplianceDocumentSet, node, server.ComplianceDocumentSetType, server)


# MARK: ComplianceDocument
class ComplianceDocument(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Result, node, server.ComplianceDocumentType, server)

    async def init(self, server: Server):
        self.applies_to = []
        self.references_markdown = []

        await super().init(server)
        self.document_name = await self.get_lads_cd_variable("DocumentName")
        self.issued_at = await self.get_lads_cd_variable("IssuedAt")
        self.valid_from = await self.get_lads_cd_variable("ValidFrom")
        self.valid_until = await self.get_lads_cd_variable("ValidUntil")
        self.mime_type = await self.get_lads_cd_variable("MimeType")
        self.content = await self.get_lads_cd_variable("Content")
        self.schema_uri = await self.get_lads_cd_variable("SchemaUri")
        self.file = await self.get_lads_cd_child("File")
        ref_type: ua.ReferenceDescription = server.HasComplianceDocument
        references = await self.get_references_of_type(ref_type)
        for desc in references:
            node = server.get_node(desc.NodeId)
            type_node_id = await node.read_type_definition()
            type_node = server.get_node(type_node_id)
            ref_type_node = server.get_node(desc.ReferenceTypeId)
            self.applies_to.append(node)
            node_name, node_type_name, ref_type_name = await asyncio.gather(node.read_display_name(), type_node.read_display_name(), ref_type_node.read_display_name())
            markdown = f"**{node_name.Text}**: *{node_type_name.Text}* -> {ref_type_name.Text} -> **{self.display_name}**"
            self.references_markdown.append(markdown)

    async def download(self):
        try:
            self.downloading = True
            _logger.debug(f"{self.display_name} start downloading file data ..")
            async with UaFile(self.file, "r") as ua_file:
                self.data = await ua_file.read()
                _logger.debug(f"{self.display_name} finished downloading file data ..")
                self.downloading = False
        except:
            _logger.error(f"{self.display_name} failed reading file")

    def has_data(self) -> bool:
        return self.data is not None

    def fetch_data(self):
        if self.downloading:
            _logger.debug(f"{self.display_name} download already active ..")
        else:
            _logger.debug(f"{self.display_name} fetching file data ..")
            self.call_async(self.download())
    """
    # Since all information is static don't attach it to a subscription group
    @property
    def variables(self) ->list[BaseVariable]:
        return remove_none([self.document_name, self.issued_at, self.valid_from, self.valid_until, self.mime_type, self.content])
    """
