"""Cover function."""

from __future__ import annotations

from asyncua import Node
from typing import Self, TYPE_CHECKING
from .base import BaseStateMachineFunction
from ..node import promote_to
from ..state_machine import FunctionalStateMachine

if TYPE_CHECKING:
    from ..server import Server
    from ..state_machine import StateMachine


# MARK: CoverFunction
class CoverFunction(BaseStateMachineFunction):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(CoverFunction, node, server.CoverFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.cover_state = await FunctionalStateMachine.promote(await self.get_lads_child("CoverState"), server)

    @property
    def state_machine(self) -> StateMachine:
        return self.cover_state
