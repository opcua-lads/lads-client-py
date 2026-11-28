"""AlarmMonitor: alarm limits and states of a function."""

from __future__ import annotations

import asyncio
from asyncua import Node
from typing import Self, TYPE_CHECKING
from .log import _logger
from .node import LADSNode, promote_to
from .util import remove_none
from .variables import StateVariable, SubscribedVariable

if TYPE_CHECKING:
    from .server import Server
    from .variables import BaseVariable


# MARK: AlarmMonitor
class AlarmMonitor(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(AlarmMonitor, node, server.ExclusiveLimitAlarmType, server)

    async def init(self, server: Server):
        await super().init(server)

        names = ["ActiveState", "HighHighLimit", "HighLimit", "LowLimit", "LowLowLimit"]
        promoters = [
            StateVariable.promote,
            SubscribedVariable.promote,
            SubscribedVariable.promote,
            SubscribedVariable.promote,
            SubscribedVariable.promote,
        ]
        nodes = await asyncio.gather(*(self.get_child(name) for name in names))
        results = await asyncio.gather(*(
            promote(node, server) for node, promote in zip(nodes, promoters) if node is not None
        ))

        result_iter = iter(results)
        values = [next(result_iter) if node is not None else None for node in nodes]
        (
            self.alarm_active_state,
            self.high_high_limit,
            self.high_limit,
            self.low_limit,
            self.low_low_limit,
        ) = values

        if self.alarm_active_state is None:
            _logger.error(f"Missing required ActiveState child on {self!r}")

        limit_state = await self.get_child("LimitState")
        if limit_state is not None:
            self.alarm_limit_state = await StateVariable.promote(await limit_state.get_child("CurrentState"), server)

    @property
    def alarm_active(self) -> bool:
        if self.alarm_active_state is None:
            return False
        return self.alarm_active_state.value_str == "Active"

    @property
    def alarm_limit(self) -> str:
        if (self.alarm_limit_state is None):
            return ""
        return self.alarm_limit_state.value_str

    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + remove_none([
            self.alarm_active_state, self.alarm_limit_state,
            self.high_high_limit, self.high_limit, self.low_limit, self.low_low_limit
        ])
