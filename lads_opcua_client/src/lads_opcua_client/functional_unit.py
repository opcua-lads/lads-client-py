"""FunctionalUnit: the unit of a device which runs programs and holds its functions."""

from __future__ import annotations

import asyncio
from asyncua import Node
from asyncua.common.events import Event
from typing import Self, TYPE_CHECKING
from .component import Lock
from .functions.base import FunctionSet
from .ids import SubscriptionLevel
from .log import _logger
from .node import LADSNode, promote_to
from .program import ProgramManager
from .state_machine import FunctionalStateMachine
from .util import unique_name_delimiter

if TYPE_CHECKING:
    from .device import Device
    from .functions.base import Function
    from .server import Server
    from .subscription import SubscriptionHandler
    from .variables import BaseVariable, StateVariable


# MARK: FunctionalUnit
class FunctionalUnit(LADSNode):
    """
    Represents a functional unit of a device.

    A functional unit can be a sensor, a control function, a state machine, etc.

    Attributes:
        functional_unit_state (FunctionalStateMachine): The state machine of the functional unit.
        function_set (FunctionSet): The set of functions of the functional unit.
        program_manager (ProgramManager): The program manager of the functional unit.
        unique_name (str): The unique name of the functional unit.
        at_name (str): The at name of the functional unit.
        current_state (StateVariable): The current state of the functional unit.
        functions (list[Function]): The functions of the functional unit.
        events (list[Event]): The events of the functional unit.
    """

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(FunctionalUnit, node, server.FunctionalUnitType, server)

    async def init(self, server: Server):
        """
        Initializes the functional unit.

        Args:
            server: Server - The server.
        """
        await super().init(server)

        self.function_set, self.functional_unit_state, self.program_manager = await asyncio.gather(
            FunctionSet.promote(await self.get_lads_child("FunctionSet"), server),
            FunctionalStateMachine.promote(await self.get_lads_child("FunctionalUnitState"), server),
            ProgramManager.promote(await self.get_lads_child("ProgramManager"), server),
        )
        self.lock = await Lock.promote(await self.get_di_child("Lock"), server) if hasattr(server, "max_inactive_lock_time") else None

    async def finalize_init(self, device: Device):
        await super().finalize_init()
        self.device = device
        if self.function_set is not None:
            await self.function_set.finalize_init(self)
        # prepare subscriptions (data change will be handled by device)
        # self.subscription_handler = SubscriptionHandler()
        # events_handler = await self.subscription_handler.subscribe_events(self.server, self)

    @property
    def subscription_handler(self) -> SubscriptionHandler:
        return self.device.subscription_handler

    @property
    def all_subscribed_variables(self) -> list[BaseVariable]:
        variables = self.subscribed_variables + self.functional_unit_state.variables

        if self.function_set is not None:
            function_variables = self.function_set.all_variables
            # function_variables = remove_none(function_variables)
            # debug- check for none
            for variable in function_variables:
                 if variable is None:
                     _logger.error(f"None variable detected in function {self.unique_name}")
            child_vars = list(filter(lambda variable: variable.subscription_level > SubscriptionLevel.Never, function_variables))
            variables = variables + child_vars
        if self.program_manager is not None:
            variables = variables + self.program_manager.variables
        if self.lock is not None:
            variables = variables + self.lock.variables
        return variables

    @property
    def unique_name(self) -> str:
        return f"{self.device.unique_name}{unique_name_delimiter}{self.display_name}"

    @property
    def at_name(self) -> str:
        device = self.device
        device_name = device.display_name
        server_name = device.server.name
        if len(device.functional_units) > 1:
            return f"{device_name}{unique_name_delimiter}{self.display_name}@{server_name}"
        else:
            return f"{device_name}@{server_name}"


    @property
    def current_state(self) -> StateVariable:
        return self.functional_unit_state.current_state

    @property
    def current_state_str(self) -> StateVariable:
        return self.functional_unit_state.current_state_str

    @property
    def functions(self) -> list[Function]:
        return self.function_set.functions

    @property
    def events(self) ->list[Event]:
        if self.subscription_handler is not None:
            return self.subscription_handler.event_list
        else:
            return []
