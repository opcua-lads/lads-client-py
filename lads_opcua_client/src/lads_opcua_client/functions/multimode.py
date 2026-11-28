"""Multi-mode control function and its controller parameters."""

from __future__ import annotations

import asyncio
from asyncua import Node
from typing import Self, TYPE_CHECKING
from .controls import BaseControlFunction
from ..node import LADSNode, promote_to
from ..sets import LADSSet
from ..variables import MultiStateDiscrete, get_lads_analog_item

if TYPE_CHECKING:
    from ..server import Server
    from ..variables import AnalogItem, BaseVariable


# MARK: MulitModeControlFunction
class ControllerParameter(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(ControllerParameter, node, server.ControllerParameterType, server)

    def __str__(self):
        return f"  {super().__str__()}\n    {self.current_value}\n    {self.target_value}"

    async def init(self, server: Server):
        await super().init(server)
        self.current_value = await get_lads_analog_item(self, "CurrentValue")
        self.target_value = await get_lads_analog_item(self, "TargetValue")

    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + [self.current_value, self.target_value]


class ControllerParameterSet(LADSSet):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(ControllerParameterSet, node, server.ControllerParameterSetType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.controller_parameters: list[ControllerParameter] = await asyncio.gather(*(ControllerParameter.promote(child, server) for child in self.children))
        self.controller_parameters.sort(key = lambda node: node.display_name)

    def controller_parameter(self, mode: str) -> ControllerParameter:
        try:
            index = self.controller_parameters.index(key = lambda node: node.display_name)
            return self.controller_parameters[index]
        except:
            return None


class MulitModeControlFunction(BaseControlFunction):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(MulitModeControlFunction, node, server.MultiModeControlFunctionType, server)

    def __str__(self):
        s = ""
        for controller_parameter in self.controller_parameters:
            s = s + f"\n  {controller_parameter.__str__()}"
        return f"{super().__str__()}{s}"

    async def init(self, server: Server):
        await super().init(server)
        self.current_mode = await MultiStateDiscrete.promote(await self.get_lads_child("CurrentMode"), server)
        self.controller_mode_set = await ControllerParameterSet.promote(await self.get_lads_child("ControllerModeSet"), server)

    @property
    def controller_parameters(self) -> list[ControllerParameter]:
        return self.controller_mode_set.controller_parameters

    @property
    def modes(self) -> list[str]:
        return list(map(lambda controller_parameter: controller_parameter.display_name, self.controller_parameters))

    @property
    def current_controller_parameter(self) -> ControllerParameter:
        mode = self.current_mode.value_str
        return self.controller_mode_set.controller_parameter(mode)

    # expose current values to achieve compatibility to AnalogControlFunction
    @property
    def target_value(self) -> AnalogItem:
        controller_parameter = self.current_controller_parameter
        if controller_parameter is None:
            return None
        else:
            return controller_parameter.target_value

    @property
    def current_value(self) -> AnalogItem:
        controller_parameter = self.current_controller_parameter
        if controller_parameter is None:
            return None
        else:
            return controller_parameter.current_value

    @property
    def variables(self) ->list[BaseVariable]:
        variables: list[BaseVariable] = []
        for controller_parameter in self.controller_parameters:
            variables.append(controller_parameter.target_value)
            variables.append(controller_parameter.current_value)
        return super().variables + variables
