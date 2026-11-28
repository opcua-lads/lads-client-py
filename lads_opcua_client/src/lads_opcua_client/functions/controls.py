"""Control functions."""

from __future__ import annotations

import asyncio
from asyncua import Node, ua
from typing import Any, Self, TYPE_CHECKING
from ..alarm_monitor import AlarmMonitor
from .base import BaseStateMachineFunction
from ..node import promote_to
from ..state_machine import FunctionalStateMachine
from ..util import remove_none
from ..variables import AnalogItem, SubscribedVariable, get_lads_analog_item, get_lads_multi_state_discrete, get_lads_two_state_discrete

if TYPE_CHECKING:
    from ..server import Server
    from ..state_machine import StateMachine
    from ..variables import BaseVariable


# MARK: BaseControlFunction
class BaseControlFunction(BaseStateMachineFunction):

    async def init(self, server: Server):
        await super().init(server)
        self.control_function_state = await FunctionalStateMachine.promote(await self.get_lads_child("ControlFunctionState"), server)

    @property
    def state_machine(self) -> StateMachine:
        return self.control_function_state


# MARK: StartStopControlFunction
class StartStopControlFunction(BaseControlFunction):#
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(StartStopControlFunction, node, server.StartStopControlFunctionType, server)


# MARK: BaseAnalogDiscreteControlFunction
class BaseAnalogDiscreteControlFunction(BaseControlFunction):
    def __str__(self):
        return f"{super().__str__()}\n  {self.current_value}\n  {self.target_value}"

    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + remove_none([self.current_value, self.target_value])


# MARK: AnalogControlFunction
class AnalogControlFunction(BaseAnalogDiscreteControlFunction):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(AnalogControlFunction, node, server.AnalogControlFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.current_value, self.target_value, self.alarm_monitor =  await asyncio.gather(
            get_lads_analog_item(self, "CurrentValue"),
            get_lads_analog_item(self, "TargetValue"),
            AlarmMonitor.promote(await self.get_lads_child("AlarmMonitor"), server)
        )

    @property
    def has_alarm_monitor(self) -> bool:
        return self.alarm_monitor is not None

    @property
    def alarm_active(self) -> bool:
        return False if self.alarm_monitor is None else self.alarm_monitor.alarm_active

    @property
    def alarm_limit(self) -> str:
        return "" if self.alarm_monitor is None else self.alarm_monitor.alarm_limit

    @property
    def variables(self) ->list[BaseVariable]:
        variables = super().variables
        if self.alarm_monitor is not None:
            variables = variables + self.alarm_monitor.variables
        return variables


# MARK: AnalogControlFunctionWithTotalizer
class AnalogControlFunctionWithTotalizer(AnalogControlFunction):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(AnalogControlFunctionWithTotalizer, node, server.AnalogControlFunctionWithTotalizerType, server)

    def __str__(self):
        return f"{super().__str__()}\n  {self.totalized_value}"

    async def init(self, server: Server):
        await super().init(server)
        self.totalized_value = await get_lads_analog_item(self, "TotalizedValue")

    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + [self.totalized_value]


# MARK: AnalogControlFunctionWithComposedTargetValue
class AnalogControlFunctionWithComposedTargetValue(AnalogControlFunction):
    """
    AnalogControlFunction whose target value is composed of partial values (LADS AnalogControlFunctionWithComposedTargetValueType):
    the mandatory TargetValueSet holds the partial target values, e.g. Flow and PercentageSolventB of an HPLC pump.
    TargetValue and CurrentValue are those of AnalogControlFunction.

    Attributes:
        target_value_set: list[BaseVariable] - the partial target values (AnalogItem if the variable is one), sorted by name.
    """
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(AnalogControlFunctionWithComposedTargetValue, node, server.AnalogControlFunctionWithComposedTargetValueType, server)

    def __str__(self):
        parts = "".join(f"\n  {variable}" for variable in self.target_value_set)
        return f"{super().__str__()}{parts}"

    async def init(self, server: Server):
        await super().init(server)
        self.target_value_set: list[BaseVariable] = []
        node = await self.get_lads_child("TargetValueSet")
        if node is None:
            return
        children = await node.get_children(refs = ua.ObjectIds.HierarchicalReferences, nodeclassmask = ua.NodeClass.Variable)

        async def promote_partial(child: Node) -> BaseVariable | None:
            if "NodeVersion" in (await child.read_browse_name()).Name:
                return None
            try:
                return await AnalogItem.promote(child, server)
            except AssertionError:
                return await SubscribedVariable.promote(child, server)

        self.target_value_set = remove_none(await asyncio.gather(*(promote_partial(child) for child in children)))
        self.target_value_set.sort(key = lambda variable: variable.display_name)

    def partial_target_value(self, name: str) -> BaseVariable | None:
        """The partial target value with this display name, None if there is none."""
        return next((variable for variable in self.target_value_set if variable.display_name == name), None)

    @property
    def variables(self) -> list[BaseVariable]:
        return super().variables + self.target_value_set


# MARK: AnalogControlFunctionWithRelativeTargetValue
class AnalogControlFunctionWithRelativeTargetValue(AnalogControlFunction):
    """
    AnalogControlFunction whose target value is typically changed by relative increments or decrements
    (LADS AnalogControlFunctionWithRelativeTargetValueType), e.g. a dispenser that aspirates or dispenses a volume.

    Attributes:
        increase_rate: AnalogItem | None - rate by which the value is increased (e.g. aspirating rate), optional
        decrease_rate: AnalogItem | None - rate by which the value is decreased (e.g. dispensing rate), optional
        has_modify_target_value_by: bool - the optional method ModifyTargetValueBy exists
    """
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(AnalogControlFunctionWithRelativeTargetValue, node, server.AnalogControlFunctionWithRelativeTargetValueType, server)

    def __str__(self):
        parts = "".join(f"\n  {variable}" for variable in self.rates)
        return f"{super().__str__()}{parts}"

    async def init(self, server: Server):
        await super().init(server)
        increase, decrease, method = await asyncio.gather(
            self.get_lads_child("IncreaseRate"), self.get_lads_child("DecreaseRate"), self.get_lads_child("ModifyTargetValueBy"))
        self.increase_rate: AnalogItem | None = await AnalogItem.promote(increase, server) if increase is not None else None
        self.decrease_rate: AnalogItem | None = await AnalogItem.promote(decrease, server) if decrease is not None else None
        self.has_modify_target_value_by: bool = method is not None

    @property
    def rates(self) -> list[AnalogItem]:
        return remove_none([self.increase_rate, self.decrease_rate])

    def modify_target_value_by(self, value: float):
        """Calls ModifyTargetValueBy(value): the target value changes by value (positive or negative)."""
        if self.has_modify_target_value_by:
            self.call_async(self.call_lads_method("ModifyTargetValueBy", ua.Variant(float(value), ua.VariantType.Double)))

    async def modify_target_value_by_async(self, value: float) -> Any:
        return await self.call_lads_method("ModifyTargetValueBy", ua.Variant(float(value), ua.VariantType.Double))

    @property
    def variables(self) -> list[BaseVariable]:
        return super().variables + self.rates


# MARK: TimerControlFunction
class TimerControlFunction(AnalogControlFunction):
    def __str__(self):
        return f"{super().__str__()}\n  {self.difference_value}"

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(TimerControlFunction, node, server.TimerControlFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.difference_value = await get_lads_analog_item(self, "DifferenceValue")

    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + remove_none([self.difference_value])


# MARK: DiscreteControlFunction
class DiscreteControlFunction(BaseAnalogDiscreteControlFunction):
    pass


# MARK: TwoStateDiscreteControlFunction
class TwoStateDiscreteControlFunction(DiscreteControlFunction):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(TwoStateDiscreteControlFunction, node, server.TwoStateDiscreteControlFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.current_value = await get_lads_two_state_discrete(self, "CurrentValue")
        self.target_value = await get_lads_two_state_discrete(self, "TargetValue")


# MARK: MultiStateDiscreteControlFunction
class MultiStateDiscreteControlFunction(DiscreteControlFunction):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(MultiStateDiscreteControlFunction, node, server.MultiStateDiscreteControlFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.current_value = await get_lads_multi_state_discrete(self, "CurrentValue")
        self.target_value = await get_lads_multi_state_discrete(self, "TargetValue")
