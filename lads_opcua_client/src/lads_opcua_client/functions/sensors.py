"""Sensor functions."""

from __future__ import annotations

from asyncua import Node
from typing import Self, TYPE_CHECKING
from ..alarm_monitor import AlarmMonitor
from .base import Function
from ..node import promote_to
from ..variables import get_lads_analog_item, get_lads_multi_state_discrete, get_lads_two_state_discrete

if TYPE_CHECKING:
    from ..server import Server
    from ..variables import BaseVariable


# MARK: BaseSensorFunction
class BaseSensorFunction(Function):

    def __str__(self):
        return f"{super().__str__()}\n  {self.sensor_value}"

    async def init(self, server: Server):
        self.sensor_value = None
        await super().init(server)

    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + [self.sensor_value]


# MARK: AnalogScalarSensorFunction
class AnalogScalarSensorFunction(BaseSensorFunction):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(AnalogScalarSensorFunction, node, server.AnalogScalarSensorFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.sensor_value = await get_lads_analog_item(self, "SensorValue")
        self.alarm_monitor = await AlarmMonitor.promote(await self.get_lads_child("AlarmMonitor"), server)

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


# MARK: AnalogScalarSensorFunctionWithCompensation
class AnalogScalarSensorFunctionWithCompensation(AnalogScalarSensorFunction):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(AnalogScalarSensorFunctionWithCompensation, node, server.AnalogScalarSensorFunctionWithCompensationType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.compensation_value = await get_lads_analog_item(self, "CompensationValue")

    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + [self.compensation_value]


class AnalogArraySensorFunction(AnalogScalarSensorFunction):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(AnalogArraySensorFunction, node, server.AnalogArraySensorFunctionType, server)


# MARK: TwoStateDiscreteSensorFunction
class TwoStateDiscreteSensorFunction(BaseSensorFunction):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(TwoStateDiscreteSensorFunction, node, server.TwoStateDiscreteSensorFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.sensor_value = await get_lads_two_state_discrete(self, "SensorValue")


# MARK: MultiStateDiscreteSensorFunction
class MultiStateDiscreteSensorFunction(BaseSensorFunction):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(MultiStateDiscreteSensorFunction, node, server.MultiStateDiscreteSensorFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.sensor_value = await get_lads_multi_state_discrete(self, "SensorValue")
