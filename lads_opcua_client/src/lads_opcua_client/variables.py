"""Variables of the LADS OPC UA model (analog items, discrete variables, counters, ...)."""

from __future__ import annotations

import asyncio
import pandas as pd
from asyncua import Node, ua
from asyncua.common.subscription import DataChangeNotif
from enum import IntEnum
from math import log10, trunc
from typing import Any, Self, TYPE_CHECKING
from .ids import SubscriptionLevel
from .log import _logger
from .node import LADSNode, promote_to
from .util import duration_to_str, variant_value_to_str

if TYPE_CHECKING:
    from .server import Server


# MARK: BaseVariable
class BaseVariable(LADSNode):
    """"
    Represents a variable in the LADS OPC UA model.

    Attributes:
        alternate_display_name: str - an alternate name for the variable
        subscription_level: SubscriptionLevel - the subscription level of the variable
        data_value: ua.DataValue - the current value of the variable
        data_type: ua.VariantType - the data type of the variable
        access_level: Set[ua.AccessLevel] - the access level of the variable
        history: pd.DataFrame - the history of the variable
        display_name: str - the display name of the variable
        has_write_access: bool - True if the variable has write access
        value: Any - the value of the variable
        value_str: str - the value of the variable as a string
    """

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(BaseVariable, node, server.BaseVariableType, server)

    def __str__(self):
        return f"{super().__str__()} = {self.value}"

    async def init(self, server: Server):
        self.alternate_display_name: str = None
        self.history: pd.DataFrame = None
        self.subscription_level = SubscriptionLevel.Never

        await super().init(server)
        (self.data_value, self.data_type, self.variant_type, self.access_level, historizing) = await asyncio.gather(
            self.read_data_value(raise_on_bad_status=False),
            self.read_data_type(),
            self.read_data_type_as_variant_type(),
            self.get_access_level(),
            self.read_attribute(ua.AttributeIds.Historizing)
        )
        if (historizing.Value.Value):
            self.subscription_level = SubscriptionLevel.Permanent
            self.history = pd.DataFrame({f"{self.display_name}": [self.value]}, index = [pd.to_datetime(self.data_value.SourceTimestamp)])

    @property
    def default_decimals(self) -> int:
        if (self.variant_type == ua.VariantType.Double) or (self.variant_type == ua.VariantType.Float):
            return 1
        else:
            return 0

    @property
    def display_name(self) -> str:
        if self.alternate_display_name is not None:
            return self.alternate_display_name
        else:
            return super().display_name

    def set_value(self, value: Any) -> ua.StatusCode:
        if value is None:
            return ua.StatusCodes.BadNoValue
        if self.has_write_access:
            self.server.call_async_queue.put(self.write_value(value, self.variant_type))
            return ua.StatusCodes.Uncertain
        else:
            return ua.StatusCodes.BadNotWritable

    async def set_value_async(self, value: Any) -> ua.StatusCode:
        if value is None:
            return ua.StatusCodes.BadNoValue
        if self.has_write_access:
            result = ua.StatusCodes.Good
            try:
                await self.write_value(value, self.variant_type)
            except:
                result = ua.StatusCodes.BadInvalidArgument
            return result
        else:
            return ua.StatusCodes.BadNotWritable

    async def update_value(self):
        self.data_value = await self.read_data_value(raise_on_bad_status=False)

    @property
    def has_write_access(self) -> bool:
        return ua.AccessLevel.CurrentWrite in self.access_level

    @property
    def value(self) -> Any:
        if self.data_value:
            return self.data_value.Value.Value
        else:
            return None

    @property
    def value_str(self) -> str:
        if self.data_value:
            # check for duration datatype
            if self.data_type == ua.NodeId(290):
                return duration_to_str(self.data_value.Value.Value, 0)
            else:
                return variant_value_to_str(self.data_value.Value)
        else:
            return ""

    def data_change_notification(self, data: DataChangeNotif):
        self.data_value = data.monitored_item.Value
        if self.history is not None:
            try:
                self.history.loc[pd.to_datetime(self.data_value.SourceTimestamp)] = self.value
            except:
                pass
            if len(self.history.index) > 600:
                self.history = self.history.tail(-1)


# MARK: SubscribedVariable
class SubscribedVariable(BaseVariable):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(SubscribedVariable, node, server.BaseVariableType, server)

    async def init(self, server: Server):
        await super().init(server)
        if self.history is None:
            self.subscription_level = SubscriptionLevel.Temporary


# MARK: NodeVersionVariable
class NodeVersionVariable(SubscribedVariable):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(NodeVersionVariable, node, server.BaseVariableType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.subscription_level = SubscriptionLevel.Permanent
        self.set: LADSNode = None

    def data_change_notification(self, data: DataChangeNotif):
        super().data_change_notification(data)
        if self.set is None: return
        try:
            self.set.node_version_changed()
        except(Exception):
            _logger.debug(f"Set {self.set.display_name} misses node_version_changed() implementation")


# MARK: StateVariable
class StateVariable(SubscribedVariable):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        variable: StateVariable = await promote_to(StateVariable, node, server.BaseVariableType, server)
        variable.subscription_level = SubscriptionLevel.Permanent
        return await promote_to(StateVariable, node, server.BaseVariableType, server)

    async def init(self, server: Server):
        await super().init(server)

        variables = await self.get_children(nodeclassmask=ua.NodeClass.Variable)
        for variable in variables:
            browse_name = await variable.read_browse_name()
            name = browse_name.Name
            if name == "EffectiveDisplayName":
                self.effective_display_name = await BaseVariable.promote(variable, server)
            elif name == "Id":
                self.id = await BaseVariable.promote(variable, server)

    @property
    def value_str(self) -> str:
        s =  super().value_str
        l = s.split(":")
        return s if len(l) < 2 else l[1]


# MARK: AnalogItem
class AnalogItem(SubscribedVariable):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(AnalogItem, node, server.AnalogItemType, server)

    def __str__(self):
        return f"{super().__str__()} [{self.eu}]"

    async def init(self, server: Server):
        await super().init(server)
        self.engineering_units: ua.EUInformation = None
        self.eu_range: ua.Range = None
        self.value_precision: int = None
        self._default_decimals: int = None
        try:
            engineering_units = await self.get_child("EngineeringUnits")
            self.engineering_units: ua.EUInformation = await engineering_units.get_value()
        except:
            self.engineering_units = None
        try:
            eu_range = await self.get_child("EURange")
            self.eu_range: ua.Range = await eu_range.get_value()
        except:
            self.eu_range = None
        try:
            value_precision = await self.get_child("ValuePrecision")
            self.value_precision = int(await value_precision.get_value())
        except:
            self.value_precision = None

    @property
    def eu(self) -> str:
        if self.engineering_units is not None:
            if isinstance(self.engineering_units, ua.EUInformation):
                result = self.engineering_units.DisplayName.Text
                if result is None:
                    return ""
                return "%" if " or pct" in result else result
        return ""

    @property
    def default_decimals(self) -> int:
        if self._default_decimals is None:
            self._default_decimals = self.decimals()
            _logger.debug(f"{self.display_name} default_decimals = {self._default_decimals}")
        return self._default_decimals

    def decimals(self, resolution = 1000, default = 1) -> int:
        if self.value_precision is not None:
            return self.value_precision
        if self.eu_range is None:
            return default
        try:
            range = abs(self.eu_range.High - self.eu_range.Low)
        except Exception:
            # _logger.debug(f"Unable to determine decimals of {self.display_name}-{self.nodeid}: EURange has no value")
            return default
        if range >= resolution:
            return default
        try:
            l = log10(resolution) - log10(range) + default
            d = trunc(l)
            return d
        except Exception:
            _logger.debug(f"Unable to determine decimals of {self.display_name}-{self.nodeid}: resolution={resolution}, range={range}")
            return default


# MARK: Enumeration
class Enumeration(SubscribedVariable):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Enumeration, node, server.BaseVariableType, server)

    def __str__(self):
        return f"{super().__str__()}\n  EnumStrings: {self.enum_strings}"

    async def init(self, server: Server):
        await super().init(server)
        data_type_node_id = await self.read_data_type()
        data_type_node = Node(self.session, data_type_node_id)
        name = await data_type_node.read_browse_name()
        try:
            enum: IntEnum = ua.__dict__[name.Name]
            for item in enum:
                self.enum_strings[item.value] = item.name
        except:
            self.enum_strings = {}

    @property
    def value_str(self) -> str:
        try:
            return self.enum_strings[int(self.value)]
        except:
            return "unknown"


# MARK: DiscreteVariable
class DiscreteVariable(SubscribedVariable):

    @property
    def values(self) -> list[ua.LocalizedText]:
        return []

    @property
    def values_as_str(self) -> list[str]:
        return list(map(lambda value: value.Text, self.values))

    def set_value_from_str(self, value_str: str):
        if value_str is None:
            return
        try:
            values = self.values_as_str
            value = values.index(value_str)
            if isinstance(self, MultiStateDiscrete):
                self.set_value(value)
            else:
                self.set_value(value == 0)
        except Exception as error:
            print(error)


# MARK: TwoStateDiscrete
class TwoStateDiscrete(DiscreteVariable):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(TwoStateDiscrete, node, server.TwoStateDiscreteType, server)

    def __str__(self):
        return f"{super().__str__()}\n  TrueState: {self.true_state.value_str}\n  FalseState: {self.false_state.value_str}"

    async def init(self, server: Server):
        """
        self.true_state: BaseVariable = None
        self.false_state: BaseVariable = None
        """
        await super().init(server)
        self.true_state = await BaseVariable.promote(await self.get_child("TrueState"), server)
        self.false_state = await BaseVariable.promote(await self.get_child("FalseState"), server)

    @property
    def value_str(self) -> str:
        if bool(self.value):
            return self.true_state.value_str
        else:
            return self.false_state.value_str

    @property
    def values(self) -> list[ua.LocalizedText]:
        return [self.true_state.data_value.Value.Value, self.false_state.data_value.Value.Value]


# MARK: MultiStateDiscrete
class MultiStateDiscrete(DiscreteVariable):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(MultiStateDiscrete, node, server.MultiStateDiscreteType, server)

    def __str__(self):
        value: list[str] = self.enum_strings.data_value.Value.Value
        s = ",".join(value)
        return f"{super().__str__()}\n  [{s}]"

    async def init(self, server: Server):
        await super().init(server)
        self.enum_strings = await BaseVariable.promote(await self.get_child("EnumStrings"), server)
        assert(self.enum_strings.data_value.Value.is_array)

    @property
    def value_str(self) -> str:
        s = self.values
        i = int(self.value)
        if i in range(len(s)):
            return s[i].Text
        else:
            "unknown"

    @property
    def values(self) -> list[ua.LocalizedText]:
        return self.enum_strings.data_value.Value.Value


# MARK: LifetimeCounter
class LifetimeCounter(AnalogItem):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(LifetimeCounter, node, server.LifetimeVariableType, server)

    def __str__(self):
        return f"{super().__str__()}\n  {self.limit_value}\n  {self.start_value}"

    async def init(self, server: Server):
        """
        self.limit_value: BaseVariable = None
        self.start_value: BaseVariable = None
        self.warning_values: BaseVariable = None
        """
        await super().init(server)
        self.limit_value, self.start_value, self.warning_values = await asyncio.gather(
            self.get_di_variable("LimitValue"),
            self.get_di_variable("StartValue"),
            self.get_di_variable("WarningValues"),
        )


async def get_lads_analog_item(parent: LADSNode, name: str) -> AnalogItem:
    node = await parent.get_lads_child(name)
    return await AnalogItem.promote(node, parent.server)


async def get_lads_two_state_discrete(parent: LADSNode, name: str) -> TwoStateDiscrete:
    node = await parent.get_lads_child(name)
    return await TwoStateDiscrete.promote(node, parent.server)


async def get_lads_multi_state_discrete(parent: LADSNode, name: str) -> MultiStateDiscrete:
    node = await parent.get_lads_child(name)
    return await MultiStateDiscrete.promote(node, parent.server)


async def get_di_variable(parent: LADSNode, name: str) -> BaseVariable:
    return await BaseVariable.promote(await parent.get_di_child(name), parent.server)


async def get_properties_and_variables(node: LADSNode) -> list[BaseVariable]:
    (variables, properties) = await asyncio.gather(node.get_variables(), node.get_properties())
    variables.extend(properties)
    return await asyncio.gather(*(BaseVariable.promote(variable, node.server) for variable in variables))
