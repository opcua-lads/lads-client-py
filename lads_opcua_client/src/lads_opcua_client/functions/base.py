"""Function and FunctionSet: base classes of all functions of a functional unit."""

from __future__ import annotations

import asyncio
from asyncua import Node
from typing import Self, TYPE_CHECKING
from ..log import _logger
from ..node import LADSNode, promote_to
from ..sets import LADSSet
from ..util import browse_types, remove_none, unique_name_delimiter
from ..variables import BaseVariable

if TYPE_CHECKING:
    from ..server import Server
    from ..state_machine import StateMachine


# MARK: Function
class Function(LADSNode):
    """
    Represents a function in the LADS OPC UA model.

    Attributes:
        functional_parent: LADSNode - Parent of the function.
        unique_name: str - Unique name of the function.
        functions: list[Function] - List of functions.
        variables: list[BaseVariable] - List of variables.
        all_variables: list[BaseVariable] - List of all variables.
    """

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Function, node, server.FunctionType, server)

    async def init(self, server: Server):
        """
        Initializes the function.

        Args:
            server: Server - The server.
        """

        self.functional_parent: LADSNode = None
        await super().init(server)

        node = await self.get_lads_child("IsEnabled")
        self.is_enabled = await BaseVariable.promote(node, server)
        self.function_set: FunctionSet = await self.get_lads_child("FunctionSet")
        if self.function_set is not None:
            self.function_set = await FunctionSet.promote(self.function_set, server)

    async def finalize_init(self, functional_parent: LADSNode):
        await super().finalize_init()
        self.functional_parent = functional_parent
        if self.function_set is not None:
            await self.function_set.finalize_init(functional_parent)

    @property
    def unique_name(self) -> str:
        parent_name = "unknown" if self.functional_parent is None else self.functional_parent.unique_name
        return f"{parent_name}{unique_name_delimiter}{self.display_name}"

    @property
    def functions(self) -> list[Function]:
        return self.function_set.functions

    @property
    def variables(self) ->list[BaseVariable]:
        if (self.is_enabled is None):
            return []
        else:
            return [self.is_enabled]

    @property
    def all_variables(self) -> list[BaseVariable]:
        nodes = self.variables
        if self.function_set:
            nodes = nodes + self.function_set.variables
            for function in self.function_set.functions:
                variables = function.all_variables
                nodes = nodes + variables
        return nodes


# MARK: FunctionSet
class FunctionSet(LADSSet):
    """
    Represents a function set in the LADS OPC UA model.

    Attributes:
        functions: list[Function] - List of functions.
        all_variables: list[BaseVariable] - List of all variables.

    Methods:
        promote: Promotes a node to a function set.
        promote_child: Promotes a child node to a function.
    """

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(FunctionSet, node, server.FunctionSetType, server)

    async def init(self, server: Server):
        """
        Initializes the function set.

        Args:
            server: Server - The server.
        """

        await super().init(server)
        self.functions: list[Function] = remove_none(await asyncio.gather(*(self.promote_child(child) for child in self.children)))
        self.functions.sort(key = lambda function: function.display_name)

    async def finalize_init(self, functional_parent: LADSNode):
        await asyncio.gather(*(function.finalize_init(functional_parent) for function in self.functions))

    async def promote_child(self, child: Node) -> Function:
        from .controls import (  # lazy: avoids a circular import
            AnalogControlFunction,
            AnalogControlFunctionWithComposedTargetValue,
            AnalogControlFunctionWithRelativeTargetValue,
            AnalogControlFunctionWithTotalizer,
            MultiStateDiscreteControlFunction,
            StartStopControlFunction,
            TimerControlFunction,
            TwoStateDiscreteControlFunction,
        )
        from .cover import CoverFunction  # lazy: avoids a circular import
        from .multimode import MulitModeControlFunction  # lazy: avoids a circular import
        from .sensors import (  # lazy: avoids a circular import
            AnalogArraySensorFunction,
            AnalogScalarSensorFunction,
            AnalogScalarSensorFunctionWithCompensation,
            MultiStateDiscreteSensorFunction,
            TwoStateDiscreteSensorFunction,
        )
        server = self.server
        types = await browse_types(server, child)
        try:
            function: Function = None
            if server.AnalogControlFunctionWithTotalizerType in types:
                function = await AnalogControlFunctionWithTotalizer.promote(child, server)
            elif server.AnalogControlFunctionWithComposedTargetValueType in types:
                function = await AnalogControlFunctionWithComposedTargetValue.promote(child, server)
            elif server.AnalogControlFunctionWithRelativeTargetValueType in types:
                function = await AnalogControlFunctionWithRelativeTargetValue.promote(child, server)
            elif server.AnalogControlFunctionType in types:
                function = await AnalogControlFunction.promote(child, server)
            elif server.TimerControlFunctionType in types:
                function = await TimerControlFunction.promote(child, server)
            elif server.AnalogScalarSensorFunctionWithCompensationType in types:
                function = await AnalogScalarSensorFunctionWithCompensation.promote(child, server)
            elif server.AnalogScalarSensorFunctionType in types:
                function = await AnalogScalarSensorFunction.promote(child, server)
            elif server.AnalogArraySensorFunctionType in types:
                function = await AnalogArraySensorFunction.promote(child, server)
            elif server.TwoStateDiscreteSensorFunctionType in types:
                function = await TwoStateDiscreteSensorFunction.promote(child, server)
            elif server.MultiStateDiscreteSensorFunctionType in types:
                function = await MultiStateDiscreteSensorFunction.promote(child, server)
            elif server.CoverFunctionType in types:
                function = await CoverFunction.promote(child, server)
            elif server.StartStopControlFunctionType in types:
                function = await StartStopControlFunction.promote(child, server)
            elif server.TwoStateDiscreteControlFunctionType in types:
                function = await TwoStateDiscreteControlFunction.promote(child, server)
            elif server.MultiStateDiscreteControlFunctionType in types:
                function = await MultiStateDiscreteControlFunction.promote(child, server)
            elif server.MultiModeControlFunctionType in types:
                function = await MulitModeControlFunction.promote(child, server)
            else:
                # _logger.debug("Unknown function ", child)
                function = await Function.promote(child, server)
        except Exception as error:
            _logger.error(error)
        return function

    @property
    def all_variables(self) -> list[BaseVariable]:
        variables = self.variables
        for function in self.functions:
            variables = variables + function.all_variables
        return variables


# MARK: BaseStateMachineFunction
class BaseStateMachineFunction(Function):
    def __str__(self):
        return f"{super().__str__()}\n  {self.current_state}"

    @property
    def variables(self) ->list[Node]:
        return super().variables + [self.state_machine.current_state]

    @property
    def current_state(self) -> BaseVariable:
        return self.state_machine.current_state

    @property
    def state_machine(self) -> StateMachine:
        _logger.error(f"Abstract function state_machine()")
