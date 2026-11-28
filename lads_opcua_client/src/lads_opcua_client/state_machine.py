"""Methods and state machines of the LADS OPC UA model."""

from __future__ import annotations

import asyncio
import pandas as pd
import re
from asyncua import Node
from typing import Self, TYPE_CHECKING
from .log import _logger
from .node import LADSNode, promote_to
from .util import remove_none
from .variables import StateVariable

if TYPE_CHECKING:
    from .server import Server
    from .variables import BaseVariable


# MARK: Method
class Method(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Method, node, None, server)

    async def get_arguments(self, name: str) -> list:
        try:
            node = await self.get_child(name)
            if (node is not None):
                value = await node.get_value()
                if (type(value) == list):
                    return value
                else:
                    return []
            else:
                return []
        except:
            return []

    async def init(self, server: Server):
        await super().init(server)
        self.input_arguments = await self.get_arguments("InputArguments")
        self.output_arguments = await self.get_arguments("OutputArguments")
        # _logger.debug(f"Method {self.display_name} inp {self.input_arguments} out {self.output_arguments}")


# MARK: StateMachine
class StateMachine(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(StateMachine, node, server.FiniteStateMachineType, server)

    async def init(self, server: Server):
        """
        self.methods: list[Method] = []
        self.methods_dict: dict[str, Method] = {}
        """
        await super().init(server)
        self.current_state = await StateVariable.promote(await self.get_child("CurrentState"), server)
        self.current_state.alternate_display_name = self.display_name
        nodes = await self.get_methods()
        self.methods = await asyncio.gather(*(Method.promote(node, server) for node in nodes))
        self.methods_dict = {method.display_name: method for method in self.methods}

    @property
    def current_state_str(self) -> str:
        result = self.current_state.value_str
        if self.current_state.effective_display_name is not None:
            result = self.current_state.effective_display_name.value_str
        return result

    @property
    def method_names(self) -> list[str]:
        return self.methods_dict.keys()

    def call_method_by_name(self, name: str, *args):
        if name is None:
            return
        try:
            method = self.methods_dict[name]
            if method is not None:
                self.server.call_async_queue.put(self.call_method(method.nodeid, *args))
        except:
            _logger.debug(f"Unknwon method {name}")

    @property
    def variables(self) -> list[BaseVariable]:
        return super().variables + remove_none([self.current_state, self.current_state.effective_display_name])


# MARK: FunctionalStateMachine
program_template_pattern = re.compile(r'^(.*?)(?:\s+\((.*)\))?$')


def extract_program_template_name(name: str) -> str:
    match = program_template_pattern.fullmatch(name)
    if not match:
        return None
    item1, item2 = match.groups()
    return item2 or item1


class FunctionalStateMachine(StateMachine):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(FunctionalStateMachine, node, server.FiniteStateMachineType, server)

    async def init(self, server: Server):
        await super().init(server)
        try:
            self.running_state_machine = await StateMachine.promote(await self.get_lads_child("RunningStateMachine"), server)
        except:
            self.running_state_machine = None

    def buildProperties(self, properties: pd.DataFrame) -> list:
        key_value_list = []
        for index, row in properties.iterrows():
            key = str(row["Key"])
            value =str(row["Value"])
            key_value = self.server.KeyValueType(
                key,
                value,
            )
            key_value_list.append(key_value)
        return key_value_list

    def start_program(self, program_template: str, properties: pd.DataFrame, supervisory_job_id: str, supervisory_task_id: str, samples: pd.DataFrame):
        program_template_name = extract_program_template_name(program_template)
        key_value_list = self.buildProperties(properties)
        sample_info_list = []
        for index, row in samples.iterrows():
            sample_info = self.server.SampleInfoType(
                str(row["ContainerId"]),
                str(row["SampleId"]),
                str(row["Position"]),
                str(row["CustomData"]),
            )
            sample_info_list.append(sample_info)
        self.call_async(self.call_lads_method("StartProgram",
                                              program_template_name,
                                              key_value_list,
                                              supervisory_job_id,
                                              supervisory_task_id,
                                              sample_info_list))

    def start(self, properties: pd.DataFrame):
        # key_value_list = self.buildProperties(properties)
        # self.call_async(self.call_lads_method("Start", key_value_list))
        self.call_async(self.call_lads_method("Start", []))

    def start_with_target_value(self, value: float):
        self.call_async(self.call_lads_method("StartWithTargetValue", value))

    def stop(self):
        self.call_async(self.call_lads_method("Stop"))
