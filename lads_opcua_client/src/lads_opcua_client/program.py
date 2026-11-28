"""Programs, program templates and results of a functional unit."""

from __future__ import annotations

from asyncua import Node
from asyncua.client.ua_file import UaFile
from typing import Any, Self, TYPE_CHECKING
from .ids import SubscriptionLevel
from .log import _logger
from .node import LADSNode, promote_to
from .sets import LADSSet, VariableSet
from .subscription import SubscriptionHandler
from .util import remove_none
from .variables import get_properties_and_variables

if TYPE_CHECKING:
    from .server import Server
    from .variables import BaseVariable


# MARK: ProgramTemplate
class ProgramTemplate(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(ProgramTemplate, node, server.ProgramTemplateType, server)

    async def init(self, server: Server):
        await super().init(server)
        self._variables = await get_properties_and_variables(self)
        self._variables.sort(key = lambda variable: variable.display_name)

    @property
    def unique_name(self) -> str:
        display_name = self.display_name
        browse_name = self.browse_name.Name
        return display_name if display_name == browse_name else f"{display_name} ({browse_name})"

    @property
    def variables(self) ->list[BaseVariable]:
        return self._variables


# MARK: ResultFile
class ResultFile(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(ResultFile, node, server.ResultFileType, server)

    async def init(self, server: Server):
        self.mime_type: BaseVariable = None
        self.name: BaseVariable = None
        self.file: LADSNode = None
        self.data: Any = None
        await super().init(server)

        self.mime_type = await self.get_lads_variable("MimeType")
        self.name = await self.get_lads_variable("Name")
        self.file = await self.get_lads_child("File")

    async def download(self):
        try:
            _logger.debug(f"{self.display_name} start downloading file data ..")
            async with UaFile(self.file, "r") as ua_file:
                self.data = await ua_file.read()
                _logger.debug(f"{self.display_name} finished downloading file data ..")
        except:
            _logger.error(f"{self.display_name} failed reading file")

    def has_data(self) -> bool:
        return self.data is not None

    def fetch_data(self):
        _logger.debug(f"{self.display_name} fetching file data ..")
        self.call_async(self.download())

    @property
    def variables(self) ->list[BaseVariable]:
        return [self.name, self.mime_type]


# MARK: Result
class Result(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Result, node, server.ResultType, server)

    async def init(self, server: Server):
        self.file_set: LADSSet = None
        self.variable_set: VariableSet = None
        self.subscription_handler: SubscriptionHandler = None
        await super().init(server)

        self._variables = remove_none(await get_properties_and_variables(self))
        self._variables.sort(key = lambda variable: variable.display_name)
        await self.update_sets()
        self.subscription_handler = SubscriptionHandler()
        node_version_vars = remove_none([self.file_set.node_version, self.variable_set.node_version])
        data_change_handlers = await self.subscription_handler.subscribe_data_change(self.server, node_version_vars)

    def update(self):
        self.call_async(self.update_async())

    async def update_async(self):
        await self.update_variables_async()
        await self.update_sets()

    async def update_sets(self):
        self.file_set = await LADSSet.promote(await self.get_lads_child("FileSet"), self.server)
        await self.file_set.promote_children(ResultFile, self.server.ResultFileType, self.server.ResultFileSetType)
        self.variable_set = await VariableSet.promote(await self.get_lads_child("VariableSet"), self.server)

    @property
    def variables(self) -> list[BaseVariable]:
        return self._variables

    @property
    def result_files(self) -> list[ResultFile]:
        return self.file_set.children


# MARK: ActiveProgram
class ActiveProgram(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(ActiveProgram, node, server.ActiveProgramType, server)

    def find_variable(self, name: str) -> BaseVariable:
        if self._variables is None:
            return None
        match = list(filter(lambda variable: name in variable.browse_name.Name , self._variables))
        return None if len(match) == 0 else match[0]

    async def init(self, server: Server):
        self.current_program_template: BaseVariable
        self.current_runtime: BaseVariable
        self.current_pause_time: BaseVariable
        self.current_step_name: BaseVariable
        self.curent_step_number: BaseVariable
        self.current_step_runtime: BaseVariable
        self.estimated_runtime: BaseVariable
        self.estimated_step_numbers: BaseVariable
        self.estimated_step_runtime: BaseVariable
        self.device_program_run_id: BaseVariable
        self._variables: list[BaseVariable]

        await super().init(server)
        self._variables = await get_properties_and_variables(self)
        self._variables.sort(key = lambda variable: variable.display_name)
        for variable in self._variables:
            variable.subscription_level = SubscriptionLevel.Temporary
        self.current_program_template = self.find_variable("CurrentProgramTemplate")
        self.current_runtime = self.find_variable("CurrentRuntime")
        self.current_pause_time = self.find_variable("CurrentPauseTime")
        self.current_step_name = self.find_variable("CurrentStepName")
        self.current_step_number = self.find_variable("CurrentStepNumber")
        self.current_step_runtime = self.find_variable("CurrentStepRuntime")
        self.estimated_runtime = self.find_variable("EstimatedRuntime")
        self.estimated_step_numbers = self.find_variable("EstimatedStepNumbers")
        self.estimated_step_runtime = self.find_variable("EstimatedStepRuntime")
        self.device_program_run_id = self.find_variable("DeviceProgramRunId")

    @property
    def variables(self) ->list[BaseVariable]:
        return self._variables

    @property
    def has_runtime_progress(self) -> bool:
        return not (self.current_runtime is None or self.estimated_runtime is None)

    @property
    def current_runtime_progress(self) -> float:
        try:
            progress = max(min(self.current_runtime.value / self.estimated_runtime.value, 1), 0)
            return progress
        except:
            return 0.0

    @property
    def has_step_runtime_progress(self) -> bool:
        return not (self.current_step_runtime is None or self.estimated_step_runtime is None)

    @property
    def current_step_runtime_progress(self) -> float:
        try:
            progress = max(min(self.current_step_runtime.value / self.estimated_step_runtime.value, 1), 0)
            return progress
        except:
            return 0.0

    @property
    def has_step_number_progress(self) -> bool:
        return not (self.current_step_number is None or self.estimated_step_numbers is None)

    @property
    def current_step_number_progress(self) -> float:
        try:
            progress = max(min(self.current_step_number.value / self.estimated_step_numbers.value, 1), 0)
            return progress
        except:
            return 0.0


# MARK: ProgramManager
class ProgramManager(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(ProgramManager, node, server.ProgramManagerType, server)

    async def init(self, server: Server):
        self.program_template_set: LADSSet = None
        self.result_set: LADSSet = None
        self.active_program: ActiveProgram = None
        await super().init(server)

        self.program_template_set = await LADSSet.promote(await self.get_lads_child("ProgramTemplateSet"), server)
        self.result_set = await LADSSet.promote(await self.get_lads_child("ResultSet"), server)
        await self.program_template_set.promote_children(ProgramTemplate, server.ProgramTemplateType, server.ProgramTemplateSetType)
        await self.result_set.promote_children(Result, server.ResultType, server.ResultSetType)
        self.active_program = await ActiveProgram.promote(await self.get_lads_child("ActiveProgram"), server)

    @property
    def variables(self) ->list[BaseVariable]:
        return self.active_program.variables + [self.program_template_set.node_version, self.result_set.node_version]

    @property
    def program_templates(self) -> list[ProgramTemplate]:
        return self.program_template_set.children

    @property
    def program_template_names(self) -> list[str]:
        return list(map(lambda template: template.unique_name, self.program_templates))

    @property
    def results(self) -> list[Result]:
        return self.result_set.children
