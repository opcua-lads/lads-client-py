"""LADSTypes: lookup of the LADS/DI/Machinery types and namespaces of a server."""

from __future__ import annotations

from asyncua import Client, Node, ua
from typing import Type
from .ids import DIObjectIds, LADSObjectIds, LADS_CD_ObjectIds, MachineryObjectIds
from .log import _logger


# MARK: LADSTypes
class LADSTypes:

    def __init__(self, client: Client):
        self.client = client

    async def init(self) -> dict:
        # read namespace indices
        from .connection import Connection  # lazy: avoids a circular import
        self.ns_DI = await self.client.get_namespace_index("http://opcfoundation.org/UA/DI/")
        self.ns_AMB = await self.client.get_namespace_index("http://opcfoundation.org/UA/AMB/")
        self.ns_Machinery = await self.client.get_namespace_index("http://opcfoundation.org/UA/Machinery/")
        self.ns_LADS = await self.client.get_namespace_index("http://opcfoundation.org/UA/LADS/")
        try:
            self.ns_LADS_CD = await self.client.get_namespace_index("http://aixengineers.de/LADS-CD/")
        except:
            self.ns_LADS_CD = None

        # get well known type nodes
        self.BaseObjectType = self.get_node(ua.ObjectIds.BaseObjectType)
        self.FiniteStateMachineType = self.get_node(ua.ObjectIds.FiniteStateMachineType)
        self.BaseVariableType = self.get_node(ua.ObjectIds.BaseVariableType)
        self.AnalogItemType = self.get_node(ua.ObjectIds.AnalogItemType)
        self.TwoStateDiscreteType = self.get_node(ua.ObjectIds.TwoStateDiscreteType)
        self.MultiStateDiscreteType = self.get_node(ua.ObjectIds.MultiStateDiscreteType)
        self.EnumerationType = self.get_node(ua.ObjectIds.Enumeration)
        self.ExclusiveLimitAlarmType = self.get_node(ua.ObjectIds.ExclusiveLimitAlarmType)
        self.LifetimeVariableType = self.get_di_node(DIObjectIds.LifetimeVariableType)
        self.MachineryItemIdentificationType = self.get_machinery_node(MachineryObjectIds.MachineryItemIdentificationType)
        self.MachineryOperationCounterType = self.get_machinery_node(MachineryObjectIds.MachineryOperationCounterType)
        self.MachineryLifeTimeCounterType = self.get_machinery_node(MachineryObjectIds.MachineryLifeTimeCounterType)
        self.DeviceType = self.get_lads_node(LADSObjectIds.DeviceType)
        self.SetType = self.get_lads_node(LADSObjectIds.SetType)
        self.ComponentSetType = self.get_lads_node(LADSObjectIds.ComponentSetType)
        self.ComponentType = self.get_lads_node(LADSObjectIds.ComponentType)
        self.LockingServicesType = self.get_di_node(DIObjectIds.LockingServicesType)
        self.FunctionalUnitSetType = self.get_lads_node(LADSObjectIds.FunctionalUnitSetType)
        self.FunctionalUnitType = self.get_lads_node(LADSObjectIds.FunctionalUnitType)
        self.FunctionSetType = self.get_lads_node(LADSObjectIds.FunctionSetType)
        self.FunctionType = self.get_lads_node(LADSObjectIds.FunctionType)
        self.AnalogScalarSensorFunctionType = self.get_lads_node(LADSObjectIds.AnalogScalarSensorFunctionType)
        self.AnalogScalarSensorFunctionWithCompensationType = self.get_lads_node(LADSObjectIds.AnalogScalarSensorFunctionWithCompensationType)
        self.AnalogArraySensorFunctionType = self.get_lads_node(LADSObjectIds.AnalogArraySensorFunctionType)
        self.TwoStateDiscreteSensorFunctionType = self.get_lads_node(LADSObjectIds.TwoStateDiscreteSensorFunctionType)
        self.MultiStateDiscreteSensorFunctionType = self.get_lads_node(LADSObjectIds.MultiStateDiscreteSensorFunctionType)
        self.AnalogControlFunctionType = self.get_lads_node(LADSObjectIds.AnalogControlFunctionType)
        self.AnalogControlFunctionWithTotalizerType = self.get_lads_node(LADSObjectIds.AnalogControlFunctionWithTotalizerType)
        self.AnalogControlFunctionWithComposedTargetValueType = self.get_lads_node(LADSObjectIds.AnalogControlFunctionWithComposedTargetValueType)
        self.AnalogControlFunctionWithRelativeTargetValueType = self.get_lads_node(LADSObjectIds.AnalogControlFunctionWithRelativeTargetValueType)
        self.TimerControlFunctionType = self.get_lads_node(LADSObjectIds.TimerControlFunctionType)
        self.TwoStateDiscreteControlFunctionType = self.get_lads_node(LADSObjectIds.TwoStateDiscreteControlFunctionType)
        self.MultiStateDiscreteControlFunctionType = self.get_lads_node(LADSObjectIds.MultiStateDiscreteControlFunctionType)
        self.MultiModeControlFunctionType = self.get_lads_node(LADSObjectIds.MulitModeControlFunctionType)
        self.ControllerParameterType = self.get_lads_node(LADSObjectIds.ControllerParameterType)
        self.ControllerParameterSetType = self.get_lads_node(LADSObjectIds.ControllerParameterSetType)
        self.StartStopControlFunctionType = self.get_lads_node(LADSObjectIds.StartStopControlFunctionType)
        self.CoverFunctionType = self.get_lads_node(LADSObjectIds.CoverFunctionType)
        self.ProgramManagerType = self.get_lads_node(LADSObjectIds.ProgramManagerType)
        self.ProgramTemplateSetType = self.get_lads_node(LADSObjectIds.ProgramTemplateSetType)
        self.ProgramTemplateType = self.get_lads_node(LADSObjectIds.ProgramTemplateType)
        self.ActiveProgramType = self.get_lads_node(LADSObjectIds.ActiveProgramType)
        self.ResultSetType = self.get_lads_node(LADSObjectIds.ResultSetType)
        self.ResultType = self.get_lads_node(LADSObjectIds.ResultType)
        self.ResultFileSetType = self.get_lads_node(LADSObjectIds.ResultFileSetType)
        self.ResultFileType = self.get_lads_node(LADSObjectIds.ResultFileType)
        self.VariableSetType = self.get_lads_node(LADSObjectIds.VariableSetType)
        # LADS Compliance Documemts (experimental)
        if self.ns_LADS_CD is not None:
            self.ComplianceDocumentSetType = self.get_lads_cd_node(LADS_CD_ObjectIds.ComplianceDocumentSetType)
            self.ComplianceDocumentType = self.get_lads_cd_node(LADS_CD_ObjectIds.ComplianceDocumentType)
            self.HasComplianceDocument = self.get_lads_cd_node(LADS_CD_ObjectIds.HasComplianceDocument)
            self.HasCalibrationCertificate = self.get_lads_cd_node(LADS_CD_ObjectIds.HasCalibrationCertificate)
            self.HasValidationReport = self.get_lads_cd_node(LADS_CD_ObjectIds.HasValidationReport)
            self.HasQualificationProtocol = self.get_lads_cd_node(LADS_CD_ObjectIds.HasQualificationProtocol)

        # read data tyoes only once - asyncua design problem..
        if Connection.data_types is None:
            Connection.data_types = {"Locked": True}
            Connection.data_types = await self.client.load_data_type_definitions(overwrite_existing=False)
            server_url = self.client.server_url
            # print(f"Datatypes loaded from server {server_url.scheme}://{server_url.netloc}")
            # kv = self.KeyValueType("MyProperty", "42.0")
            # dt = self.client.get_node(kv.data_type)
            # dt_def = await dt.read_data_type_definition()

    def data_type(self, name: str) -> Type:
        from .connection import Connection  # lazy: avoids a circular import
        try:
            return Connection.data_types[name]
        except:
            _logger.error(f"Unable to load datatype {name}")
            return None

    @property
    def KeyValueType(self) -> Type:
        return self.data_type("KeyValueType")

    @property
    def SampleInfoType(self) -> Type:
        return self.data_type("SampleInfoType")

    def get_node(self, id: ua.NodeId | int) -> Node | None:
        if isinstance(id, ua.NodeId):
            return self.client.get_node(id)
        else:
            return self.client.get_node(ua.NodeId(int(id)))

    def get_di_node(self, id: int) -> Node | None:
        return self.client.get_node(ua.NodeId(int(id), self.ns_DI))

    def get_machinery_node(self, id: int) -> Node | None:
        return self.client.get_node(ua.NodeId(int(id), self.ns_Machinery))

    def get_lads_node(self, id: int) -> Node | None:
        return self.client.get_node(ua.NodeId(int(id), self.ns_LADS))

    def get_lads_cd_node(self, id: int) -> Node | None:
        if self.ns_LADS_CD is not None:
            return self.client.get_node(ua.NodeId(int(id), self.ns_LADS_CD))
        else:
            return None
