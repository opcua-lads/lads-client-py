"""Python LADS OPC UA Client
    
    This library provides a Python OPC UA client for the LADS OPC UA server.
    The client is based on the asyncua library and provides a set of classes to interact with a LADS OPC UA server.
    The classes are based on the LADS OPC UA model, providing a high level interface.

    The library provides the following classes:
    - Server: Represents a LADS OPC UA server with a set of devices and functional units.
    - SubscriptionHandler: Handles data change and event notifications for subscribed variables.
    - LADSNode: Represents a node in the LADS OPC UA model.
    - BaseVariable: Represents a variable in the LADS OPC UA model.
    - Device: Represents a device in the LADS OPC UA model.
    - Function: Represents a function in the LADS OPC UA model.
    - FunctionSet: Represents a set of functions in the LADS OPC UA model.
    - FunctionalUnit: Represents a functional unit in the LADS OPC UA model.
    - Connection: Represents a connection to a LADS OPC UA server.
    - Connections: Represents a set of connections to LADS OPC UA servers.

    Copyright (c) 2023 - 2025 Dr. Matthias Arnold, AixEngineers, Aachen, Germany.
    
    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
"""

import asyncio
import logging
import sys
import threading
import time
import json
import pandas as pd
import datetime as dt
from typing import List, Type, NewType, Any, Self, Tuple, Set
from asyncua import Client, ua, Node
from asyncua.common.subscription import DataChangeNotif
from asyncua.common.events import Event
from asyncua.common.ua_utils import is_subtype, get_node_supertypes
from enum import IntEnum
from queue import Queue

AFOSupport = True

if AFOSupport:
    from .afo import DictionaryEntry, get_entry

# initialize logger
level = logging.DEBUG
_logger = logging.getLogger(__name__)
_logger.setLevel(level)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(level)
formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d in %(funcName)s() → %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
if not _logger.handlers:
    _logger.addHandler(console_handler)

# pre-define some types
LADSNode = NewType("LADSNode", Node)
BaseVariable = NewType("BaseVariable", LADSNode)
Method = NewType("Method", LADSNode)
Component = NewType("Component", LADSNode)
Device = NewType("Device", Component)
FunctionalUnit = NewType("FunctionalUnit", LADSNode)
Function = NewType("Function", LADSNode)

# MARK: Node IDs
# pylint: disable=C0103
class LADSObjectIds(IntEnum):
    """LADS specific numerical node-ids"""
    DeviceType = 1002
    ComponentSetType = 1025
    ComponentType = 1024
    SetType = 61
    FunctionalUnitSetType = 1023
    FunctionalUnitType= 1003
    FunctionSetType = 1026
    FunctionType= 1004
    AnalogScalarSensorFunctionType = 1016
    AnalogScalarSensorFunctionWithCompensationType = 1000
    AnalogArraySensorFunctionType = 1015
    TwoStateDiscreteSensorFunctionType = 1031
    MultiStateDiscreteSensorFunctionType = 1037
    AnalogControlFunctionType = 1009
    AnalogControlFunctionWithTotalizerType = 1014
    TimerControlFunctionType = 1013
    TwoStateDiscreteControlFunctionType = 1042
    MultiStateDiscreteControlFunctionType = 1045
    MulitModeControlFunctionType = 1047
    ControllerParameterType = 1048
    ControllerParameterSetType = 1049
    StartStopControlFunctionType = 1032
    CoverFunctionType = 1011
    ProgramManagerType = 1006
    ProgramTemplateSetType = 1019
    ProgramTemplateType = 1018
    ActiveProgramType = 1040
    ResultSetType = 1020
    ResultType = 1021
    ResultFileSetType = 1022
    ResultFileType = 1001
    VariableSetType = 1041

class LADS_CD_ObjectIds(IntEnum):
    """LADS Compliance Document specific numerical node-ids"""
    ComplianceDocumentSetType = 1000
    ComplianceDocumentType = 1001
    HasComplianceDocument = 4000
    HasCalibrationCertificate = 4001
    HasValidationReport = 4002
    HasQualificationProtocol = 4003

class MachineryObjectIds(IntEnum):
    """Machinery specific numerical node-ids"""
    MachineryItemIdentificationType = 1004
    MachineryOperationCounterType = 1009
    MachineryLifeTimeCounterType = 1015

class DIObjectIds(IntEnum):
    """DI specific numerical node-ids"""
    LockingServicesType = 6388
    MaxInactiveLockTime = 6387
    DeviceHealthEnumeration = 6244
    LifetimeVariableType = 468

# MARK: SubscriptionLevel
class SubscriptionLevel(IntEnum):
    """Subscription levels for variables"""
    Never = 0
    Temporary = 1
    Permanent = 2

# MARK: LADSTypes
class LADSTypes:

    def __init__(self, client: Client):
        self.client = client

    async def init(self) -> dict:
        # read namespace indices
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

# MARK: Server
class Server(LADSTypes):
    """
    Represents a LADS OPC UA server with a set of devices and functional units.

    Attributes:
        client: Client - the client object used to connect to the server
        name: str - the name of the server
        devices: list[Device] - the devices in the server
        initialized: bool - True if the server has been initialized
        running: bool - True if the server is running
        call_async_queue: Queue - queue for async calls
        functional_units: list[FunctionalUnit] - all functional units in the server
    """

    def __init__(self, client: Client):
        super().__init__(client)
        self.name = "Unknown Server"
        self.devices: list[Device] = []
        self.initialized = False
        self.running = True
        self.call_async_queue = Queue()

    async def _disconnect(self):
        print(f"Disconnecting from {self.name}")
        await self.client.disconnect()
        self.running = False
        self.initialized = False
        print("Disconnected successfully.")

    async def init(self) -> dict:
        data_types = await super().init()
            
        # browse for devices in DeviceSet
        product_uri = self.client.get_node(ua.ObjectIds.Server_ServerStatus_BuildInfo_ProductUri)
        self.name: str = await product_uri.read_value()

        # locking services
        self.max_inactive_lock_time = await BaseVariable.promote(self.get_di_node(DIObjectIds.MaxInactiveLockTime), self)
        if self.max_inactive_lock_time is not None:
            await self.max_inactive_lock_time.read_data_value()
        
        # devices
        device_set = await self.client.nodes.objects.get_child(f"{self.ns_DI}:DeviceSet")
        nodes = await device_set.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        for node in nodes:
            try:
                await self.client.check_connection()
                device: Device = await Device.promote(node, self)
                await device.finalize_init()
                self.devices.append(device)
            except Exception as error:
                _logger.error(error, extra=["node", node])
                return data_types

        self.initialized = True
        

        return data_types

    async def evaluate(self):
        if not self.call_async_queue.empty():
            item = self.call_async_queue.get()
            if item is not None:
                try:
                    result = await item
                    _logger.debug(f"Evaluated item {item} with result {result}")
                except Exception as error:
                    _logger.debug(error)
        await asyncio.sleep(0.01)

    @property
    def functional_units(self) -> list[FunctionalUnit]:
        if not self.initialized: return []
        functional_units: list[FunctionalUnit] = []
        for device in self.devices:
            functional_units = functional_units + device.functional_units
        return functional_units
    
async def get_parent_nodes(server: Server, node: Node, root_node: Node = None) -> list[Node]:
    if node is None:
        return[]
    parent = await node.get_parent()
    if parent == root_node:
        return [parent]
    else:
        return [parent] + await get_parent_nodes(server, parent, root_node)

async def browse_types(server: Server, node: Node) -> list[Node]:
    type_node_id =  await node.read_type_definition()
    type_node = server.get_node(type_node_id)
    return await get_node_supertypes(type_node, includeitself=True)

async def is_of_type(server: Server, node: Node, super_type_node: Node) -> bool:
    type_node_id = await node.read_type_definition()
    type_node = server.client.get_node(type_node_id)
    result = await is_subtype(type_node, super_type_node.nodeid)
    return result

unique_name_delimiter = "/"

def variant_value_to_str(variant: ua.Variant) -> str:
    if variant is None:
        return "unknown"
    value = variant.Value
    if isinstance(value,ua.LocalizedText):
        return value.Text if value.Text is not None else ""
    elif isinstance(value, ua.QualifiedName):
        return  value.Name
    elif isinstance(value, dt.datetime):
        return  value.strftime("%d.%m.%Y %H:%M:%S")
    else:
        s = str(value)
        if s.startswith("NameNodeIdDataType"):
            try:
                name: ua.LocalizedText = value.Name
                node_id: ua.NodeId = value.NodeId
                return name.Text
            except:
                return s
        else:
            return s

def duration_to_str(ms: float, ms_digits: int = 3) -> str:
    if ms is None:
        return ""
    if not 0 <= ms_digits <= 3:
        raise ValueError("ms_digits must be between 0 and 3")
    factor = 10 ** (3 - ms_digits)
    total_ms = int(round(ms / factor) * factor)

    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)

    result = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    if ms_digits:
        fraction = millis // factor
        result += f",{fraction:0{ms_digits}d}"

    return result

def remove_none(nodes: list[Node]) -> list[Node]:
    return list(filter(lambda node: node is not None, nodes))

# MARK: SubscriptionHandler
class SubscriptionHandler(object):
    """
    Handles data change and event notifications for subscribed variables.

    Attributes:
        subscription: Subscription - the subscription object from the asyncua library.
        subscribed_variables: dict[ua.NodeId, BaseVariable] - the subscribed variables as a dictionary.
        event_node: LADSNode - the event LADS node.
        events: pd.DataFrame - the events as a pandas dataframe.
        last_event_update: dt.datetime - the last event update time.
    
    Methods:
        subscribe_data_change(self, server: Server, nodes: list[BaseVariable], period: float = 500) - subscribe to data change notifications.
        subscribe_events(self, server: Server, node: Node, period: float = 500) - subscribe to event notifications.
        datachange_notification(self, node: Node, val: Any, data: DataChangeNotif) - handle data change notifications.
        event_notification(self, event: Event) - Notification of an event.
        status_change_notification(self, status: Any) - Notification of a status change.
    """

    def __init__(self) -> None:
        super().__init__()
        self.subscription = None
        self.subscribed_variables = {}
        self.event_node = None
        self.events: pd.DataFrame = None
        self.last_event_update = dt.datetime.now()

    async def subscribe_data_change(self, server: Server, nodes: list[BaseVariable], period: float = 500):
        """
        Subscribe to data change.

        Args:
            server: Server - the server object.
            nodes: list[BaseVariable] - the variables to subscribe to.
            period: float - the subscription period.
        """
        # make sure nodes are not none
        if len(nodes) == 0: 
            return
        nodes = remove_none(nodes)
        if self.subscription is None:
            self.subscription = await server.client.create_subscription(period, self)
        self.subscribed_variables = dict((node.nodeid, node) for node in nodes)
        result = await self.subscription.subscribe_data_change(nodes) 
        return result
 
    async def subscribe_events(self, server: Server, node: Node, period: float = 500):
        """
        Subscribe to events.

        Args:
            server: Server - the server object.
            node: Node - the event node.
            period: float - the subscription period.
        """
        
        if self.subscription is None:
            self.subscription = await server.client.create_subscription(period, self)
        self.event_node: LADSNode = node
        return await self.subscription.subscribe_events(node)        
 
    async def datachange_notification(self, node: Node, val: Any, data: DataChangeNotif):
        """
        Notification of a data change.

        Args:
            node: Node - the node.
            val: Any - the value.
            data: DataChangeNotif - the data change notification.
        """

        try:
            variable: Node = self.subscribed_variables[node.nodeid]
            variable.data_change_notification(data)
        except Exception as error:
            _logger.error(f"data_change_notification error {error}")

    async def event_notification(self, event: Event):
        """
        Notification of an event.

        Args:
            event: Event - the event.
        """

        # obviously there is a bug in the library subscription.py
        #   async def _call_event(self, eventlist: ua.EventNotificationList) -> None: 
        # ua.EventNotificationList has always only one element, even if multiple events are sent..
        fields_dict = event.get_event_props_as_fields_dict()
        event_fields = {}
        try:
            event_fields = {k: variant_value_to_str(v) for k, v in fields_dict.items()}
        except Exception as error:
            _logger.error(error)
        print(event_fields["Time"], event_fields["SourceName"], event_fields["Message"])
        key = pd.to_datetime(dt.datetime.now())
        if self.events is None:
            self.events = pd.DataFrame(event_fields, index = [key])
        else:
            self.events.loc[key] = event_fields
            if len(self.events.index) > 1000:
                self.events = self.events.tail(-10)
        self.last_event_update = key

    async def status_change_notification(self, status: Any):
        """
        Notification of a status change.

        Args:
            status: Any - the status.
        """

        print(status)

# MARK: LADSNode
class LADSNode(Node):
    """
    Represents a node in the LADS OPC UA model.

    Attributes:
        server: Server - the server object.
        browse_name: ua.QualifiedName - the browse name of the node.
        display_name: str - the display name of the node.
        description: str - the description of the node.
        unique_name: str - the unique name of the node.
        variables: list[BaseVariable] - the variables of the node.
        subscribed_variables: list[BaseVariable] - the subscribed variables of the node.
        permanent_subscribed_variables: list[BaseVariable] - the permanent subscribed variables of the node.
        temporary_subscribed_variables: list[BaseVariable] - the temporary subscribed variables of the node.
    
    Methods:
        promote(cls, node: Node, server: Server) -> Self - promote a asyncua node to a LADSNode.
        variable_named(self, name: str) -> BaseVariable - get a variable by name.
        update_variables_async(self) - update the variables of the node asynchronously.
        update_variables(self) - update the variables of the node.
        call_async(self, func) - call a function asynchronously.
        get_child_or_none(self, name: ua.QualifiedName) -> Node - get a child node or None by name.
        get_di_child(self, name: str) -> Node - get a DI child node by name.
        get_di_variable(self, name: str) -> BaseVariable - get a DI variable by name.
        get_machinery_child(self, name: str) -> Node - get a machinery child node by name.
        get_machinery_variable(self, name: str) -> BaseVariable - get a machinery variable by name.
        get_lads_child(self, name: str) -> Node - get a LADS child node by name.
        get_lads_variable(self, name: str) -> BaseVariable - get a LADS variable by name.
        get_child_objects(self, parent: Node = None) -> list[Node] - get all child objects of a node or parent node.
        call_lads_method(self, name: str, *args: Any) -> ua.StatusCode - call a LADS method by name.
    """

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(LADSNode, node, server.BaseObjectType, server)

    def __str__(self):
        return f"{self.__class__.__name__}({self.display_name})"
    
    async def init(self, server: Server):
        """
        Initialize the LADSNode.

        Args:
            server: Server - the server object.
        """

        self.server: Server = server
        (self.browse_name, self._display_name, self.description, self.dictionary_entries)  = await asyncio.gather(
            self.read_browse_name(),
            self.read_display_name(),
            self.read_description(),
            self.read_dictionary_entries()
        )
        if AFOSupport:
            self._dictionary_entry_objects = None
            self._dcitionary_entries_as_markdown = None
        msg = f"Initializing {self.__class__.__name__}({self.display_name})"
        _logger.info(msg)

    async def finalize_init(self):
        pass

    ###################################################
    # Work around for buggy _to_nodeid() implementation
    # Issue number: TBD
    async def get_references(
        self,
        refs: int = ua.ObjectIds.References,
        direction: ua.BrowseDirection = ua.BrowseDirection.Both,
        nodeclassmask: ua.NodeClass = ua.NodeClass.Unspecified,
        includesubtypes: bool = True,
        result_mask: ua.BrowseResultMask = ua.BrowseResultMask.All
    ) -> List[ua.ReferenceDescription]:
        """
        returns references of the node based on specific filter defined with:

        refs = ObjectId of the Reference
        direction = Browse direction for references
        nodeclassmask = filter nodes based on specific class
        includesubtypes = If true subtypes of the reference (ref) are also included
        result_mask = define what results information are requested
        """
        def _to_nodeid(nodeid: int):
            if nodeid <= 255:
                return ua.TwoByteNodeId(nodeid)
            elif nodeid <= 65535:
                return ua.FourByteNodeId(nodeid)
            else:
                return ua.NumericNodeId(nodeid)
        
        desc = ua.BrowseDescription()
        desc.BrowseDirection = direction
        desc.ReferenceTypeId = _to_nodeid(refs)
        desc.IncludeSubtypes = includesubtypes
        desc.NodeClassMask = nodeclassmask
        desc.ResultMask = result_mask
        desc.NodeId = self.nodeid
        params = ua.BrowseParameters()
        params.View.Timestamp = ua.get_win_epoch()
        params.NodesToBrowse.append(desc)
        params.RequestedMaxReferencesPerNode = 0
        results = await self.session.browse(params)
        references = await self._browse_next(results)
        return references
    
    async def get_references_of_type(
        self,
        reference_type: Node,
        direction: ua.BrowseDirection = ua.BrowseDirection.Both,
        nodeclassmask: ua.NodeClass = ua.NodeClass.Unspecified,
        includesubtypes: bool = True,
        result_mask: ua.BrowseResultMask = ua.BrowseResultMask.All
    ) -> List[ua.ReferenceDescription]:
        """
        returns references of the node based on specific filter defined with:

        refs = ObjectId of the Reference
        direction = Browse direction for references
        nodeclassmask = filter nodes based on specific class
        includesubtypes = If true subtypes of the reference (ref) are also included
        result_mask = define what results information are requested
        """
        desc = ua.BrowseDescription()
        desc.BrowseDirection = direction
        desc.ReferenceTypeId = reference_type.nodeid
        desc.IncludeSubtypes = includesubtypes
        desc.NodeClassMask = nodeclassmask
        desc.ResultMask = result_mask
        desc.NodeId = self.nodeid
        params = ua.BrowseParameters()
        params.View.Timestamp = ua.get_win_epoch()
        params.NodesToBrowse.append(desc)
        params.RequestedMaxReferencesPerNode = 0
        results = await self.session.browse(params)
        references = await self._browse_next(results)
        return references


    @property
    def display_name(self) -> str:
        if self._display_name is not None:
            return self._display_name.Text
        else:
            return self.browse_name.Name

    @property
    def unique_name(self) -> str:
        return self.display_name
    
    @property
    def variables(self) ->list[BaseVariable]:
        return []
    
    @property
    def subscribed_variables(self) ->list[BaseVariable]:
        return list(filter(lambda variable: variable.subscription_level > SubscriptionLevel.Never, self.variables))

    @property
    def permanent_subscribed_variables(self) ->list[BaseVariable]:
        return list(filter(lambda variable: variable.subscription_level == SubscriptionLevel.Permanent, self.variables))
    
    @property
    def temporary_subscribed_variables(self) ->list[BaseVariable]:
        return list(filter(lambda variable: variable.subscription_level == SubscriptionLevel.Temporary, self.variables))
    
    def variable_named(self, name: str) -> BaseVariable:
        for variable in self.variables:
            if name == variable.browse_name.Name:
                return variable
        return None
    
    async def update_variables_async(self):
        variables = remove_none(self.variables)
        await asyncio.gather(*(variable.update_value() for variable in variables))

    def update_variables(self):
        self.call_async(self.update_variables_async())

    def call_async(self, func):
        self.server.call_async_queue.put(func)

    async def get_child_or_none(self, name : ua.QualifiedName) -> Node:
        try:
            return await self.get_child(name)
        except:
            return None
        
    async def get_di_child(self, name : str) -> Node:
        return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_DI))
    
    async def get_di_variable(self, name : str) -> BaseVariable:
        return await BaseVariable.promote(await self.get_di_child(name), self.server)
    
    async def get_amb_child(self, name : str) -> Node:
        return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_AMB))
    
    async def get_amb_variable(self, name : str) -> BaseVariable:
        return await BaseVariable.promote(await self.get_amb_child(name), self.server)
    
    async def get_machinery_child(self, name : str) -> Node:
        return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_Machinery))
    
    async def get_machinery_variable(self, name : str) -> BaseVariable:
        return await BaseVariable.promote(await self.get_machinery_child(name), self.server)
    
    async def get_lads_child(self, name : str) -> Node:
        return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_LADS))
    
    async def get_lads_variable(self, name : str) -> BaseVariable:
        return await BaseVariable.promote(await self.get_lads_child(name), self.server)
    
    async def get_lads_cd_child(self, name : str) -> Node:
        if self.server.ns_LADS_CD is not None:
            return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_LADS_CD))
        else:
            return None
    
    async def get_lads_cd_variable(self, name : str) -> BaseVariable:
        if self.server.ns_LADS_CD is not None:
            return await BaseVariable.promote(await self.get_lads_cd_child(name), self.server)
        else:
            return None
            
    async def get_child_objects(self, parent: Node = None) -> list[Node]:
        if parent is None: parent = self
        # search for HasChild and Organizes references
        (has_child_objects, organizes_objects) = await asyncio.gather(
            parent.get_children(refs = ua.ObjectIds.Aggregates, nodeclassmask = ua.NodeClass.Object),
            parent.get_children(refs = ua.ObjectIds.Organizes, nodeclassmask = ua.NodeClass.Object)
        )
        # reduce results to set
        child_objects = set(has_child_objects)
        child_objects.update(organizes_objects)
        return list(child_objects)

    async def call_namespace_method(self, name: str, ns: int, *args: Any) -> ua.StatusCode:
        try:
            _logger.debug(f"Call method {name} with args {args}")
            return await self.call_method(ua.QualifiedName(name, ns), *args)
        except Exception as error:
            _logger.error(error)
            return ua.StatusCodes.BadNotImplemented
        
    async def call_lads_method(self, name: str, *args: Any) -> ua.StatusCode:
        return await self.call_namespace_method(name, self.server.ns_LADS, *args)
    
    async def call_di_method(self, name: str, *args: Any) -> ua.StatusCode:
        return await self.call_namespace_method(name, self.server.ns_DI, *args)
    
    if AFOSupport:
        @property
        def dictionary_entry_objects(self) -> list[DictionaryEntry]:
            if self._dictionary_entry_objects is None:
                self._dictionary_entry_objects: list[DictionaryEntry] = list(filter(lambda entry: entry is not None, map(lambda dictionary_entry: get_entry(dictionary_entry), self.dictionary_entries)))

            return self._dictionary_entry_objects 

    @property
    def dictionary_entries_as_markdown(self) -> str:
        if AFOSupport:
            if self._dcitionary_entries_as_markdown is None:
                definitions: list[str] = []
                for entry in self.dictionary_entry_objects:
                    if entry is not None:
                        markdown = f"**{entry.prefLabel}**  \r\n{entry.definition}  "
                    definitions.append(markdown)
                self._dcitionary_entries_as_markdown = "\n\r".join(definitions)
            return self._dcitionary_entries_as_markdown
        else:
            return ""

    async def read_dictionary_entries(self) -> list[str]:
        if AFOSupport:
            try:
                nodes = await self.get_referenced_nodes(refs = ua.ObjectIds.HasDictionaryEntry, direction = ua.BrowseDirection.Forward, nodeclassmask = ua.NodeClass.Object)
                if  len(nodes) == 0:
                    return []
                names = list(map(lambda node: node.nodeid.Identifier, nodes))
                return names
            except Exception as err:
                print(f"Exception when reading dictionary entries {err}")
                return []
        else:
            return []

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
from math import log10, trunc
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
            _logger.debug(f"Unable to determine decimals of {self.display_name}-{self.nodeid}: EURange has no value")            
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
import re
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
        key_value_list = None
        # key_value_list = []
        for index, row in properties.iterrows():
            key = str(row["Key"])
            value =str(row["Value"])
            key_value = self.server.KeyValueType(
                key,
                value,
            )
            if key_value_list is None:
                key_value_list = []
            key_value_list.append(key_value)
        return key_value_list

    def start_program(self, program_template: str, properties: pd.DataFrame, supervisory_job_id: str, supervisory_task_id: str, samples: pd.DataFrame):
        program_template_name = extract_program_template_name(program_template)
        key_value_list = self.buildProperties(properties)
        sample_info_list = None
        # sample_info_list = []
        for index, row in samples.iterrows():
            sample_info = self.server.SampleInfoType(
                str(row["ContainerId"]),
                str(row["SampleId"]),
                str(row["Position"]),
                str(row["CustomData"]),
            )
            if sample_info_list is None:
                sample_info_list = []
            sample_info_list.append(sample_info)
        self.call_async(self.call_lads_method("StartProgram", 
                                              program_template_name, 
                                              key_value_list, 
                                              supervisory_job_id, 
                                              supervisory_task_id, 
                                              sample_info_list))
            
    def start(self, properties: pd.DataFrame):
        key_value_list = self.buildProperties(properties)
        self.call_async(self.call_lads_method("Start", key_value_list))

    def start_with_target_value(self, value: float):
        self.call_async(self.call_lads_method("StartWithTargetValue", value))

    def stop(self):
        self.call_async(self.call_lads_method("Stop"))

# MARK: LADSSet
class LADSSet(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(LADSSet, node, server.SetType, server)
    
    async def init(self, server: Server):
        self.node_version: NodeVersionVariable = None
        await super().init(server)
        try:
            # node_version variable is optional
            node_version = await self.get_child("NodeVersion")
            self.node_version = await NodeVersionVariable.promote(node_version, server)
            self.node_version.set = self
        except Exception as error:
             _logger.warning(error)
        self.children = await self.get_child_objects()

    async def promote_children(self, child_class: Type, child_type: Node, set_type: Node):
        if self.children is None: 
            return
        if set_type is not None:
            assert(await is_of_type(self.server, self, set_type))
        self.child_class = child_class
        self.child_type = child_type
        self.children = await asyncio.gather(*(self.promote_child(child) for child in self.children))
        self.children.sort(key = lambda child: child.display_name)

    async def promote_child(self, child: Node) -> LADSNode:
        return await promote_to(self.child_class, child, self.child_type, self.server)
    
    @property
    def variables(self) ->list[BaseVariable]:
        return [] if self.node_version is None else [self.node_version]
    
    def node_version_changed(self):
        _logger.debug(f"NodeVersion of {self.display_name} changed.")
        self.call_async(self.update_children())
    
    async def update_children(self):
        current_nodes = await self.get_child_objects(self)
        current_node_ids = set(map(lambda node: node.nodeid, current_nodes))
        previous_nodes = self.children
        previous_node_ids = set(map(lambda node: node.nodeid, previous_nodes))
        new_node_ids = current_node_ids.difference(previous_node_ids)
        deleted_node_ids = previous_node_ids.difference(current_node_ids)
        if len(new_node_ids) > 0:
            for node_id in new_node_ids:
                nodes = list(filter(lambda node: node.nodeid == node_id, current_nodes))
                assert(len(nodes) == 1)
                node = await self.promote_child(nodes[0])
                self.children.append(node)
        if len(deleted_node_ids) > 0:
            for node_id in deleted_node_ids:
                nodes = list(filter(lambda node: node.nodeid == node_id, previous_nodes))
                assert(len(nodes) == 1)
                node = nodes[0]
                self.children.remove(node)

# MARK: ComponentSet
class ComponentSet(LADSSet):
    # since the Machinery type Components is not derived from LADS.SetType we need a different type check
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(ComponentSet, node, server.ComponentSetType, server)

# MARK: OperationCounters
class OperationCounters(LADSNode):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(OperationCounters, node, server.MachineryOperationCounterType, server)
    
    async def init(self, server: Server):
        await super().init(server)
        self.operation_cycle_counter, self.operation_duration, self.power_on_duration = await asyncio.gather(
            self.get_di_variable("OperationCycleCounter"),
            self.get_di_variable("OperationDuration"),
            self.get_di_variable("PowerOnDuration"),
        )
        for variable in self.variables:
            variable.subscription_level = SubscriptionLevel.Temporary
            
    @property
    def variables(self) -> list[BaseVariable]:
        return remove_none([self.operation_cycle_counter, self.operation_duration, self.power_on_duration])

# MARK: LifetimeCounters
class LifetimeCounters(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(LifetimeCounters, node, server.MachineryLifeTimeCounterType, server)
        
    async def init(self, server: Server):
        await super().init(server)
        nodes = await get_properties_and_variables(self)
        self.lifetime_counters: list[LifetimeCounter] = await asyncio.gather(*(LifetimeCounter.promote(node, server) for node in nodes))
        self.lifetime_counters.sort(key = lambda node: node.display_name)

    @property
    def variables(self) -> list[Node]:
        return super().variables + self.lifetime_counters

# MARK: Identification
class Identification(LADSNode):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Identification, node, server.MachineryItemIdentificationType, server)
    
    async def init(self, server: Server):
        self.asset_id: BaseVariable = None
        self.component_name: BaseVariable = None
        self.location: BaseVariable = None

        await super().init(server)
        self._variables = await get_properties_and_variables(self)
        self._variables.sort(key = lambda variable: variable.display_name)
        self.asset_id = self.variable_named("AssetId")
        self.component_name = self.variable_named("ComponentName")
        self.location = self.variable_named("Location")
        subscription_variables: list[BaseVariable] = remove_none([self.asset_id, self.component_name, self.location])
        for variable in subscription_variables:
            variable.subscription_level = SubscriptionLevel.Permanent

    @property
    def variables(self) ->list[BaseVariable]:
        return self._variables

# MARK: Component
class Component(LADSNode):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Component, node, server.ComponentType, server)
    
    async def init(self, server: Server):
        self.component_set: LADSSet = None
        self.device_health: Enumeration = None
        self.operation_counters: OperationCounters = None
        self.lifetime_counter_set: LifetimeCounters = None
        self.identification: Identification = None
        await super().init(server)
        
        self.operation_counters = await OperationCounters.promote(await self.get_di_child("OperationCounters"), server)
        self.lifetime_counter_set = await LifetimeCounters.promote(await self.get_machinery_child("LifetimeCounters"), server)
        self.identification = await Identification.promote(await self.get_di_child("Identification"), server)
        self._variables = await get_properties_and_variables(self)
        if self.operation_counters is not None:
            self._variables = self._variables + self.operation_counters.variables
        self._variables.sort(key = lambda variable: variable.display_name)
        self.device_health = self.variable_named("DeviceHealth")
        if self.device_health is not None:
            self.device_health = await Enumeration.promote(self.device_health, server)
        self.component_set = await ComponentSet.promote(await self.get_machinery_child("Components"), server)
        if self.component_set is not None:
            await self.component_set.promote_children(Component, server.ComponentType, server.ComponentSetType)

    @property
    def components(self) -> list[Component]:
        return [] if self.component_set is None else self.component_set.children
        
    @property
    def lifetime_counters(self) -> list[LifetimeCounter]:
        return [] if self.lifetime_counter_set is None else self.lifetime_counter_set.lifetime_counters
    
    @property
    def variables(self) ->list[BaseVariable]:
        ltc_variables: list[BaseVariable] = []
        for ltc in self.lifetime_counters:
            ltc_variables.append(ltc)
        return self._variables + ltc_variables + [] if self.component_set is None else remove_none([self.component_set.node_version])

    @property
    def name_plate_variables(self) ->list[BaseVariable]:
        return self._variables if self.identification is None else remove_none([self.device_health] + self.identification.variables)

# MARK: Device
class Device(Component):
    """
    Represents a device in the LADS OPC UA model.

    Attributes:
        device_state: StateMachine - State machine representing the device state.
        machinery_item_state: StateMachine - State machine representing the machinery item state.
        machinery_operation_mode: StateMachine - State machine representing the machinery operation mode.
        location: SubscribedVariable - Location of the device.
        hierarchical_location: SubscribedVariable - Hierarchical location of the device.
        operational_location: SubscribedVariable - Operational location of the device.
        state_machine_variables: list[BaseVariable] - List of state machine variables of the device.
        location_variables: list[BaseVariable] - List of location variables of the device.
        geographical_location: Tuple[float, float] - Geographical location of the device.
        unique_name: str - Unique name of the device.
        variables: list[BaseVariable] - List of variables of the device.
        events: list[Event] - List of events of the device.
    """

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Device, node, server.DeviceType, server)
    
    async def init(self, server: Server):
        self.device_state: StateMachine = None
        self.machinery_item_state: StateMachine = None
        self.machinery_operation_mode: StateMachine = None
        self.location: SubscribedVariable = None
        self.hierarchical_location: SubscribedVariable = None
        self.operational_location: SubscribedVariable = None
        self.state_machine_variables: list[BaseVariable] = []
        self.device_type_images = []
        self.compliance_document_set: LADSSet = None
        await super().init(server)            
                
        functional_unit_set = await self.get_lads_child("FunctionalUnitSet")
        nodes = await self.get_child_objects(functional_unit_set)
        self.functional_units: list[FunctionalUnit] = await asyncio.gather(*(FunctionalUnit.promote(node, server) for node in nodes))
        self.device_state, self.machinery_item_state, self.machinery_operation_mode, self.lock = await asyncio.gather(
            StateMachine.promote(await self.get_lads_child("DeviceState"), server),
            StateMachine.promote(await self.get_machinery_child("MachineryItemState"), server),
            StateMachine.promote(await self.get_machinery_child("MachineryOperationMode"), server),
            Lock.promote(await self.get_di_child("Lock"), server)
        )
        state_machines: list[StateMachine] = remove_none([self.device_state, self.machinery_item_state, self.machinery_operation_mode])
        self.state_machine_variables = list(map(lambda state_machine: state_machine.current_state, state_machines))
        if self.device_health is not None:
            self.state_machine_variables.append(self.device_health)

        # location
        self.hierarchical_location, self.operational_location = await asyncio.gather(
            self.get_amb_variable("HierarchicalLocation"),
            self.get_amb_variable("OperationalLocation"),
        )
        self.location = None if self.identification is None else self.identification.location
        for location in self.location_variables:
            location.subscription_level = SubscriptionLevel.Temporary
            
        # device type images
        self.device_type_images = []
        device_type_image = await self.get_di_child("DeviceTypeImage")
        if device_type_image is not None:
            nodes = await device_type_image.get_variables()
            for node in nodes:
                variable = await BaseVariable.promote(node, server)
                value = variable.value
                from sys import getsizeof
                if value != None and getsizeof(value) > 0:
                    self.device_type_images.append(value)
        
        # compliance documents
        if server.ns_LADS_CD is not None:
            self.compliance_document_set = await ComplianceDocumentSet.promote(await self.get_lads_cd_child("ComplianceDocumentSet"), self.server)
            if self.compliance_document_set is not None:
                _logger.debug("loading compliance documents")
                await self.compliance_document_set.promote_children(ComplianceDocument, self.server.ComplianceDocumentType, self.server.ComplianceDocumentSetType)
            else:
                _logger.debug("unable to find compliance document set")


    async def finalize_init(self):
        await super().finalize_init()
        await asyncio.gather(*(functional_unit.finalize_init(self) for functional_unit in self.functional_units))
        # prepare subscriptions
        variables = self.subscribed_variables + self.location_variables
        for functional_unit in self.functional_units:
            variables = variables + functional_unit.all_subscribed_variables
        if self.identification is not None:
            variables = variables + self.identification.subscribed_variables
        if self.lock is not None:
            variables = variables + self.lock.variables
        for component in self.components:
            variables = variables + component.subscribed_variables 
            if component.identification is not None:
                variables = variables + component.identification.subscribed_variables
            if component.operation_counters is not None:
                variables = variables + component.operation_counters.subscribed_variables
            for lifetime_counter in component.lifetime_counters:
                variables = variables + lifetime_counter.subscribed_variables
        if self.operation_counters is not None:
            variables = variables + self.operation_counters.subscribed_variables
        self.subscription_handler = SubscriptionHandler()
        variable_set = set(variables)
        _logger.debug(f"Device {self.display_name} subscribing to {len(variable_set)}/{len(variables)} variables")
        data_change_handlers = await self.subscription_handler.subscribe_data_change(self.server, variable_set)
        try:
            events_handler = await self.subscription_handler.subscribe_events(self.server, self)
        except:
            try:
                events_handler = await self.subscription_handler.subscribe_events(self.server, self.server.client.get_server_node())
            except:
                _logger.warning("Unable to subscribe to events")
        

    @property
    def location_variables(self) ->list[BaseVariable]:
        return remove_none([self.location, self.hierarchical_location, self.operational_location])

    @property
    def geographical_location(self) -> Tuple[float, float] | None:
        location = self.location if self.location is not None else self.operational_location
        # location = self.operational_location
        if location is not None:
            try:
                position = location.value_str
                l = position.split(" ")
                if len(l) == 4:
                    lon = float(l[1]) * (-1 if "S" in l[0].upper() else 1)
                    lat = float(l[3]) * (-1 if "W" in l[0].upper() else 1)
                    return (lon, lat)
            except:
                return None
        return None

    @property
    def unique_name(self) -> str:
        return f"{self.server.name}{unique_name_delimiter}{self.display_name}"
    
    @property
    def variables(self) ->list[BaseVariable]:
        document_set_node_version = self.compliance_document_set.node_version if self.compliance_document_set is not None else None
        _logger.debug(document_set_node_version)
        vars = self.name_plate_variables + self.state_machine_variables
        vars.append(self.compliance_document_set.node_version if self.compliance_document_set is not None else None)
        return remove_none(vars)
    
    @property
    def events(self) ->list[Event]:
        if self.subscription_handler is not None:
            return self.subscription_handler.event_list
        else:
            return []

# MARK: Lock
class Lock(LADSNode):
    INIT_LOCK = "InitLock"
    EXIT_LOCK = "ExitLock"
    BREAK_LOCK = "BreakLock"
    RENEW_LOCK = "RenewLock"
    COMMANDS = [INIT_LOCK, EXIT_LOCK, BREAK_LOCK, RENEW_LOCK]
    
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Lock, node, server.LockingServicesType, server)
    
    async def init(self, server: Server):
        await super().init(server)
        self.max_inactive_lock_time = server.max_inactive_lock_time
        (self.locked, self.locking_client, self.locking_user, self.remaining_lock_time) = await asyncio.gather(
            self.get_di_variable("Locked"),
            self.get_di_variable("LockingClient"),
            self.get_di_variable("LockingUser"),
            self.get_di_variable("RemainingLockTime"),
        )
        parent = await LADSNode.promote(await self.get_parent(), server)
        _logger.debug(f"Locking service of {parent.display_name} initialized")
        
    def init_lock(self, context:str = "LADS Cient"):
        self.call_async(self.call_di_method(Lock.INIT_LOCK, context))
        
    def exit_lock(self):
        self.call_async(self.call_di_method(Lock.EXIT_LOCK))
        
    def break_lock(self):
        self.call_async(self.call_di_method(Lock.BREAK_LOCK))

    def renew_lock(self):
        self.call_async(self.call_di_method(Lock.RENEW_LOCK))

    def call_method_by_name(self, name: str, *args):
        if name is None:
            return
        match name:
            case Lock.INIT_LOCK:
                self.init_lock()
            case Lock.EXIT_LOCK:
                self.exit_lock()
            case Lock.BREAK_LOCK:
                self.break_lock()
            case Lock.RENEW_LOCK:
                self.renew_lock()
        
    @property
    def variables(self) ->list[BaseVariable]:
        return remove_none([self.locked, self.locking_client, self.locking_user, self.remaining_lock_time])
      
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
        server = self.server
        types = await browse_types(server, child)
        try:
            function: Function = None
            if server.AnalogControlFunctionWithTotalizerType in types:
                function = await AnalogControlFunctionWithTotalizer.promote(child, server)
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

# MARK: VariableSet
class VariableSet(LADSSet):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(VariableSet, node, server.VariableSetType, server)
    
    async def init(self, server: Server):
        await super().init(server)
        await self.update_children()

    async def update_children(self):
        self._variables: list[BaseVariable] = []
        await self.collect_variables(self)

    async def collect_variables(self, lads_node: LADSNode, path: str = ""):
        # collect variables and properties of current node
        variables = await get_properties_and_variables(lads_node)
        for variable in variables:
            if not "NodeVersion" in variable.display_name:
                variable.alternate_display_name = unique_name_delimiter.join([path, variable.display_name])
                self._variables.append(variable)
        # recurse objects if any
        nodes = await self.get_child_objects(lads_node)
        for node in nodes:
            parent = await LADSNode.promote(node, self.server)
            await self.collect_variables(parent, unique_name_delimiter.join([path, parent.display_name]))

    @property
    def variables(self) -> list[BaseVariable]:
        return self._variables

# MARK: ResultFile
from asyncua.client.ua_file import UaFile
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

# MARK: class ComplianceDocumentSet(LADSSet):
class ComplianceDocumentSet(LADSSet):
    # since the Machinery type Components is not derived from LADS.SetType we need a different type check
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(ComplianceDocumentSet, node, server.ComplianceDocumentSetType, server)

# MARK: ComplianceDocument
class ComplianceDocument(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Result, node, server.ComplianceDocumentType, server)

    async def init(self, server: Server):
        self.applies_to = []
        self.references_markdown = []

        await super().init(server)        
        self.document_name = await self.get_lads_cd_variable("DocumentName")
        self.issued_at = await self.get_lads_cd_variable("IssuedAt")
        self.valid_from = await self.get_lads_cd_variable("ValidFrom")
        self.valid_until = await self.get_lads_cd_variable("ValidUntil")
        self.mime_type = await self.get_lads_cd_variable("MimeType")
        self.content = await self.get_lads_cd_variable("Content")
        self.schema_uri = await self.get_lads_cd_variable("SchemaUri")
        self.file = await self.get_lads_cd_child("File")
        ref_type: ua.ReferenceDescription = server.HasComplianceDocument
        references = await self.get_references_of_type(ref_type)
        for desc in references:
            node = server.get_node(desc.NodeId)
            type_node_id = await node.read_type_definition()
            type_node = server.get_node(type_node_id)
            ref_type_node = server.get_node(desc.ReferenceTypeId)
            self.applies_to.append(node)
            node_name, node_type_name, ref_type_name = await asyncio.gather(node.read_display_name(), type_node.read_display_name(), ref_type_node.read_display_name())
            markdown = f"**{node_name.Text}**: *{node_type_name.Text}* -> {ref_type_name.Text} -> **{self.display_name}**"
            self.references_markdown.append(markdown)

    async def download(self):
        try:
            self.downloading = True
            _logger.debug(f"{self.display_name} start downloading file data ..")
            async with UaFile(self.file, "r") as ua_file:
                self.data = await ua_file.read()
                _logger.debug(f"{self.display_name} finished downloading file data ..")
                self.downloading = False
        except:
            _logger.error(f"{self.display_name} failed reading file")
        
    def has_data(self) -> bool:
        return self.data is not None

    def fetch_data(self):
        if self.downloading:
            _logger.debug(f"{self.display_name} download already active ..")
        else:
            _logger.debug(f"{self.display_name} fetching file data ..")
            self.call_async(self.download())
    """
    # Since all information is static don't attach it to a subscription group
    @property
    def variables(self) ->list[BaseVariable]:
        return remove_none([self.document_name, self.issued_at, self.valid_from, self.valid_until, self.mime_type, self.content])
    """

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
        
        self.function_set, self.functional_unit_state, self.program_manager, self.lock = await asyncio.gather(
            FunctionSet.promote(await self.get_lads_child("FunctionSet"), server),
            FunctionalStateMachine.promote(await self.get_lads_child("FunctionalUnitState"), server),
            ProgramManager.promote(await self.get_lads_child("ProgramManager"), server),
            Lock.promote(await self.get_di_child("Lock"), server)
        )

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

# MARK: AlarmMonitor
class AlarmMonitor(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(AlarmMonitor, node, server.ExclusiveLimitAlarmType, server)

    async def init(self, server: Server):
        await super().init(server)        
        
        self.alarm_active_state, self.high_high_limit, self.high_limit, self.low_limit, self.low_low_limit = await asyncio.gather(
            StateVariable.promote(await self.get_child("ActiveState"), server),
            SubscribedVariable.promote(await self.get_child("HighHighLimit"), server),
            SubscribedVariable.promote(await self.get_child("HighLimit"), server),
            SubscribedVariable.promote(await self.get_child("LowLimit"), server),
            SubscribedVariable.promote(await self.get_child("LowLowLimit"), server),
        )
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
    
# MARK: CoverFunction
class CoverFunction(BaseStateMachineFunction):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(CoverFunction, node, server.CoverFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.cover_state = await FunctionalStateMachine.promote(await self.get_lads_child("CoverState"), server)

    @property
    def state_machine(self) -> StateMachine:
        return self.cover_state

#MARK: Promotion of generic OPC UA nodes to specfic LADS objects
async def promote_to(cls: Type, node: Node, super_type_node: Node, server: Server) -> LADSNode:
    if node is None: return None
    node_class = await node.read_node_class()
    if node_class != ua.NodeClass.Method:
        type = await node.read_type_definition()
        type_node = server.client.get_node(type)
        result = await is_subtype(type_node, super_type_node.nodeid)
        assert result, f"node {node.nodeid} is expexted to be of type {type_node.nodeid}"
    node.__class__ = cls
    # promoted_node : cls = node
    promoted_node = node
    await promoted_node.init(server)
    return promoted_node

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

# MARK: Connection
class Connection:
    """
    Connection class for managing a LADS OPC UA client-server connection.

    Attributes:
        client (Client): The LADS OPC UA client instance.
        server (Server): The LADS OPC UA server instance.
        url (str): The URL of the LADS OPC UA server.
        user (str): The username for authentication.
        password (str): The password for authentication.
        thread (threading.Thread): The thread responsible for running the connection.

    Methods:
        initialized:
            Checks if the server is initialized.
        connect():
            Starts the connection thread (non-asynchronous) and waits until the server is initialized.
        disconnect():
            Disconnects the server (non-asynchronous).
    """

    data_types: dict = None

    def __init__(self, url = None, user: str = None, password: str = None) -> None:
        """
        Prepares the connection to a LADS OPC UA server. 

        Args:
            url (str): The URL of the LADS OPC UA server. Defaults to None.
            user (str, optional): The username for authentication. Defaults to None.
            password (str, optional): The password for authentication. Defaults to None.
        """

        self.client: Client = None
        self.server: Server = None
        self.url = url
        self.user = user
        self.password = password
        self.running = False
        self.thread = threading.Thread(target=self._run_connection, daemon=True, name=f"LADS OPC UA Connection {url}")
        #self.thread.start()

    @property
    def initialized(self) -> bool:
        """Checks if the server is initialized."""
        if self.server is None:
            return False
        else:
            return self.server.initialized
    
    def _run_connection(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_connection_async())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def connect(self):
        """Starts the connection thread (non-asynchronous)."""
        if self.thread.is_alive():
            return

        if not self.thread.is_alive() and self.thread._started.is_set():
            self.thread = threading.Thread(
                target=self._run_connection, daemon=True, name=f"LADS OPC UA Connection {self.url}")

        self.running = True
        self.thread.start()
    
    def disconnect(self):
        """Disconnects the server and stops the thread (non-asynchronous)."""
        if self.server is not None:
            if self.server.running:
                print(f"Disconnecting from {self.server.name}...", end="")
                self.running = False
                self.thread.join()
                while self.server.initialized:
                    time.sleep(0.1)
                print(f"Done!")
    
    async def _run_connection_async(self):
        while self.running:
            self.client = Client(self.url)
            self.server = Server(self.client)
            if (self.user is not None) and (self.password is not None):
                self.client.set_user(self.user)
                self.client.set_password(self.password)
            try:
                async with self.client:
                    await self.server.init()
                    while self.server.running and self.running:
                        await self.server.evaluate()
                        await self.client.check_connection()
                    if self.server.running:
                        self.server.running = False
                        self.server.initialized = False
            except (TimeoutError, ConnectionError, ua.UaError) as error:
                _logger.warning(f"Reconnecting in 2 seconds: {error}")
                await asyncio.sleep(2)
            #except Exception as error:
                # _logger.error(error)
                
def get_value(data: dict, key: str) -> any:
    if key in data:
        return data[key]
    else:
        return None

# MARK: Connections
class Connections:
    """
    Connections class for managing multiple LADS OPC UA client-server connections.

    Attributes:
        connections (list[Connection]): The list of connections.
        urls (list[str]): The list of URLs of the connections.
        initialized (bool): Checks if all servers are initialized.
        functional_units (list[FunctionalUnit]): The list of functional units from all connections.
    
    Methods:
        add(url, user, password):
            Adds a new connection with the given parameters.
        connect():
            Starts all connection threads (non-asynchronous).
        disconnect():
            Disconnects from all server connections (non-asynchronous).
    """

    connections: list[Connection] = []

    def __init__(self, config_file = "config.json") -> None:
        """
        Initializes the connections with the given configuration file.

        Args:
            config_file (str): The path to the configuration file. Defaults to "config.json".
        """

        try:
            with open(config_file) as f:
                print(f"parsing config file {config_file}")
                data = json.load(f)
                for connection in data["connections"]:
                    url = connection["url"]
                    user = get_value(connection, "user") 
                    password = get_value(connection, "password")
                    enabled = get_value(connection, "enabled") if not None else True
                    if enabled:
                        print(f"Add conncetion with url {url}")
                        self.add(url, user, password)
        except Exception as error:
            _logger.error(f"Invalid config file {config_file}: {error}")

    def add(self, url: str, user: str = None, password: str = None) -> Connection:
        """
        Adds a new connection with the given parameters.

        Args:
            url (str): The URL of the OPC UA server.
            user (str, optional): The username for authentication. Defaults to None.
            password (str, optional): The password for authentication. Defaults to None.
        """

        connection = Connection(url, user, password)
        self.connections.append(connection)
        return connection
    
    def connect(self):
        """Starts the connection threads (non-asynchronous)."""
        for connection in self.connections:
            connection.connect()
    
    def disconnect(self):
        """Disconnects the server/s (non-asynchronous)."""
        for connection in self.connections:
            if connection.server is not None:
                connection.disconnect()

    @property
    def urls(self) -> list[str]:
        return list(map(lambda connection: connection.url, self.connections))

    @property
    def initialized(self) -> bool:
        if len(self.connections) == 0:
            return False
        result = True
        for connection in self.connections:
            result = result and connection.initialized
        return result
    
    @property
    def functional_units(self) -> list[FunctionalUnit]:
        result: list[FunctionalUnit] = []
        if self.initialized:
            for connection in self.connections:
                result.extend(connection.server.functional_units)
        return result