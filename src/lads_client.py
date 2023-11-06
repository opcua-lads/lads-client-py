import asyncio
import logging
import pandas as pd
import datetime as dt
from typing import Type, NewType, Any, Self, Tuple, Set
from asyncua import Client, ua, Node
from asyncua.common.subscription import DataChangeNotif
from asyncua.common.events import Event
from enum import IntEnum
from queue import Queue

_logger = logging.getLogger(__name__)

# BaseVariable = NewType("BaseVariable", LADSNode)
# NodeVersionVariable = NewType("NodeVersionVariable", BaseVariable)
# AnalogItem = NewType("AnalogItem", BaseVariable)
# LifetimeCounter = NewType("LifetimeCounter", AnalogItem)
LADSNode = NewType("LADSNode", Node)
#LADSSet = NewType("LADSSet", LADSNode)
BaseVariable = NewType("BaseVariable", LADSNode)
Method = NewType("Method", LADSNode)
Component = NewType("Component", LADSNode)
Device = NewType("Device", Component)
FunctionalUnit = NewType("FunctionalUnit", LADSNode)
Function = NewType("Function", LADSNode)
      
class LADSObjectIds(IntEnum):
    DeviceType = 1002
    ComponentSetType = 1025
    ComponentType = 1024
    SetType = 61
    FunctionalUnitSetType = 1023
    FunctionalUnitType= 1003
    FunctionSetType = 1026
    FunctionType= 1004
    AnalogSensorFunctionType = 1016
    AnalogArraySensorFunctionType = 1015
    AnalogControlFunctionType = 1009
    AnalogControlFunctionWithTotalizerType = 1014
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

class MachineryObjectIds(IntEnum):
    MachineryOperationCounterType = 1009
    MachineryLifeTimeCounterType = 1015

class DIObjectIds(IntEnum):
    LifetimeVariableType = 468

class SubscriptionLevel(IntEnum):
    Never = 0
    Temporary = 1
    Permanent = 2

class Server():
    def __init__(self, client: Client, name: str) -> None:
        self.client = client
        self.name = name
        self.server = self
        self.devices: list[Device] = []
        self.initialized = False
        self.running = True
        self.call_async_queue = Queue()

    async def init(self):
        # read namespace indices
        self.ns_DI = await self.client.get_namespace_index("http://opcfoundation.org/UA/DI/")
        self.ns_AMB = await self.client.get_namespace_index("http://opcfoundation.org/UA/AMB/")
        self.ns_Machinery = await self.client.get_namespace_index("http://opcfoundation.org/UA/Machinery/")
        self.ns_LADS = await self.client.get_namespace_index("http://opcfoundation.org/UA/LADS/")

        # get well known type nodes
        self.BaseObjectType = self.client.get_node(ua.ObjectIds.BaseObjectType)
        self.FiniteStateMachineType = self.client.get_node(ua.ObjectIds.FiniteStateMachineType)
        self.BaseVariableType = self.client.get_node(ua.ObjectIds.BaseVariableType)
        self.AnalogItemType = self.client.get_node(ua.ObjectIds.AnalogItemType)
        self.TwoStateDiscreteType = self.client.get_node(ua.ObjectIds.TwoStateDiscreteType)
        self.MultiStateDiscreteType = self.client.get_node(ua.ObjectIds.MultiStateDiscreteType)
        self.LifetimeVariableType = self.get_di_node(DIObjectIds.LifetimeVariableType)
        self.MachineryOperationCounterType = self.get_machinery_node(MachineryObjectIds.MachineryOperationCounterType)
        self.MachineryLifeTimeCounterType = self.get_machinery_node(MachineryObjectIds.MachineryLifeTimeCounterType)
        self.DeviceType = self.get_lads_node(LADSObjectIds.DeviceType)
        self.SetType = self.get_lads_node(LADSObjectIds.SetType)
        self.ComponentSetType = self.get_lads_node(LADSObjectIds.ComponentSetType)
        self.ComponentType = self.get_lads_node(LADSObjectIds.ComponentType)
        self.FunctionalUnitSetType = self.get_lads_node(LADSObjectIds.FunctionalUnitSetType)
        self.FunctionalUnitType = self.get_lads_node(LADSObjectIds.FunctionalUnitType)
        self.FunctionSetType = self.get_lads_node(LADSObjectIds.FunctionSetType)
        self.FunctionType = self.get_lads_node(LADSObjectIds.FunctionType)
        self.AnalogSensorFunctionType = self.get_lads_node(LADSObjectIds.AnalogSensorFunctionType)
        self.AnalogArraySensorFunctionType = self.get_lads_node(LADSObjectIds.AnalogArraySensorFunctionType)
        self.AnalogControlFunctionType = self.get_lads_node(LADSObjectIds.AnalogControlFunctionType)
        self.AnalogControlFunctionWithTotalizerType = self.get_lads_node(LADSObjectIds.AnalogControlFunctionWithTotalizerType)
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

        # read data tyoes
        self.data_types = await self.client.load_data_type_definitions()

        # browse for devices in DeviceSet
        device_set = await self.client.nodes.objects.get_child(f"{self.ns_DI}:DeviceSet")
        nodes = await device_set.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        for node in nodes:
            device: Device = await Device.propagate(node, self)
            await device.finalize_init()
            self.devices.append(device)
        self.initialized = True

    async def evaluate(self):
        if not self.call_async_queue.empty():
            item = self.call_async_queue.get()
            if item is not None:
                try:
                    await item
                except Exception as error:
                    _logger.debug(error)
        await asyncio.sleep(0.01)

    def get_di_node(self, id: int) -> Node | None:
        return self.client.get_node(ua.NodeId(id, self.ns_DI))

    def get_machinery_node(self, id: int) -> Node | None:
        return self.client.get_node(ua.NodeId(id, self.ns_Machinery))

    def get_lads_node(self, id: int) -> Node | None:
        return self.client.get_node(ua.NodeId(id, self.ns_LADS))
    
    @property
    def functional_units(self) -> list[FunctionalUnit]:
        if not self.initialized: return []
        functional_units: list[FunctionalUnit] = []
        for device in self.devices:
            functional_units = functional_units + device.functional_units
        return functional_units
    
async def get_parent_nodes(server: Server, node: Node, root_node: Node = None) -> list[Node]:
    parent = await node.get_parent()
    if parent == root_node:
        return [parent]
    else:
        return [parent] + await get_parent_nodes(server, parent, root_node)

async def browse_types(server: Server, node: Node) -> list[Node]:
    type_node_id =  await node.read_type_definition()
    type_node = Node(node.session, type_node_id)

    root_node = server.BaseObjectType
    node_class = await node.read_node_class()
    if node_class == ua.NodeClass.Variable:
        root_node = server.BaseVariableType
    else:
        root_node = server.BaseObjectType
    return [type_node] + await get_parent_nodes(server, type_node, root_node)

async def is_of_type(server: Server, node: Node, type_node: Node) -> bool:
    types = await browse_types(server, node)
    return type_node in types

unique_name_delimiter = "/"

def variant_value_to_str(variant: ua.Variant) -> str:
    if variant is None:
        return "result"
    value = variant.Value
    if isinstance(value,ua.LocalizedText):
        return value.Text if value.Text is not None else ""
    elif isinstance(value, ua.QualifiedName):
        return  value.Name
    elif isinstance(value, dt.datetime):
        return  value.strftime("%d.%m.%Y %H:%M:%S")
    else:
        return str(value)

def remove_none(nodes: list[Node]) -> list[Node]:
    return list(filter(lambda node: node is not None, nodes))

class SubscriptionHandler(object):

    def __init__(self) -> None:
        super().__init__()
        self.subscription = None
        self.subscribed_variables = {}
        self.event_node = None
        self.events: pd.DataFrame = None
        self.last_event_update = dt.datetime.now()

    async def subscribe_data_change(self, server: Server, nodes: list[BaseVariable], period: float = 500):
        if len(nodes) == 0: 
            return
        if self.subscription is None:
            self.subscription = await server.client.create_subscription(period, self)
        self.subscribed_variables = dict((node.nodeid, node) for node in nodes)
        result = await self.subscription.subscribe_data_change(nodes) 
        return result
 
    async def subscribe_events(self, server: Server, node: Node, period: float = 500):
        if self.subscription is None:
            self.subscription = await server.client.create_subscription(period, self)
        self.event_node: LADSNode = node
        return await self.subscription.subscribe_events(node)        
 
    def datachange_notification(self, node: Node, val: Any, data: DataChangeNotif):
        try:
            variable: Node = self.subscribed_variables[node.nodeid]
            variable.data_change_notification(data)
        except Exception as error:
            _logger.error(f"datachange_notification error {error}")

    def event_notification(self, event: Event):
        fields_dict = event.get_event_props_as_fields_dict()
        event_fields = {}
        try:
            event_fields = {k: variant_value_to_str(v) for k, v in fields_dict.items()}
        except Exception as error:
            _logger.error(error)
        key = pd.to_datetime(dt.datetime.now())
        if self.events is None:
            self.events = pd.DataFrame(event_fields, index = [key])
        else:
            self.events.loc[key] = event_fields
            if len(self.events.index) > 1000:
                self.events = self.events.tail(-10)
        self.last_event_update = key

    def status_change_notification(self, status: Any):
        print(status)

class LADSNode(Node):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(LADSNode, node, server.BaseObjectType, server)

    alternate_display_name: str = None

    async def init(self, server: Server):
        self.server: Server = server
        (self.browse_name, self._display_name, self.description)  = await asyncio.gather(
            self.read_browse_name(),
            self.read_display_name(),
            self.read_description()
        )

    async def finalize_init(self):
        pass

    #def datachange_notification(self, node: Node, val: Any, data: DataChangeNotif):
    #    pass

    #def event_notification(self, event):
    #    print("New event", event)

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
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.display_name})"
    
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
        return await BaseVariable.propagate(await self.get_di_child(name), self.server)
    
    async def get_machinery_child(self, name : str) -> Node:
        return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_Machinery))
    
    async def get_machinery_variable(self, name : str) -> BaseVariable:
        return await BaseVariable.propagate(await self.get_machinery_child(name), self.server)
    
    async def get_lads_child(self, name : str) -> Node:
        return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_LADS))
    
    async def get_lads_variable(self, name : str) -> BaseVariable:
        return await BaseVariable.propagate(await self.get_lads_child(name), self.server)
    
    async def get_child_objects(self, parent: Node = None) -> list[Node]:
        if parent is None: parent = self
        # search for HasChild and Organizes references
        (has_child_objects, organizes_objects) = await asyncio.gather(
            parent.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object),
            parent.get_children(refs = ua.ObjectIds.Organizes, nodeclassmask = ua.NodeClass.Object)
        )
        # reduce results to set
        child_objects = set(has_child_objects)
        child_objects.update(organizes_objects)
        return list(child_objects)
    
    async def call_lads_method(self, name: str, *args: Any) -> ua.StatusCode:
        try:
            return await self.call_method(ua.QualifiedName(name, self.server.ns_LADS), *args)
        except Exception as error:
            _logger.error(error)
            return ua.StatusCodes.BadNotImplemented
        
class Method(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(Method, node, None, server)

class BaseVariable(LADSNode):
    subscription_level = SubscriptionLevel.Never
    data_value: ua.DataValue
    data_type: ua.VariantType
    access_level: Set[ua.AccessLevel]
    history: pd.DataFrame

    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(BaseVariable, node, server.BaseVariableType, server)

    def __str__(self):
        return f"{super().__str__()} = {self.value}"
    
    async def init(self, server: Server):
        await super().init(server)
        (self.data_value, self.data_type, self.access_level, historizing) = await asyncio.gather(
            self.read_data_value(raise_on_bad_status=False),
            self.read_data_type_as_variant_type(),
            self.get_access_level(),
            self.read_attribute(ua.AttributeIds.Historizing)
        )
        self.history = None
        if (historizing.Value.Value):
            self.subscription_level = SubscriptionLevel.Permanent
            self.history = pd.DataFrame({f"{self.display_name}": [self.value]}, index = [pd.to_datetime(self.data_value.SourceTimestamp)])

    def set_value(self, value: Any) -> ua.StatusCode:
        if self.has_write_access:
            self.server.call_async_queue.put(self.write_value(value, self.data_type))
            return ua.StatusCodes.Uncertain
        else:
            return ua.StatusCodes.BadNotWritable

    async def set_value_async(self, value: Any) -> ua.StatusCode:
        if self.has_write_access:
            result = ua.StatusCodes.Good
            try:
                await self.write_value(value, self.data_type)
            except:
                result = ua.StatusCodes.BadInvalidArgument
            return result
        else:
            return ua.StatusCodes.BadNotWritable

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
            return variant_value_to_str(self.data_value.Value)
        else:
            return ""
            
    def data_change_notification(self, data: DataChangeNotif):
        self.data_value = data.monitored_item.Value
        if self.history is not None:
            self.history.loc[pd.to_datetime(self.data_value.SourceTimestamp)] = self.value
            if len(self.history.index) > 600:
                self.history = self.history.tail(-1)

class SubscribedVariable(BaseVariable):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(SubscribedVariable, node, server.BaseVariableType, server)
    
    async def init(self, server: Server):
        await super().init(server)
        if self.history is None:
            self.subscription_level = SubscriptionLevel.Temporary

class NodeVersionVariable(SubscribedVariable):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(NodeVersionVariable, node, server.BaseVariableType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.subscription_level = SubscriptionLevel.Permanent
        self.set: LADSSet = None

    def data_change_notification(self, data: DataChangeNotif):
        super().data_change_notification(data)
        if self.set is None: return
        self.set.node_version_changed()

class StateVariable(SubscribedVariable):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        variable: StateVariable = await propagate_to(StateVariable, node, server.BaseVariableType, server)
        variable.subscription_level = SubscriptionLevel.Permanent
        return await propagate_to(StateVariable, node, server.BaseVariableType, server)

    @property
    def value_str(self) -> str:
        s =  super().value_str
        l = s.split(":")
        return s if len(l) < 2 else l[1]

class AnalogItem(SubscribedVariable):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(AnalogItem, node, server.AnalogItemType, server)

    def __str__(self):
        return f"{super().__str__()} [{self.eu}]"
    
    async def init(self, server: Server):
        await super().init(server)
        self.engineering_units: ua.EUInformation = None
        self.eu_range: ua.Range = None
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
    
    @property
    def eu(self) -> str:
        result = ""
        if self.engineering_units is not None:
            if isinstance(self.engineering_units, ua.EUInformation):
                result = self.engineering_units.DisplayName.Text
        return result

class TwoStateDiscrete(SubscribedVariable):
    true_state: BaseVariable
    false_state: BaseVariable

    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(TwoStateDiscrete, node, server.TwoStateDiscreteType, server)

    def __str__(self):
        return f"{super().__str__()}\n  TrueState: {self.true_state.value_str}\n  FalseState: {self.false_state.value_str}"
    
    async def init(self, server: Server):
        await super().init(server)
        self.true_state = await BaseVariable.propagate(await self.get_child("TrueState"), server)
        self.false_state = await BaseVariable.propagate(await self.get_child("FalseState"), server)

    @property
    def value_str(self) -> str:
        if bool(self.value):
            return self.true_state.value_str
        else:
            return self.false_state.value_str
    
class MultiStateDiscrete(SubscribedVariable):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(MultiStateDiscrete, node, server.MultiStateDiscreteType, server)

    def __str__(self):
        value: list[str] = self.enum_strings.data_value.Value.Value
        s = ",".join(value)
        return f"{super().__str__()}\n  [{s}]"
    
    async def init(self, server: Server):
        await super().init(server)
        self.enum_strings = await BaseVariable.propagate(await self.get_child("EnumStrings"), server)
        assert(self.enum_strings.data_value.Value.is_array)

    @property
    def value_str(self) -> list[str]:
        s = self.values
        i = int(self.value)
        if i in range(len(s)):
            return s[i]
        else:
            "unknown"
    
    @property
    def values(self) -> list[str]:
        return self.enum_strings.data_value.Value.Value

class LifetimeCounter(AnalogItem):
    limit_value: BaseVariable
    start_value: BaseVariable
    warning_values: BaseVariable

    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(LifetimeCounter, node, server.LifetimeVariableType, server)

    def __str__(self):
        return f"{super().__str__()}\n  {self.limit_value}\n  {self.start_value}"

    async def init(self, server: Server):
        await super().init(server)
        self.limit_value, self.start_value, self.warning_values = await asyncio.gather(
            self.get_di_variable("LimitValue"),
            self.get_di_variable("StartValue"),
            self.get_di_variable("WarningValues"),
        )

class StateMachine(LADSNode):
    methods: list[Method] = []
    methods_dict: dict[str, Method]
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(StateMachine, node, server.FiniteStateMachineType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.current_state = await StateVariable.propagate(await self.get_child("CurrentState"), server)
        self.current_state.alternate_display_name = self.display_name
        nodes = await self.get_methods()
        self.methods = await asyncio.gather(*(Method.propagate(node, server) for node in nodes))
        self.methods_dict = {method.display_name: method for method in self.methods}
    
    @property
    def method_names(self) -> list[str]:
        return self.methods_dict.keys()
    
    def call_method_by_name(self, name: str, *args):
        try:
            method = self.methods_dict[name]
            if method is not None:
                self.server.call_async_queue.put(self.call_method(method.nodeid, *args))
        except:
            _logger.debug(f"Unknwon method {name}")

    @property
    def variables(self) -> list[BaseVariable]:
        return super().variables + [self.current_state]

class FunctionalStateMachine(StateMachine):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(FunctionalStateMachine, node, server.FiniteStateMachineType, server)
    
    def start_program(self, program_template: str, properties: pd.DataFrame, supervisory_job_id: str, supervisory_task_id: str, samples: pd.DataFrame):
        key_value_list = None
        for index, row in properties.iterrows():
            key_value_cls = self.server.data_types["KeyValueType"]
            key_value = key_value_cls(
                str(row["Key"]), 
                str(row["Value"])
            )
            if key_value_list is None:
                key_value_list = []
            key_value_list.append(key_value)
        sample_info_list = None
        for index, row in samples.iterrows():
            sample_info_cls = self.server.data_types["SampleInfoType"]
            sample_info = sample_info_cls(
                str(row["ContainerId"]),
                str(row["SampleId"]),
                str(row["Position"]),
                str(row["CustomData"]),
            )
            if sample_info_list is None:
                sample_info_list = []
            sample_info_list.append(sample_info)
        self.call_async(self.call_lads_method("StartProgram", 
                                              program_template, 
                                              key_value_list, 
                                              supervisory_job_id, 
                                              supervisory_task_id, 
                                              sample_info_list))
            
    def start(self):
        self.call_async(self.call_lads_method("Start"))

    async def stop(self)-> ua.StatusCode:
        self.call_async(self.call_lads_method("Stop"))

class LADSSet(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(LADSSet, node, server.SetType, server)
    
    node_version: NodeVersionVariable
    children: list[Node] = []
    child_class: Type
    child_type: Node

    async def init(self, server: Server):
        await super().init(server)
        try:
            # node_vesion variable is optional
            node_version = await self.get_child("NodeVersion")
            if node_version is not None:
                self.node_version = await NodeVersionVariable.propagate(node_version, server)
                self.node_version.set = self
        except Exception as error:
            _logger.warning("LADSSet", error)
        finally:
            self.children = await self.get_child_objects()

    async def propagate_children(self, child_class: Type, child_type: Node, set_type: Node):
        if self.children is None: 
            return
        if set_type is not None:
            assert(await is_of_type(self.server, self, set_type))
        self.child_class = child_class
        self.child_type = child_type
        self.children = await asyncio.gather(*(self.propagate_child(child) for child in self.children))
        self.children.sort(key = lambda child: child.display_name)

    async def propagate_child(self, child: Node) -> LADSNode:
        return await propagate_to(self.child_class, child, self.child_type, self.server)
    
    @property
    def variables(self) ->list[BaseVariable]:
        return [] if self.node_version is None else [self.node_version]
    
    def node_version_changed(self):
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
                node = await self.propagate_child(nodes[0])
                self.children.append(node)
        if len(deleted_node_ids) > 0:
            for node_id in deleted_node_ids:
                nodes = list(filter(lambda node: node.nodeid == node_id, previous_nodes))
                assert(len(nodes) == 1)
                node = nodes[0]
                self.children.remove(node)
        
class ComponentSet(LADSSet):
    # since the Machinery type Components is not derived from LADS.SetType with meed a different type check
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(ComponentSet, node, server.ComponentSetType, server)
    
class OperationCounters(LADSNode):
    operation_cycle_counter: BaseVariable
    operation_duration: BaseVariable
    power_on_duration: BaseVariable

    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(OperationCounters, node, server.MachineryOperationCounterType, server)
    
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

class LifetimeCounters(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(LifetimeCounters, node, server.MachineryLifeTimeCounterType, server)
        
    async def init(self, server: Server):
        await super().init(server)
        nodes = await get_properties_and_variables(self)
        self.lifetime_counters: list[LifetimeCounter] = await asyncio.gather(*(LifetimeCounter.propagate(node, server) for node in nodes))
        self.lifetime_counters.sort(key = lambda node: node.display_name)

    @property
    def variables(self) -> list[Node]:
        return super().variables + self.lifetime_counters

class Component(LADSNode):
    component_set: LADSSet
    operation_counters: OperationCounters
    lifetime_counter_set: LifetimeCounters

    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(Component, node, server.ComponentType, server)
    
    async def init(self, server: Server):
        await super().init(server)
        self._variables = await get_properties_and_variables(self)
        self._variables.sort(key = lambda variable: variable.display_name)
        self.component_set = await ComponentSet.propagate(await self.get_machinery_child("Components"), server)
        if self.component_set is not None:
            await self.component_set.propagate_children(Component, server.ComponentType, server.ComponentSetType)
        self.operation_counters = await OperationCounters.propagate(await self.get_di_child("OperationCounters"), server)
        self.lifetime_counter_set = await LifetimeCounters.propagate(await self.get_machinery_child("LifetimeCounters"), server)

    @property
    def components(self) -> list[Component]:
        return [] if self.component_set is None else self.component_set.children
        
    @property
    def lifetime_counters(self) -> list[LifetimeCounter]:
        return [] if self.lifetime_counter_set is None else self.lifetime_counter_set.lifetime_counters
    
    @property
    def variables(self) ->list[BaseVariable]:
        return self._variables +  remove_none([self.component_set.node_version])

    def variable_named(self, name: str) -> BaseVariable:
        for variable in self.variables:
            if name == variable.browse_name.Name:
                return variable
        return None
    
    @property
    def name_plate_variables(self) ->list[BaseVariable]:
        return self._variables

class Device(Component):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(Device, node, server.DeviceType, server)
    
    state_machine: StateMachine
    machinery_item_state: StateMachine
    machinery_operation_mode: StateMachine
    device_health: SubscribedVariable
    location: SubscribedVariable
    hierarchical_location: SubscribedVariable
    operational_location: SubscribedVariable

    async def init(self, server: Server):
        await super().init(server)
        functional_unit_set = await self.get_lads_child("FunctionalUnitSet")
        nodes = await self.get_child_objects(functional_unit_set)
        self.functional_units: list[FunctionalUnit] = await asyncio.gather(*(FunctionalUnit.propagate(node, server) for node in nodes))
        self.state_machine, self.machinery_item_state, self.machinery_operation_mode, self.device_health = await asyncio.gather(
            StateMachine.propagate(await self.get_lads_child("StateMachine"), server),
            StateMachine.propagate(await self.get_machinery_child("MachineryItemState"), server),
            StateMachine.propagate(await self.get_machinery_child("MachineryOperationMode"), server),
            SubscribedVariable.propagate(await self.get_di_child("DeviceHealth"), server)
        )
        state_machines: list[StateMachine] = remove_none([self.state_machine, self.machinery_item_state, self.machinery_operation_mode])
        self.state_machine_variables = list(map(lambda state_machine: state_machine.current_state, state_machines))
        if self.device_health is not None:
            self.state_machine_variables.append(self.device_health)

        self.hierarchical_location = self.variable_named("HierarchicalLocation")
        self.operational_location = self.variable_named("OperationalLocation")
        self.location = self.variable_named("Location")
        for location in self.location_variables:
            location.subscription_level = SubscriptionLevel.Temporary

    async def finalize_init(self):
        await super().finalize_init()
        await asyncio.gather(*(functional_unit.finalize_init(self) for functional_unit in self.functional_units))
        # prepare subscriptions
        variables = self.subscribed_variables
        for functional_unit in self.functional_units:
            variables = variables + functional_unit.all_subscribed_variables
        self.subscription_handler = SubscriptionHandler()
        data_change_handlers = await self.subscription_handler.subscribe_data_change(self.server, variables)
        events_handler = await self.subscription_handler.subscribe_events(self.server, self)

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
        return self.name_plate_variables + self.state_machine_variables
    
    @property
    def events(self) ->list[Event]:
        if self.subscription_handler is not None:
            return self.subscription_handler.event_list
        else:
            return []

class FunctionSet(LADSSet):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(FunctionSet, node, server.FunctionSetType, server)
    
    async def init(self, server: Server):
        await super().init(server)
        self.functions: list[Function] = await asyncio.gather(*(self.propagate_child(child) for child in self.children))
        self.functions.sort(key = lambda function: function.display_name)

    async def propagate_child(self, child: Node) -> Function:
        server = self.server
        types = await browse_types(server, child)
        try:
            function: Function = None
            if server.AnalogControlFunctionWithTotalizerType in types:
                function = await AnalogControlFunctionWithTotalizer.propagate(child, server)
            elif server.AnalogControlFunctionType in types:
                function = await AnalogControlFunction.propagate(child, server)
            elif server.AnalogSensorFunctionType in types:
                function = await AnalogSensorFunction.propagate(child, server)
            elif server.AnalogArraySensorFunctionType in types:
                function = await AnalogArraySensorFunction.propagate(child, server)
            elif server.CoverFunctionType in types:
                function = await CoverFunction.propagate(child, server)
            elif server.StartStopControlFunctionType in types:
                function = await StartStopControlFunction.propagate(child, server)
            elif server.TwoStateDiscreteControlFunctionType in types:
                function = await TwoStateDiscreteControlFunction.propagate(child, server)
            elif server.MultiStateDiscreteControlFunctionType in types:
                function = await MultiStateDiscreteControlFunction.propagate(child, server)
            elif server.MultiModeControlFunctionType in types:
                function = await MulitModeControlFunction.propagate(child, server)
            else:
                function = await Function.propagate(child, server)
        except Exception as error:
            _logger.error(error)
        return function

    @property
    def all_variables(self) -> list[BaseVariable]:        
        variables = self.variables
        for function in self.functions:
            variables = variables + function.all_variables
        return variables

class ProgramTemplate(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(ProgramTemplate, node, server.ProgramTemplateType, server)

    async def init(self, server: Server):
        await super().init(server)
        self._variables = await get_properties_and_variables(self)
        self._variables.sort(key = lambda variable: variable.display_name)

    @property
    def variables(self) ->list[BaseVariable]:
        return self._variables

class Result(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(Result, node, server.ResultType, server)

    async def init(self, server: Server):
        await super().init(server)
        self._variables = await get_properties_and_variables(self)
        self._variables.sort(key = lambda variable: variable.display_name)

    @property
    def variables(self) ->list[BaseVariable]:
        return self._variables

class ActiveProgram(LADSNode):
    current_program_template: BaseVariable
    current_runtime: BaseVariable
    current_pause_time: BaseVariable
    current_step_name: BaseVariable
    curent_step_number: BaseVariable
    current_step_runtime: BaseVariable
    estimated_runtime: BaseVariable
    estimated_step_numbers: BaseVariable
    estimated_step_runtime: BaseVariable
    device_program_run_id: BaseVariable
    _variables: list[BaseVariable]

    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(ActiveProgram, node, server.ActiveProgramType, server)

    def find_variable(self, name: str) -> BaseVariable:
        if self._variables is None:
            return None
        match = list(filter(lambda variable: name in variable.browse_name.Name , self._variables))
        return None if len(match) == 0 else match[0]

    async def init(self, server: Server):
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
    def has_progress(self) -> bool:
        return not (self.current_runtime is None or self.estimated_runtime is None)
    
    @property
    def current_progress(self) -> float:
        try:
            return self.current_runtime.value / self.estimated_runtime.value
        except:
            return 0.0

    @property
    def has_step_progress(self) -> bool:
        return not (self.current_step_runtime is None or self.estimated_step_runtime is None)
    
    @property
    def current_step_progress(self) -> float:
        try:
            return self.current_step_runtime.value / self.estimated_step_runtime.value
        except:
            return 0.0

class ProgramManager(LADSNode):
    program_template_set: LADSSet
    result_set: LADSSet
    active_program: ActiveProgram

    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(ProgramManager, node, server.ProgramManagerType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.program_template_set = await LADSSet.propagate(await self.get_lads_child("ProgramTemplateSet"), server)
        self.result_set = await LADSSet.propagate(await self.get_lads_child("ResultSet"), server)
        await self.program_template_set.propagate_children(ProgramTemplate, server.ProgramTemplateType, server.ProgramTemplateSetType)
        await self.result_set.propagate_children(Result, server.ResultType, server.ResultSetType)
        self.active_program = await ActiveProgram.propagate(await self.get_lads_child("ActiveProgram"), server)

    @property
    def variables(self) ->list[BaseVariable]:
        return self.active_program.variables + [self.program_template_set.node_version, self.result_set.node_version]
    
    @property
    def program_templates(self) -> list[ProgramTemplate]:
        return self.program_template_set.children

    @property
    def program_template_names(self) -> list[str]:
        return list(map(lambda template: template.display_name, self.program_templates))
    
    @property
    def results(self) -> list[Result]:
        return self.result_set.children
    
class FunctionalUnit(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(FunctionalUnit, node, server.FunctionalUnitType, server)
    
    state_machine: FunctionalStateMachine
    function_set: FunctionSet
    program_manager: ProgramManager

    async def init(self, server: Server):
        await super().init(server)
        self.function_set, self.state_machine, self.program_manager = await asyncio.gather(
            FunctionSet.propagate(await self.get_lads_child("FunctionSet"), server),
            FunctionalStateMachine.propagate(await self.get_lads_child("StateMachine"), server),
            ProgramManager.propagate(await self.get_lads_child("ProgramManager"), server),
        )

    async def finalize_init(self, device: Device):
        await super().finalize_init()
        self.device = device
        if self.function_set is not None:
            await asyncio.gather(*(function.finalize_init(self) for function in self.function_set.functions))
        # prepare subscriptions (data change will be handled by device)
        self.subscription_handler = SubscriptionHandler()
        events_handler = await self.subscription_handler.subscribe_events(self.server, self)

    @property
    def all_subscribed_variables(self) -> list[BaseVariable]:
        variables = self.subscribed_variables + self.state_machine.variables

        if self.function_set is not None:
            function_variables = self.function_set.all_variables
            # debug- check for none
            for variable in function_variables:
                if variable is None:
                    _logger.error(f"None variable detected in function {self.unique_name}")
            child_vars = list(filter(lambda variable: variable.subscription_level > SubscriptionLevel.Never, function_variables))
            variables = variables + child_vars
        if self.program_manager is not None:
            variables = variables + self.program_manager.variables
        return variables

    @property
    def unique_name(self) -> str:
        return f"{self.device.unique_name}{unique_name_delimiter}{self.display_name}"
    
    @property
    def functions(self) -> list[Function]:
        return self.function_set.functions
    
    @property
    def events(self) ->list[Event]:
        if self.subscription_handler is not None:
            return self.subscription_handler.event_list
        else:
            return []

class Function(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(Function, node, server.FunctionType, server)
    
    async def init(self, server: Server):
        await super().init(server)
        node = await self.get_lads_child("IsEnabled")
        self.is_enabled = await BaseVariable.propagate(node, server)
        self.function_set: FunctionSet = await self.get_lads_child("FunctionSet")
        if self.function_set is not None:
            self.function_set = await FunctionSet.propagate(self.function_set, server)

    async def finalize_init(self, functional_unit: FunctionalUnit):
        await super().finalize_init()
        self.functional_unit = functional_unit

    @property
    def unique_name(self) -> str:
        return f"{self.functional_unit.unique_name}{unique_name_delimiter}{self.display_name}"
    
    @property
    def functions(self) -> list[Function]:
        return self.function_set.functions
    
    @property
    def variables(self) ->list[BaseVariable]:
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

class BaseStateMachineFunction(Function):
    state_machine: StateMachine

    def __str__(self):
        return f"{super().__str__()}\n  {self.current_state}"
    
    async def init(self, server: Server):
        await super().init(server)        
        self.state_machine = await StateMachine.propagate(await self.get_lads_child("StateMachine"), server)

    @property
    def variables(self) ->list[Node]:
        return super().variables + [self.state_machine.current_state]

    @property
    def current_state(self) -> BaseVariable:
        return self.state_machine.current_state

class BaseFunctionalStateMachineFunction(BaseStateMachineFunction):
    async def init(self, server: Server):
        await super().init(server)
        if self.state_machine:        
            self.state_machine = await FunctionalStateMachine.propagate(self.state_machine, server)

class StartStopControlFunction(BaseFunctionalStateMachineFunction):#
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(StartStopControlFunction, node, server.StartStopControlFunctionType, server)
    
class AnalogSensorFunction(Function):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(AnalogSensorFunction, node, server.AnalogSensorFunctionType, server)

    def __str__(self):
        return f"{super().__str__()}\n  {self.sensor_value}"
    
    async def init(self, server: Server):
        await super().init(server)        
        self.sensor_value = await get_lads_analog_item(self, "SensorValue")

    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + [self.sensor_value]

class AnalogArraySensorFunction(AnalogSensorFunction):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(AnalogArraySensorFunction, node, server.AnalogArraySensorFunctionType, server)

class BaseControlFunction(BaseFunctionalStateMachineFunction):
    def __str__(self):
        return f"{super().__str__()}\n  {self.current_value}\n  {self.target_value}"
    
    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + [self.current_value, self.target_value]
    
class AnalogControlFunction(BaseControlFunction):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(AnalogControlFunction, node, server.AnalogControlFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)        
        self.current_value = await get_lads_analog_item(self, "CurrentValue")
        self.target_value = await get_lads_analog_item(self, "TargetValue")

class AnalogControlFunctionWithTotalizer(AnalogControlFunction):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(AnalogControlFunctionWithTotalizer, node, server.AnalogControlFunctionWithTotalizerType, server)

    def __str__(self):
        return f"{super().__str__()}\n  {self.totalized_value}"
    
    async def init(self, server: Server):
        await super().init(server)        
        self.totalized_value = await get_lads_analog_item(self, "TotalizedValue")

    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + [self.totalized_value]

class TwoStateDiscreteControlFunction(BaseControlFunction):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(TwoStateDiscreteControlFunction, node, server.TwoStateDiscreteControlFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)        
        self.current_value = await get_lads_two_state_discrete(self, "CurrentValue")
        self.target_value = await get_lads_two_state_discrete(self, "TargetValue")

class MultiStateDiscreteControlFunction(BaseControlFunction):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(MultiStateDiscreteControlFunction, node, server.MultiStateDiscreteControlFunctionType, server)

    async def init(self, server: Server):
        await super().init(server)        
        self.current_value = await get_lads_two_state_discrete(self, "CurrentValue")
        self.target_value = await get_lads_two_state_discrete(self, "TargetValue")

class ControllerParameter(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(ControllerParameter, node, server.ControllerParameterType, server)

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
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(ControllerParameterSet, node, server.ControllerParameterSetType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.controller_parameters: list[ControllerParameter] = await asyncio.gather(*(ControllerParameter.propagate(child, server) for child in self.children))
        self.controller_parameters.sort(key = lambda node: node.display_name)

class MulitModeControlFunction(BaseFunctionalStateMachineFunction):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(MulitModeControlFunction, node, server.MultiModeControlFunctionType, server)

    def __str__(self):
        s = ""
        for controller_parameter in self.controller_parameters:
            s = s + f"\n  {controller_parameter.__str__()}"
        return f"{super().__str__()}{s}"
    
    async def init(self, server: Server):
        await super().init(server)
        self.current_mode = await MultiStateDiscrete.propagate(await self.get_lads_child("CurrentMode"), server)
        self.controller_mode_set = await ControllerParameterSet.propagate(await self.get_lads_child("ControllerModeSet"), server)

    @property
    def controller_parameters(self) -> list[ControllerParameter]:
        return self.controller_mode_set.controller_parameters
    
    @property
    def variables(self) ->list[BaseVariable]:
        variables: list[BaseVariable] = []
        for controller_parameter in self.controller_parameters:
            variables.append(controller_parameter.target_value)
            variables.append(controller_parameter.current_value)
        return super().variables + variables
    
class CoverFunction(BaseStateMachineFunction):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(CoverFunction, node, server.CoverFunctionType, server)
             
async def propagate_to(cls: Type, node: Node, type_node: Node, server: Server) -> LADSNode:
    if node is None: return None
    node_class = await node.read_node_class()
    if node_class != ua.NodeClass.Method:
        assert await is_of_type(server, node, type_node), f"node {node.nodeid} is expexted to be of type {type_node.nodeid}"
    node.__class__ = cls
    propagated_node : cls = node
    await propagated_node.init(server)
    return propagated_node

async def get_lads_analog_item(parent: LADSNode, name: str) -> AnalogItem:
    node = await parent.get_lads_child(name)
    return await AnalogItem.propagate(node, parent.server)

async def get_lads_two_state_discrete(parent: LADSNode, name: str) -> TwoStateDiscrete:
    node = await parent.get_lads_child(name)
    return await TwoStateDiscrete.propagate(node, parent.server)

async def get_lads_multi_state_discrete(parent: LADSNode, name: str) -> MultiStateDiscrete:
    node = await parent.get_lads_child(name)
    return await MultiStateDiscrete.propagate(node, parent.server)

async def get_di_variable(parent: LADSNode, name: str) -> BaseVariable:
    return await BaseVariable.propagate(await parent.get_di_child(name), parent.server)

async def get_properties_and_variables(node: LADSNode) -> list[BaseVariable]:
    
    (variables, properties) = await asyncio.gather(node.get_variables(), node.get_properties())
    variables.extend(properties)
    result: list[BaseVariable] = await asyncio.gather(*(BaseVariable.propagate(variable, node.server) for variable in variables))
    return result

async def run_connection_async(client: Client, server: Server):
    reconnecting = False
    while server.running:
        try:
            async with client:
                if reconnecting:
                    # does not work :-(
                    # await client.disconnect()
                    await client.disconnect_socket()
                    await asyncio.sleep(2)
                    await client.connect()
                    reconnecting = False
                await server.init()
                while server.running:
                    await server.evaluate()
                    await client.check_connection()
        except (ConnectionError, ua.UaError):
            _logger.warning("Reconnecting in 2 seconds")
            reconnecting = True
        except Exception as error:
            _logger.error(error)
            await asyncio.sleep(2)
    await client.disconnect()
    _logger.warning("disconnected")

import threading

def run_connection(client: Client, server: Server):
    asyncio.run(run_connection_async(client, server))

DefaultServerUrl = "opc.tcp://localhost:26543"

import time

def create_connection(url = "opc.tcp://localhost:26543") -> Server:
    client = Client(url)
    server = Server(client, "My Server")
    t = threading.Thread(target=run_connection, args=[client, server], daemon=True, name=f"LADS OPC UA Connection {server.name}")
    t.start()
    while not server.initialized:
        time.sleep(0.1)
    if True:
        for device in server.devices:
            print (device)
            for component in device.components:
                print(component)
            for functional_unit in device.functional_units:
                print(functional_unit)
                for function in functional_unit.functions:
                    print(function) 
    return server 

def main():
    server = create_connection()
    _logger.info(f"Connected to {server}")
    if False:
        fu = server.devices[0].functional_units[0]
        sm: FunctionalStateMachine = fu.state_machine
        sm.start_program("MyTemplate", 
                         pd.DataFrame({"Key": ["Temperature"], "Value": [37.0]}), 
                         "MyJob", 
                         "MyTask", 
                         pd.DataFrame())
        time.sleep(10)
        server.running = False
    # time.sleep(1)
    while True:
        _logger.info("ping")
        time.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()