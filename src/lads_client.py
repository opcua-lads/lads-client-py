import asyncio
import logging
import pandas as pd
import datetime as dt
from typing import Type, NewType, Any, Self
from asyncua import Client, ua, Node
from asyncua.common.subscription import DataChangeNotif
from asyncua.common.events import Event
from enum import IntEnum

_logger = logging.getLogger(__name__)

class ObjectIds(IntEnum):
    DeviceType = 1002
    ComponentSetType = 1025
    ComponentType = 1024
    FunctionalUnitSetType = 1023
    FunctionalUnitType= 1003
    FunctionSetType = 1026
    FunctionType= 1004
    AnalogSensorFunctionType = 1016
    AnalogControlFuntionType = 1009
    StartStopControlFunctionType = 1032
    CoverFunctionType = 1011

class Server():
    BaseObjectType: Node
    FiniteStateMachineType: Node
    DeviceType: Node
    ComponentSetType: Node
    ComponentType: Node
    FunctionalUnitSetType: Node
    FunctionalUnitType: Node
    FunctionSetType: Node
    FunctionType: Node
    AnalogSensorFunctionType: Node
    AnalogControlFuntionType: Node
    StartStopControlFunctionType: Node
    CoverFunctionType: Node

    def __init__(self, client: Client, name: str) -> None:
        self.client = client
        self.name = name
        self.server = self
        self.devices: list[Device] = []
        self.intialized = False
        self.running = True

    async def init(self):
        # read namespace indices
        self.ns_DI = await self.client.get_namespace_index("http://opcfoundation.org/UA/DI/")
        self.ns_AMB = await self.client.get_namespace_index("http://opcfoundation.org/UA/AMB/")
        self.ns_Machinery = await self.client.get_namespace_index("http://opcfoundation.org/UA/Machinery/")
        self.ns_LADS = await self.client.get_namespace_index("http://opcfoundation.org/UA/LADS/")

        # get well known type nodes
        Server.BaseObjectType = self.client.get_node(ua.ObjectIds.BaseObjectType)
        Server.FiniteStateMachineType = self.client.get_node(ua.ObjectIds.FiniteStateMachineType)
        Server.BaseVariableType = self.client.get_node(ua.ObjectIds.BaseVariableType)
        Server.AnalogItemType = self.client.get_node(ua.ObjectIds.AnalogItemType)
        Server.DeviceType = self.get_lads_node(ObjectIds.DeviceType)
        Server.ComponentSetType = self.get_lads_node(ObjectIds.ComponentSetType)
        Server.ComponentType = self.get_lads_node(ObjectIds.ComponentType)
        Server.FunctionalUnitSetType = self.get_lads_node(ObjectIds.FunctionalUnitSetType)
        Server.FunctionalUnitType = self.get_lads_node(ObjectIds.FunctionalUnitType)
        Server.FunctionSetType = self.get_lads_node(ObjectIds.FunctionSetType)
        Server.FunctionType = self.get_lads_node(ObjectIds.FunctionType)
        Server.AnalogSensorFunctionType = self.get_lads_node(ObjectIds.AnalogSensorFunctionType)
        Server.AnalogControlFuntionType = self.get_lads_node(ObjectIds.AnalogControlFuntionType)
        Server.StartStopControlFunctionType = self.get_lads_node(ObjectIds.StartStopControlFunctionType)
        Server.CoverFunctionType = self.get_lads_node(ObjectIds.CoverFunctionType)

        # browse for devices in DeviceSet
        device_set = await self.client.nodes.objects.get_child(f"{self.ns_DI}:DeviceSet")
        nodes = await device_set.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        for node in nodes:
            device: Device = await Device.propagate(node, self)
            await device.finalize_init()
            self.devices.append(device)
        self.intialized = True

    def get_lads_node(self, id: int) -> Node | None:
        return self.client.get_node( ua.NodeId(id, self.ns_LADS))
    
async def get_parent_nodes(node: Node, root_node: Node = None) -> list[Node]:
    parent = await node.get_parent()
    if parent == root_node:
        return [parent]
    else:
        return [parent] + await get_parent_nodes(parent, root_node)

async def browse_types(node: Node) -> list[Node]:
    type_node_id =  await node.read_type_definition()
    type_node = Node(node.session, type_node_id)
    root_node = Server.BaseObjectType
    node_class = await node.read_node_class()
    if node_class == ua.NodeClass.Variable:
        root_node = Server.BaseVariableType
    else:
        root_node = Server.BaseObjectType
    return [type_node] + await get_parent_nodes(type_node, root_node)

async def is_of_type(node: Node, type_node: Node) -> bool:
    types = await browse_types(node)
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

class SubscriptionHandler(object):

    def __init__(self) -> None:
        super().__init__()
        self.subscription = None
        self.subscribed_variables = None
        self.event_node = None
        self.events: pd.DataFrame = None
        self.last_event_update = dt.datetime.now()

    async def subscribe_data_change(self, server: Server, nodes: list[Node], period: float = 500):
        if self.subscription is None:
            self.subscription = await server.client.create_subscription(period, self)
        self.subscribed_variables = dict((node.nodeid, node) for node in nodes)
        return await self.subscription.subscribe_data_change(nodes)        
 
    async def subscribe_events(self, server: Server, node: Node, period: float = 500):
        if self.subscription is None:
            self.subscription = await server.client.create_subscription(period, self)
        self.event_node: LADSNode = node
        return await self.subscription.subscribe_events(node)        
 
    def datachange_notification(self, node
                                : Node, val: Any, data: DataChangeNotif):
        variable: Node = self.subscribed_variables[node.nodeid]
        assert(variable is not None)
        variable.data_change_notification(data)
        # print(f"{variable.display_name} = {val}")

    def event_notification(self, event: Event):
        fields_dict = event.get_event_props_as_fields_dict()
        event_fields = {}
        try:
            event_fields = {k: variant_value_to_str(v) for k, v in fields_dict.items()}
        except Exception as error:
            print(error)
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


LADSNode = NewType("BaseVariable", Node)
BaseVariable = NewType("BaseVariable", LADSNode)
NodeVersionVariable = NewType("NodeVersionVariable", BaseVariable)
Component = NewType("Component", LADSNode)
FunctionalUnit = NewType("FunctionalUnit", LADSNode)
Function = NewType("Function", LADSNode)
      
class LADSNode(Node):

    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        pass

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

    def datachange_notification(self, node: Node, val: Any, data: DataChangeNotif):
        print("New data change event", node, val)

    def event_notification(self, event):
        print("New event", event)

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
    def variables(self) ->list[Node]:
        return []
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.display_name})"

    async def get_child_or_none(self, name : ua.QualifiedName) -> Node:
        try:
            return await self.get_child(name)
        except:
            return None
        
    async def get_di_child(self, name : str) -> Node:
            return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_DI))
    
    async def get_machinery_child(self, name : str) -> Node:
            return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_Machinery))
    
    async def get_lads_child(self, name : str) -> Node:
            return await self.get_child_or_none(ua.QualifiedName(name, self.server.ns_LADS))
    
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

class StateMachine(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(StateMachine, node, server.FiniteStateMachineType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.current_state: BaseVariable = await StateVariable.propagate(await self.get_child("CurrentState"), server)
        self.current_state.alternate_display_name = self.display_name

class LADSSet(LADSNode):
    node_version: NodeVersionVariable
    children: list[Node] = []

    async def init(self, server: Server):
        await super().init(server)
        try:
            node_version = await self.get_child("NodeVersion")
            self.node_version = await NodeVersionVariable.propagate(node_version, server)
            self.node_version.set = self
        except:
            self.node_version = None
        finally:
            self.children = await self.get_child_objects()
        
    @property
    def variables(self) ->list[Node]:
        return [] if self.node_version is None else [self.node_version]
    
    async def node_version_changed(self):
        current_nodes = await self.children
        current_node_ids = set(map(lambda node: node.nodeid, current_nodes))
        previous_nodes = self.components
        previous_node_ids = set(map(lambda node: node.nodeid, previous_nodes))
        new_node_ids = current_node_ids.difference(previous_node_ids)
        deleted_node_ids = previous_node_ids.difference(current_node_ids)
        new_nodes = filter(lambda node_id: current_nodes , new_node_ids)

class ComponentSet(LADSSet):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(ComponentSet, node, server.ComponentSetType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.components: list[Component] = await asyncio.gather(*(Component.propagate(node, server) for node in self.children))
        self.components.sort(key = lambda node: node.display_name)

class Component(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(Component, node, server.ComponentType, server)
    
    async def init(self, server: Server):
        await super().init(server)
        self._variables = await get_properties_and_variables(self)
        self._variables.sort(key = lambda variable: variable.display_name)
        self.component_set = await ComponentSet.propagate(await self.get_machinery_child("Components"), server)

    @property
    def components(self) -> list[Component]:
        return [] if self.component_set is None else self.component_set.components
        
    @property
    def variables(self) ->list[BaseVariable]:
        return self._variables

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

    async def init(self, server: Server):
        await super().init(server)
        functional_unit_set = await self.get_lads_child("FunctionalUnitSet")
        nodes = await self.get_child_objects(functional_unit_set)
        self.functional_units: list[FunctionalUnit] = await asyncio.gather(*(FunctionalUnit.propagate(node, server) for node in nodes))
        self.state_machine, self.machinery_item_state, self.machinery_operation_mode = await asyncio.gather(
            StateMachine.propagate(await self.get_lads_child("StateMachine"), server),
            StateMachine.propagate(await self.get_machinery_child("MachineryItemState"), server),
            StateMachine.propagate(await self.get_machinery_child("MachineryOperationMode"), server),
        )

    async def finalize_init(self):
        await super().finalize_init()
        await asyncio.gather(*(functional_unit.finalize_init(self) for functional_unit in self.functional_units))
        self.subscription_handler = SubscriptionHandler()
        await self.subscription_handler.subscribe_data_change(self.server, self.variables)
        await self.subscription_handler.subscribe_events(self.server, self)

    @property
    def unique_name(self) -> str:
        return f"{self.server.name}{unique_name_delimiter}{self.display_name}"
    

    @property
    def variables(self) ->list[BaseVariable]:
        return self.name_plate_variables + [self.state_machine.current_state, self.machinery_item_state.current_state, self.machinery_operation_mode.current_state,]
    
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
        self.functions: list[Function] = await asyncio.gather(*(self.propagate_to_function(node) for node in self.children))
        self.functions.sort(key = lambda function: function.display_name)
    
    async def propagate_to_function(self, node: Node) -> Function:
            server = self.server
            types = await browse_types(node)
            if Server.AnalogControlFuntionType in types:
                function: AnalogControlFunction = await AnalogControlFunction.propagate(node, server)
            elif Server.AnalogSensorFunctionType in types:
                function: AnalogSensorFunction = await AnalogSensorFunction.propagate(node, server)
            elif Server.CoverFunctionType in types:
                function: CoverFunction = await CoverFunction.propagate(node, server)
            elif Server.StartStopControlFunctionType in types:
                function: StartStopControlFunction = await StartStopControlFunction.propagate(node, server)
            else:
                function = await Function.propagate(node, server)
            return function


    @property
    def all_variables(self) -> list[BaseVariable]:
        nodes = self.variables        
        for function in self.functions:
            nodes = nodes + function.all_variables
        return nodes

class FunctionalUnit(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(FunctionalUnit, node, server.FunctionalUnitType, server)
    
    state_machine: StateMachine
    function_set: FunctionSet

    async def init(self, server: Server):
        await super().init(server)
        self.function_set, self.state_machine = await asyncio.gather(
            FunctionSet.propagate(await self.get_lads_child("FunctionSet"), server),
            StateMachine.propagate(await self.get_lads_child("StateMachine"), server)
        )

    async def finalize_init(self, device: Device):
        await super().finalize_init()
        self.device = device
        nodes = self.variables
        if self.function_set is not None:
            await asyncio.gather(*(function.finalize_init(self) for function in self.function_set.functions))
            variables = self.function_set.all_variables
            nodes = nodes + variables
        self.subscription_handler = SubscriptionHandler()
        await self.subscription_handler.subscribe_data_change(self.server, nodes)
        await self.subscription_handler.subscribe_events(self.server, self)

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

class BaseVariable(LADSNode):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(BaseVariable, node, server.BaseVariableType, server)

    def __str__(self):
        return f"{super().__str__()} = {self.value}"
    
    async def init(self, server: Server):
        await super().init(server)
        (self.data_value, historizing) = await asyncio.gather(
            self.read_data_value(raise_on_bad_status=False),
            self.read_attribute(ua.AttributeIds.Historizing)
        )
        self.history = None
        if (historizing.Value.Value):
            self.history = pd.DataFrame({f"{self.display_name}": [self.value]}, index = [pd.to_datetime(self.data_value.SourceTimestamp)])

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

class NodeVersionVariable(BaseVariable):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(NodeVersionVariable, node, server.BaseVariableType, server)

    set: LADSSet = None

    def data_change_notification(self, data: DataChangeNotif):
        super().data_change_notification(data)
        if self.set is None: return
        self.set.node_version_changed()

class StateVariable(BaseVariable):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(StateVariable, node, server.BaseVariableType, server)

    @property
    def value_str(self) -> str:
        s =  super().value_str
        l = s.split(":")
        return s if len(l) < 2 else l[1]

class AnalogItem(BaseVariable):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(AnalogItem, node, server.AnalogItemType, server)

    def __str__(self):
        return f"{super().__str__()} [{self.engineering_units.DisplayName.Text}]"
    
    async def init(self, server: Server):
        await super().init(server)
        self.engineering_units: ua.EUInformation = None
        self.eu_range: ua.Range = None
        try:
            self.engineering_units = await self.get_child("EngineeringUnits")
        except:
            pass
        finally:
            self.engineering_units: ua.EUInformation = await self.engineering_units.get_value()
        try:
            self.eu_range = await self.get_child("EURange")
        except:
            pass
        finally:
            self.eu_range: ua.Range = await self.eu_range.get_value()

    @property
    def eu(self) -> str:
        if self.engineering_units is not None:
            return self.engineering_units.DisplayName.Text
        else:
            return ""

class BaseStateMachineFunction(Function):
    state_machine: StateMachine
    
    async def init(self, server: Server):
        await super().init(server)        
        self.state_machine = await StateMachine.propagate(await self.get_lads_child("StateMachine"), server)

    @property
    def variables(self) ->list[Node]:
        return super().variables + [self.state_machine.current_state]

    @property
    def current_state(self) -> BaseVariable:
        return self.state_machine.current_state

class BaseControlFunction(BaseStateMachineFunction):
    def __str__(self):
        return f"{super().__str__()}\n  {self.current_state}"

    async def init(self, server: Server):
        await super().init(server)        

class StartStopControlFunction(BaseControlFunction):#
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(StartStopControlFunction, node, server.StartStopControlFunctionType, server)
    
class AnalogControlFunction(BaseControlFunction):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(AnalogControlFunction, node, server.AnalogControlFuntionType, server)

    def __str__(self):
        return f"{super().__str__()}\n  {self.current_value}\n  {self.target_value}"
    
    async def init(self, server: Server):
        await super().init(server)        
        self.current_value = await get_lads_analog_item(self, "CurrentValue")
        self.target_value = await get_lads_analog_item(self, "TargetValue")

    @property
    def variables(self) ->list[BaseVariable]:
        return super().variables + [self.current_value, self.target_value]

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

class CoverFunction(BaseStateMachineFunction):
    @classmethod
    async def propagate(cls, node: Node, server: Server) -> Self:
        return await propagate_to(CoverFunction, node, server.CoverFunctionType, server)
       
async def propagate_to(cls: Type, node: Node, type_node: Node, server: Server) -> LADSNode:
    if node is None: return None
    assert await is_of_type(node, type_node)
    node.__class__ = cls
    propagated_node : cls = node
    await propagated_node.init(server)
    return propagated_node

async def get_lads_analog_item(parent: LADSNode, name: str) -> AnalogItem:
    node = await parent.get_lads_child(name)
    return await AnalogItem.propagate(node, parent.server)

async def get_di_variable(parent: LADSNode, name: str) -> BaseVariable:
    return await BaseVariable.propagate(await parent.get_di_child(name), parent.server)

async def get_properties_and_variables(node: LADSNode) -> list[BaseVariable]:
    
    (variables, properties) = await asyncio.gather(node.get_variables(), node.get_properties())
    variables.extend(properties)
    result: list[BaseVariable] = await asyncio.gather(*(BaseVariable.propagate(variable, node.server) for variable in variables))
    return result

async def run_connection_async(client: Client, server: Server):
    async with client:
        try:
            # Node objects have methods to read and write node attributes as well as browse or populate address space
            # get a specific node knowing its node id
            #var = client.get_node(ua.NodeId(1002, 2))
            #var = client.get_node("ns=3;i=2002")
            #print(var)
            #await var.read_data_value() # get value of node as a DataValue object
            #await var.read_value() # get value of node as a python builtin
            #await var.write_value(ua.Variant([23], ua.VariantType.Int64)) #set node value using explicit data type
            #await var.write_value(3.9) # set node value using implicit data type

            # Now getting a variable node using its browse path
            # server = await create_lads_server(client, "My Server")
            await server.init()
            for device in server.devices:
                print (device)
                for component in device.components:
                    print(component)
                for functional_unit in device.functional_units:
                    print(functional_unit)
                    for function in functional_unit.functions:
                        print(function) 
            while server.running:
                await asyncio.sleep(0.1)
        #except Exception as error:
            #print(error)
        finally:
            # await client.close_session()
            await client.disconnect()
            print("disconnected")

        # calling a method on server
        # res = await obj.call_method("2:multiply", 3, "klk")
        # _logger.info("method result is: %r", res)

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
    while not server.intialized:
        time.sleep(0.1)
    return server

def main():
    server = create_connection()
    print(server)
    # time.sleep(10)
    # server.running = False
    # time.sleep(1)
    while True:
        print("ping")
        time.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()