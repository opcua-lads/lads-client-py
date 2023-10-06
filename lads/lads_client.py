import asyncio
import logging
from typing import Type, NewType, Any
from asyncua import Client, ua, Node
from asyncua.common.subscription import DataChangeNotif
from asyncua.common.events import Event
from enum import IntEnum

_logger = logging.getLogger(__name__)

class ObjectIds(IntEnum):
    DeviceType = 1002
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
    DeviceType: Node
    ComponentType: Node
    FunctionalUnitSetType: Node
    FunctionalUnitType: Node
    FunctionSetType: Node
    FunctionType: Node
    AnalogSensorFunctionType: Node
    AnalogControlFuntionType: Node
    StartStopControlFunctionType: Node
    CoverFunctionType: Node

    def __init__(self, client: Client, uri: str) -> None:
        self.client = client
        self.uri = uri
        self.server = self
        self.devices: list[Device] = []

    async def init(self):
        # read namespace indices
        self.ns_DI = await self.client.get_namespace_index("http://opcfoundation.org/UA/DI/")
        self.ns_AMB = await self.client.get_namespace_index("http://opcfoundation.org/UA/AMB/")
        self.ns_Machinery = await self.client.get_namespace_index("http://opcfoundation.org/UA/Machinery/")
        self.ns_LADS = await self.client.get_namespace_index("http://opcfoundation.org/UA/LADS/")
        self.ns = await self.client.get_namespace_index(self.uri)

        # get well known type nodes
        Server.BaseObjectType = self.client.get_node(ua.ObjectIds.BaseObjectType)
        Server.BaseVariableType = self.client.get_node(ua.ObjectIds.BaseVariableType)
        Server.AnalogItemType = self.client.get_node(ua.ObjectIds.AnalogItemType)
        Server.DeviceType = self.get_lads_node(ObjectIds.DeviceType)
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
            device: Device = await propagate_to_Device(node, self)
            await device.finalize_init()
            self.devices.append(device)

    def get_lads_node(self, id: int) -> Node | None:
        return self.client.get_node( ua.NodeId(id, self.ns_LADS))
    
async def create_lads_server(client: Client, uri: str) -> Server:
    server = Server(client, uri)
    await server.init()
    return server

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

class SubscriptionHandler(object):

    def __init__(self) -> None:
        super().__init__()
        self.subscription = None
        self.subscribed_variables = None
        self.event_node = None
        self.events: list[Event] = []

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
 
    def datachange_notification(self, node: Node, val: Any, data: DataChangeNotif):
        variable = self.subscribed_variables[node.nodeid]
        assert(variable is not None)
        variable.data_change_notification(data)
        print(f"{variable.display_name} = {val}")

    def event_notification(self, event: Event):
        self.events.append(event)
        if len(self.events) > 1000:
            self.events.pop(0)
        print(f"{self.event_node.display_name}", event.__dict__["Message"])

class LADSNode(Node):

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
    def variables(self) ->list[Node]:
        return []
    
    def __str__(self):
        return f"{__class__} {self.display_name}"

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

class LADSSet(LADSNode):

    async def init(self, server: Server):
        await super().init(server)
        self.node_version = await propagate_to_BaseVariable(await self.get_child("NodeVersion"), server)

    @property
    def variables(self) ->list[Node]:
        return [self.node_version]

class Component(LADSNode):

    async def init(self, server: Server):
        await super().init(server)
        (self.manufacturer, self.model, self.serial_number, self.device_health) = await asyncio.gather(
            get_di_variable(self, "Manufacturer"),
            get_di_variable(self, "Model"),
            get_di_variable(self, "SerialNumber"),
            get_di_variable(self, "DeviceHealth")
        )
        
    @property
    def variables(self) ->list[Node]:
        return [self.manufacturer, self.model, self.serial_number, self.device_health]

class Device(Component):

    async def init(self, server: Server):
        await super().init(server)
        functional_unit_set = await self.get_lads_child("FunctionalUnitSet")
        nodes = await self.get_child_objects(functional_unit_set)
        self.functional_units: list[FunctionalUnit] = await asyncio.gather(*(propagate_to_FunctionalUnit(node, server) for node in nodes))

    async def finalize_init(self):
        await super().finalize_init()
        await asyncio.gather(*(functional_unit.finalize_init() for functional_unit in self.functional_units))
        self.subscription_handler = SubscriptionHandler()
        await self.subscription_handler.subscribe_data_change(self.server, self.variables)
        await self.subscription_handler.subscribe_events(self.server, self)

    @property
    def variables(self) ->list[Node]:
        return super().variables

    @property
    def events(self) ->list[Event]:
        if self.subscription_handler is not None:
            return self.subscription_handler.events
        else:
            return []

Function = NewType("Function", LADSNode)

class FunctionSet(LADSSet):

    async def init(self, server: Server):
        await super().init(server)
        nodes = await self.get_child_objects()
        self.functions: list[Function] = await asyncio.gather(*(self.propagate_to_function(node) for node in nodes))
    
    async def propagate_to_function(self, node: Node) -> Function:
            server = self.server
            types = await browse_types(node)
            if Server.AnalogControlFuntionType in types:
                function: AnalogControlFunction = await propagate_to_AnalogControlFunction(node, server)
            elif Server.AnalogSensorFunctionType in types:
                function: AnalogSensorFunction = await propagate_to_AnalogSensorFunction(node, server)
            elif Server.CoverFunctionType in types:
                function: CoverFunction = await propagate_to_CoverFunction(node, server)
            elif Server.StartStopControlFunctionType in types:
                function: StartStopControlFunction = await propagate_to_StartStopControlFunction(node, server)
            else:
                function = await propagate_to_Function(node, server)
            return function


    @property
    def all_variables(self) -> list[Node]:
        nodes = self.variables        
        for function in self.functions:
            nodes = nodes + function.all_variables
        return nodes

class FunctionalUnit(LADSNode):

    async def init(self, server: Server):
        await super().init(server)
        self.function_set: FunctionSet = await self.get_lads_child("FunctionSet")
        if self.function_set is not None:
            self.function_set = await propagate_to_FunctionSet(self.function_set, server)

    async def finalize_init(self):
        await super().finalize_init()
        nodes = self.variables
        if self.function_set is not None:
            variables = self.function_set.all_variables
            nodes = nodes + variables
        self.subscription_handler = SubscriptionHandler()
        await self.subscription_handler.subscribe_data_change(self.server, nodes)
        await self.subscription_handler.subscribe_events(self.server, self)

    @property
    def functions(self) -> list[Function]:
        return self.function_set.functions
    
    @property
    def events(self) ->list[Event]:
        if self.subscription_handler is not None:
            return self.subscription_handler.events
        else:
            return []

class Function(LADSNode):

    async def init(self, server: Server):
        await super().init(server)
        node = await self.get_lads_child("IsEnabled")
        self.is_enabled = await propagate_to_BaseVariable(node, server)
        self.function_set: FunctionSet = await self.get_lads_child("FunctionSet")
        if self.function_set is not None:
            self.function_set = await propagate_to_FunctionSet(self.function_set, server)

    @property
    def functions(self) -> list[Function]:
        return self.function_set.functions
    
    @property
    def variables(self) ->list[Node]:
        return [self.is_enabled]
    
    @property
    def all_variables(self) -> list[Node]:
        nodes = self.variables
        if self.function_set:
            nodes = nodes + self.function_set.variables
            for function in self.function_set.functions:
                variables = function.all_variables
                nodes = nodes + variables
        return nodes

class BaseVariable(LADSNode):
    def __str__(self):
        return f"BaseVariable(BrowseName={self.display_name}) = {self.value}"
    
    async def init(self, server: Server):
        await super().init(server)
        self.data_value = await self.read_data_value(raise_on_bad_status=False)

    @property
    def value(self) -> Any:
        if self.data_value:
            return self.data_value.Value.Value
        else:
            return None
        
    def data_change_notification(self, data: DataChangeNotif):
        self.data_value = data.monitored_item.Value

class AnalogItem(BaseVariable):
    def __str__(self):
        return f"AnalogItem(BrowseName={self.display_name}) = {self.value} [{self.engineering_units.DisplayName.Text}]"
    
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

class BaseControlFunction(Function):
    
    async def init(self, server: Server):
        await super().init(server)        
        state_machine = await self.get_lads_child("StateMachine")
        self.current_state: BaseVariable = await propagate_to_BaseVariable(await state_machine.get_child("CurrentState"), server)

    @property
    def variables(self) ->list[Node]:
        return super().variables + [self.current_state]

class StartStopControlFunction(BaseControlFunction):
    def __str__(self):
        return f"StartStopControlFunction({self.display_name})\n  {self.current_state}"
    
class AnalogControlFunction(BaseControlFunction):
    def __str__(self):
        return f"AnalogControlFunction({self.display_name})\n  {self.current_state}\n  {self.current_value}\n  {self.target_value}"
    
    async def init(self, server: Server):
        await super().init(server)        
        self.current_value = await get_lads_analog_item(self, "CurrentValue")
        self.target_value = await get_lads_analog_item(self, "TargetValue")

    @property
    def variables(self) ->list[Node]:
        return super().variables + [self.current_value, self.target_value]

class AnalogSensorFunction(Function):
    def __str__(self):
        return f"AnalogSensorFunction(BrowseName={self.display_name})\n  {self.sensor_value}"
    
    async def init(self, server: Server):
        await super().init(server)        
        self.sensor_value = await get_lads_analog_item(self, "SensorValue")

    @property
    def variables(self) ->list[Node]:
        return super().variables + [self.sensor_value]

class CoverFunction(Function):    
    def __str__(self):
        return f"CoverFunction({self.display_name})\n  {self.current_state.value}"
       
    async def init(self, server: Server):
        await super().init(server)        
        state_machine = await self.get_lads_child("StateMachine")
        self.current_state: BaseVariable = await propagate_to_BaseVariable(await state_machine.get_child("CurrentState"), server)

    @property
    def variables(self) ->list[Node]:
        return super().variables + [self.current_state]

async def propagate_to(cls: Type, node: Node, type_node: Node, server: Server) -> LADSNode:
    if node is None: return None
    assert await is_of_type(node, type_node)
    node.__class__ = cls
    propagated_node : cls = node
    await propagated_node.init(server)
    return propagated_node

async def propagate_to_BaseVariable(node: Node, server: Server) -> BaseVariable:
    return await propagate_to(BaseVariable, node, Server.BaseVariableType, server)

async def propagate_to_AnalogItem(node: Node, server: Server) -> AnalogItem:
    return await propagate_to(AnalogItem, node, Server.AnalogItemType, server)

async def propagate_to_Device(node: Node, server: Server) -> Device:
    return await propagate_to(Device, node, Server.DeviceType, server)

async def propagate_to_FunctionSet(node: Node, server: Server) -> FunctionSet:
    return await propagate_to(FunctionSet, node, Server.FunctionSetType, server)

async def propagate_to_FunctionalUnit(node: Node, server: Server) -> FunctionalUnit:
    return await propagate_to(FunctionalUnit, node, Server.FunctionalUnitType, server)

async def propagate_to_Function(node: Node, server: Server) -> Function:
    return await propagate_to(Function, node, Server.FunctionType, server)

async def propagate_to_StartStopControlFunction(node: Node, server: Server) -> StartStopControlFunction:
    return await propagate_to(StartStopControlFunction, node, Server.StartStopControlFunctionType, server)

async def propagate_to_AnalogControlFunction(node: Node, server: Server) -> AnalogControlFunction:
    return await propagate_to(AnalogControlFunction, node, Server.AnalogControlFuntionType, server)

async def propagate_to_AnalogSensorFunction(node: Node, server: Server) -> AnalogControlFunction:
    return await propagate_to(AnalogSensorFunction, node, Server.AnalogSensorFunctionType, server)

async def propagate_to_CoverFunction(node: Node, server: Server) -> CoverFunction:
    return await propagate_to(CoverFunction, node, Server.CoverFunctionType, server)

async def get_lads_analog_item(parent: LADSNode, name: str) -> AnalogItem:
    node = await parent.get_lads_child(name)
    return await propagate_to_AnalogItem(node, parent.server)

async def get_di_variable(parent: LADSNode, name: str) -> BaseVariable:
    return await propagate_to_BaseVariable(await parent.get_di_child(name), parent.server)

async def main():
    url = "opc.tcp://localhost:26543"
    async with Client(url=url) as client:
        _logger.info("Root node is: %r", client.nodes.root)
        _logger.info("Objects node is: %r", client.nodes.objects)

        # Node objects have methods to read and write node attributes as well as browse or populate address space
        _logger.info("Children of root are: %r", await client.nodes.root.get_children())

        # get a specific node knowing its node id
        #var = client.get_node(ua.NodeId(1002, 2))
        #var = client.get_node("ns=3;i=2002")
        #print(var)
        #await var.read_data_value() # get value of node as a DataValue object
        #await var.read_value() # get value of node as a python builtin
        #await var.write_value(ua.Variant([23], ua.VariantType.Int64)) #set node value using explicit data type
        #await var.write_value(3.9) # set node value using implicit data type

        # Now getting a variable node using its browse path
        uri = "http://spectaris.de/LuminescenceReader/"
        server = await create_lads_server(client, uri)
        for device in server.devices:
            print (device)
            for functional_unit in device.functional_units:
                print(functional_unit)
                for function in functional_unit.functions:
                    print(function) 

        # calling a method on server
        # res = await obj.call_method("2:multiply", 3, "klk")
        # _logger.info("method result is: %r", res)
        while True:
            
            await asyncio.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    asyncio.run(main())