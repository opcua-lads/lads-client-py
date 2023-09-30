import asyncio
import logging
from typing import Type
from asyncua import Client, ua, Node
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
    CoverFunctionType = 1011

class SubHandler(object):
    def datachange_notification(self, node, val, data):
        print("New data change event", node, val, data)

    def event_notification(self, event):
        print("New event", event)

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
        Server.CoverControlFunctionType = self.get_lads_node(ObjectIds.CoverFunctionType)

        # browse for devices in DeviceSet
        device_set = await self.client.nodes.objects.get_child(f"{self.ns_DI}:DeviceSet")
        nodes = await device_set.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        for node in nodes:
            device: Device = await propagate_to_Device(node, self)
            self.devices.append(device)

    def get_lads_node(self, id: int) -> Node | None:
        return self.client.get_node( ua.NodeId(id, self.ns_LADS))
    
async def create_lads_server(client: Client, uri: str) -> Server:
    server = Server(client, uri)
    await server.init()
    return server

async def get_parents(node: Node, root_node: Node = None) -> list[Node]:
    parent = await node.get_parent()
    if parent == root_node:
        return [parent]
    else:
        return [parent] + await get_parents(parent, root_node)

async def browse_types(node: Node) -> list[Node]:
    type_node_id =  await node.read_type_definition()
    type_node = Node(node.session, type_node_id)
    root_node = Server.BaseObjectType
    node_class = await node.read_node_class()
    if node_class == ua.NodeClass.Variable:
        root_node = Server.BaseVariableType
    else:
        root_node = Server.BaseObjectType
    return [type_node] + await get_parents(type_node, root_node)

async def is_of_type(node: Node, type_node: Node) -> bool:
    types = await browse_types(node)
    return type_node in types

class LADSNode(Node):

    async def init(self, server: Server):
        self.server: Server = server
        self.browse_name = await self.read_browse_name()
        self._display_name = await self.read_display_name()

    @property
    def display_name(self) -> str:
        if self._display_name is not None:
            return self._display_name.Text
        else:
            return self.browse_name.Name

    def __str__(self):
        return f"{__class__} {self.display_name}"

    async def get_lads_child(self, name : str) -> Node:
        try:
            return await self.get_child(ua.QualifiedName(name, self.server.ns_LADS))
        except:
            return None

class LADSSet(LADSNode):

    async def init(self, server: Server):
        await super().init(server)
        node = await self.get_child("NodeVersion")
        self.node_version = await propagate_to_BaseVariable(node, server)

class Device(LADSNode):

    async def init(self, server: Server):
        await super().init(server)
        functional_unit_set = await self.get_lads_child("FunctionalUnitSet")
        nodes = await functional_unit_set.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        self.functional_units: list[FunctionalUnit] = []
        for node in nodes:
            functional_unit: FunctionalUnit = await propagate_to_FunctionalUnit(node, server)
            self.functional_units.append(functional_unit)

class FunctionSet(LADSSet):

    async def init(self, server: Server):
        await super().init(server)
        nodes = await self.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        self.functions: list[Function] = []
        for node in nodes:
            types = await browse_types(node)
            if Server.AnalogControlFuntionType in types:
                function: AnalogControlFunction = await propagate_to_AnalogControlFunction(node, server)
            elif Server.AnalogSensorFunctionType in types:
                function: AnalogSensorFunction = await propagate_to_AnalogSensorFunction(node, server)
            else:
                function = await propagate_to_Function(node, server)
            self.functions.append(function)

class FunctionalUnit(LADSNode):

    async def init(self, server: Server):
        await super().init(server)
        self.subscription_handler = SubHandler()
        self.subscription = await server.client.create_subscription(500, self.subscription_handler)
        self.function_set = await self.get_lads_child("FunctionSet")
        self.functions = None
        if self.function_set is not None:
            self.function_set = await propagate_to_FunctionSet(self.function_set, server, self)
            self.functions = self.function_set.functions
    
class Function(LADSNode):

    async def init(self, server: Server):
        await super().init(server)
        node = await self.get_lads_child("IsEnabled")
        self.is_enabled = await propagate_to_BaseVariable(node, server)
        self.functions = None
        self.function_set = await self.get_lads_child("FunctionSet")
        if self.function_set is not None:
            self.function_set = await propagate_to_FunctionSet(self.function_set, server)
            self.functions = self.function_set.functions

class BaseVariable(LADSNode):
    def __str__(self):
        return f"BaseVariable(BrowseName={self.display_name}) = {self.value}"
    
    async def init(self, server: Server):
        await super().init(server)
        self.value = await self.get_value()

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

class AnalogControlFunction(Function):
    def __str__(self):
        return f"AnalogControlFunction(BrowseName={self.display_name})\n  {self.current_value}\n  {self.target_value}"
    
    async def init(self, server: Server):
        await super().init(server)        
        state_machine = await self.get_lads_child("StateMachine")
        self.current_state = await state_machine.get_child("CurrentState")
        self.current_value = await get_lads_analog_item(self, "CurrentValue")
        self.target_value = await get_lads_analog_item(self, "TargetValue")
        handler = await self.subscription.subscribe_data_change([self.current_state, self.target_value, self.current_value])

class AnalogSensorFunction(Function):
    def __str__(self):
        return f"AnalogSensorFunction(BrowseName={self.display_name})\n  {self.sensor_value}"
    
    async def init(self, server: Server):
        await super().init(server)        
        self.sensor_value = await get_lads_analog_item(self, "SensorValue")
        # handler = await self.subscription.subscribe_data_change([self.sensor_value])

async def propagate_to(cls: Type, node: Node, type_node: Node, server: Server) -> LADSNode:
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

async def propagate_to_Function(node: Node, server: Server, functional_unit: FunctionalUnit) -> Function:
    return await propagate_to(Function, node, Server.FunctionType, server)

async def propagate_to_AnalogControlFunction(node: Node, server: Server) -> AnalogControlFunction:
    return await propagate_to(AnalogControlFunction, node, Server.AnalogControlFuntionType, server)

async def propagate_to_AnalogSensorFunction(node: Node, server: Server) -> AnalogControlFunction:
    return await propagate_to(AnalogSensorFunction, node, Server.AnalogSensorFunctionType, server)

async def get_lads_analog_item(parent: LADSNode, name: str) -> AnalogItem:
    node = await parent.get_lads_child(name)
    return await propagate_to_AnalogItem(node, parent.server)


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


        functionalUnit = server.devices[0].functional_units[0]
        functionSet = await functionalUnit.get_child(["5:FunctionSet"])
        temperatureController = await functionSet.get_child(["6:TemperatureController"])
        # setting up AnalogControlfunction
        temperatureSP = await temperatureController.get_child(["5:TargetValue"])
        temperaturePV = await temperatureController.get_child(["5:CurrentValue"])
        temperatureState = await temperatureController.get_child(["5:StateMachine", "0:CurrentState"])


        # subscribing to a variable node
        handler = SubHandler()
        sub = await client.create_subscription(100, handler)
        # handle = await sub.subscribe_data_change([temperatureSP, temperaturePV, temperatureState])
        await asyncio.sleep(0.1)

        # we can also subscribe to events from server
        await sub.subscribe_events(temperatureController)
        # await sub.unsubscribe(handle)
        # await sub.delete()

        # calling a method on server
        # res = await obj.call_method("2:multiply", 3, "klk")
        # _logger.info("method result is: %r", res)
        while True:
            
            await asyncio.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    asyncio.run(main())