import asyncio
import logging
from typing import Type, NewType, Any
from asyncua import Client, ua, Node
from asyncua.common.subscription import Subscription, SubscriptionItemData, DataChangeNotif
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
            await device.finalize_init()
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

class SubHandler(object):
    def datachange_notification(self, node, val, data):
        print("New data change event", node, val)

    def event_notification(self, event):
        print("New event", event)

class LADSNode(Node):

    async def init(self, server: Server):
        self.server: Server = server
        self.browse_name = await self.read_browse_name()
        self._display_name = await self.read_display_name()
        self.description = await self.read_description()

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
        self.manufacturer = await get_di_variable(self, "Manufacturer")
        self.model = await get_di_variable(self, "Model")
        self.serial_number = await get_di_variable(self, "SerialNumber")
        self.device_health = await get_di_variable(self, "DeviceHealth")

    @property
    def variables(self) ->list[Node]:
        return [self.manufacturer, self.model, self.serial_number, self.device_health]

class Device(Component):

    async def init(self, server: Server):
        await super().init(server)
        functional_unit_set = await self.get_lads_child("FunctionalUnitSet")
        nodes = await functional_unit_set.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        self.functional_units: list[FunctionalUnit] = []
        for node in nodes:
            functional_unit: FunctionalUnit = await propagate_to_FunctionalUnit(node, server)
            self.functional_units.append(functional_unit)

    async def finalize_init(self):
        await super().finalize_init()
        self.subscription = await self.server.client.create_subscription(500, SubHandler())
        handler = await self.subscription.subscribe_data_change(self.variables)
        for functional_unit in self.functional_units:
            await functional_unit.finalize_init()

    @property
    def variables(self) ->list[Node]:
        return super().variables

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

    @property
    def all_variables(self) -> list[Node]:
        nodes = self.variables        
        for function in self.functions:
            nodes = nodes + function.all_variables
        return nodes

Function = NewType("Function", LADSNode)

class FunctionalUnit(LADSNode):

    async def init(self, server: Server):
        await super().init(server)
        self.function_set: FunctionSet = await self.get_lads_child("FunctionSet")
        if self.function_set is not None:
            self.function_set = await propagate_to_FunctionSet(self.function_set, server)

    async def finalize_init(self):
        await super().finalize_init()
        self.subscription = await self.server.client.create_subscription(500, self)
        nodes = self.variables
        if self.function_set is not None:
            variables = self.function_set.all_variables
            nodes = nodes + variables
        self.subscribed_variables = dict((node.nodeid, node) for node in nodes)
        handler = await self.subscription.subscribe_data_change(nodes)

    def datachange_notification(self, node: Node, val: Any, data: DataChangeNotif):
        variable = self.subscribed_variables[node.nodeid]
        assert(variable is not None)
        variable.data_change_notification(data)
        print(f"{self.display_name}.{variable.display_name} = {val}")

    @property
    def functions(self) -> list[Function]:
        return self.function_set.functions
    
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

class AnalogControlFunction(BaseControlFunction):
    def __str__(self):
        return f"AnalogControlFunction(BrowseName={self.display_name})\n  {self.current_value}\n  {self.target_value}"
    
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

async def propagate_to_AnalogControlFunction(node: Node, server: Server) -> AnalogControlFunction:
    return await propagate_to(AnalogControlFunction, node, Server.AnalogControlFuntionType, server)

async def propagate_to_AnalogSensorFunction(node: Node, server: Server) -> AnalogControlFunction:
    return await propagate_to(AnalogSensorFunction, node, Server.AnalogSensorFunctionType, server)

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
        #handle = await sub.subscribe_data_change([temperatureSP, temperaturePV, temperatureState])
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