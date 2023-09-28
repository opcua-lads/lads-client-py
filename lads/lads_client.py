import asyncio
import logging

from asyncua import Client, ua, Node
_logger = logging.getLogger(__name__)

async def get_parents(node: Node, root_node: Node = None) -> list[Node]:
    parent = await node.get_parent()
    if parent == root_node:
        return []
    else:
        return [parent] + await get_parents(parent, root_node)

from enum import IntEnum

class ObjectIds(IntEnum):
    DeviceType = 1002
    ComponentType = 1024
    FunctionalUnitType= 1003
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
    FunctionalUnitType: Node
    FunctionType: Node
    AnalogSensorFunctionType: Node
    AnalogControlFuntionType: Node
    CoverFunctionType: Node

    def __init__(self, client: Client, uri: str) -> None:
        self.client = client
        self.uri = uri
        self.server = self
        self.devices: list[Device] = []

    async def _init(self):
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
        Server.FunctionalUnitType = self.get_lads_node(ObjectIds.FunctionalUnitType)
        Server.FunctionType = self.get_lads_node(ObjectIds.FunctionType)
        Server.AnalogSensorFunctionType = self.get_lads_node(ObjectIds.AnalogSensorFunctionType)
        Server.AnalogControlFuntionType = self.get_lads_node(ObjectIds.AnalogControlFuntionType)
        Server.CoverControlFunctionType = self.get_lads_node(ObjectIds.CoverFunctionType)

        # browse for devices in DeviceSet
        device_set = await self.client.nodes.objects.get_child(f"{self.ns_DI}:DeviceSet")
        nodes = await device_set.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        for node in nodes:
            device: Device = await create_lads_node(Device, self, node)
            self.devices.append(device)

    def get_lads_node(self, id: int) -> Node | None:
        return self.client.get_node( ua.NodeId(id, self.ns_LADS))
    
async def create_lads_server(client: Client, uri: str) -> Server:
    server = Server(client, uri)
    await server._init()
    return server

class LADSNode(Node):

    def __init__(self, session, nodeid: ua.NodeId):
        super().__init__(session, nodeid)
        self.type_nodes = None
        self.server: Server = None
        
    # propagate from Node
    @classmethod
    def from_Node(cls, server: Server, node: Node):
        lads_node = cls(node.session, node.nodeid)
        lads_node.server = server
        for key, value in node.__dict__.items():
            lads_node.__dict__[key] = value
        return lads_node
    
    async def _init(self, parent: Node):
        self.browse_name = await self.read_browse_name()

    def __str__(self):
        return f"{__class__} {self.browse_name}"

    async def is_of_type(self, type_node: Node) -> bool:
        if self.type_nodes is None:
            self.type_nodes = await self._browse_types()
        for node in self.type_nodes:
            if node == type_node:
                return True
        return False

    async def _browse_types(self) -> list[Node]:
        type_node_id =  await self.read_type_definition()
        type_node = Node(self.session, type_node_id)
        root_node = Server.BaseObjectType
        node_class = await self.read_node_class()
        if node_class == ua.NodeClass.Variable:
            root_node = Server.BaseVariableType
        else:
            root_node = Server.BaseObjectType
        return [type_node] + await get_parents(type_node, root_node)
    
    async def get_lads_child(self, name : str) -> Node:
        return await self.get_child(ua.QualifiedName(name, self.server.ns_LADS))

async def create_lads_node(cls, server: Server, node: Node, parent: LADSNode = None) -> LADSNode:
    lads_node: LADSNode = cls.from_Node(server, node)
    await lads_node._init(parent)
    return lads_node
    
class Device(LADSNode):

    def __init__(self, session, nodeid: ua.NodeId):
        super().__init__(session, nodeid)

    def __str__(self):
        return f"Device(BrowseName={self.browse_name})"

    async def _init(self, parent: LADSNode):
        await super()._init(parent)
        assert await self.is_of_type(Server.DeviceType)
        functional_unit_set = await self.get_lads_child("FunctionalUnitSet")
        nodes = await functional_unit_set.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        self.functional_units: list[FunctionalUnit] = []
        for node in nodes:
            functional_unit: FunctionalUnit = await create_lads_node(FunctionalUnit, self.server, node, self)
            self.functional_units.append(functional_unit)

class FunctionalUnit(LADSNode):
    def __str__(self):
        return f"FunctionalUnit(BrowseName={self.browse_name})"

    async def _init(self, parent: LADSNode):
        await super()._init(parent)
        assert await self.is_of_type(Server.FunctionalUnitType)
        self.parent: Device = parent
        self.subscription_handler = SubHandler()
        self.subscription = await self.server.client.create_subscription(500, self.subscription_handler)
        function_set = await self.get_lads_child("FunctionSet")
        nodes = await function_set.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        self.functions: list[Function] = []
        for node in nodes:
            abstract_function: Function = await create_function(self, node)
            function = abstract_function
            if await abstract_function.is_of_type(Server.AnalogControlFuntionType):
                function: AnalogControlFunction = await create_lads_node(AnalogControlFunction, self.server, function, self)
            elif await abstract_function.is_of_type(Server.AnalogSensorFunctionType):
                function: AnalogSensorFunction = await create_lads_node(AnalogSensorFunction, self.server, function, self)
            else:
                function = abstract_function
            self.functions.append(function)

class Function(LADSNode):

    async def _init(self, parent: LADSNode):
        await super()._init(parent)
        self.parent: FunctionalUnit | Function = parent
        assert await self.is_of_type(Server.FunctionType)
        self.is_enabled = await self.get_lads_child("IsEnabled")

    @property
    def subscription(self):
        return self.parent.subscription

async def create_function(parent: FunctionalUnit | Function, node: Node) -> Function:
    return await create_lads_node(Function, parent.server, node, parent) 

class AnalogItem(LADSNode):
    def __str__(self):
        return f"AnalogItem(BrowseName={self.browse_name}) = {self.value} [{self.eu.DisplayName.Text}]"
    
    async def _init(self, parent: LADSNode):
        await super()._init(parent)
        assert await self.is_of_type(Server.AnalogItemType)
        self.parent: Function = parent
        self.value = await self.get_value()
        self.eu: ua.EUInformation = None
        self.range: ua.Range = None
        try:
            self.engineering_units = await self.get_child("EngineeringUnits")
        except:
            pass
        finally:
            self.eu: ua.EUInformation = await self.engineering_units.get_value()
        try:
            self.eu_range = await self.get_child("EURange")
        except:
            pass
        finally:
            self.range: ua.Range = await self.eu_range.get_value()

async def create_analog_item(parent: Function, name: str) -> AnalogItem: 
    return await create_lads_node(AnalogItem, parent.server, await parent.get_lads_child(name), parent)

class AnalogControlFunction(Function):
    def __str__(self):
        return f"AnalogControlFunction(BrowseName={self.browse_name})\n  {self.current_value}\n  {self.target_value}"
    
    async def _init(self, parent: LADSNode):
        await super()._init(parent)
        assert await self.is_of_type(Server.AnalogControlFuntionType)
        state_machine = await self.get_lads_child("StateMachine")
        self.current_state = await state_machine.get_child("CurrentState")
        self.current_value = await create_analog_item(self, "CurrentValue")
        self.target_value = await create_analog_item(self, "TargetValue")
        handler = await self.subscription.subscribe_data_change([self.current_state, self.target_value, self.current_value])

class AnalogSensorFunction(Function):
    def __str__(self):
        return f"AnalogSensorFunction(BrowseName={self.browse_name})\n  {self.sensor_value}"
    
    async def _init(self, parent: LADSNode):
        await super()._init(parent)
        assert await self.is_of_type(Server.AnalogSensorFunctionType)
        self.sensor_value = await create_analog_item(self, "SensorValue")
        handler = await self.subscription.subscribe_data_change([self.sensor_value])

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