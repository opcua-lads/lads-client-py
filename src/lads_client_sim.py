import pandas as pd
import datetime as dt
import random
import time
import threading

class Node():
    def __init__(self, name: str) -> None:
        self.display_name = name

    @property
    def unique_name(self) -> str:
        return self.display_name
        
class Server(Node):
    def __init__(self, name: str, uri: str) -> None:
        super().__init__(name)
        self.uri = uri
        self.devices: list[Device] = []

class Device(Node):
    def __init__(self, name: str, server: Server) -> None:
        super().__init__(name)
        self.server = server
        self.functional_units: list[FunctionalUnit] = []

    @property
    def unique_name(self) -> str:
        return f"{self.server.unique_name}/{self.display_name}"

class FunctionalUnit(Node):
    def __init__(self, name: str, device: Device) -> None:
        super().__init__(name)
        self.device = device
        self.functions: list[Function] = []

    @property
    def unique_name(self) -> str:
        return f"{self.device.unique_name}/{self.display_name}"
    
class Function(Node):
    def __init__(self, name: str, functional_unit: FunctionalUnit) -> None:
        super().__init__(name)
        self.functional_unit = functional_unit
        self.is_enabled = True

    @property
    def unique_name(self) -> str:
        return f"{self.functional_unit.unique_name}/{self.display_name}"

class AnalogItem(Node):
    def __init__(self, name: str, value: float, eu: str, history = False) -> None:
        super().__init__(name)
        self._value = value
        self.eu = eu
        self.history = None
        if history:
            self.history = pd.DataFrame({f"{name}": [value]}, index = [pd.to_datetime(dt.datetime.now())])
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, x):
        self._value = x
        if self.history is not None:
            self.history.loc[pd.to_datetime(dt.datetime.now())] = x
            if len(self.history.index) > 600:
                self.history = self.history.tail(-1)


class AnalogControlFunction(Function):
    def __init__(self, name: str, functional_unit: FunctionalUnit, target_value: float, current_value: float, eu: str) -> None:
        super().__init__(name, functional_unit)
        self.target_value = AnalogItem("Target Value", target_value, eu)
        self.current_value = AnalogItem("Currrent Value", current_value, eu, history=True)

class AnalogSensorFunction(Function):
    def __init__(self, name: str, functional_unit: FunctionalUnit, sensor_value: float, eu: str) -> None:
        super().__init__(name, functional_unit)
        self.sensor_value = AnalogItem("Sensor Value", sensor_value, eu, history=True)

def simulate(server: Server):
    while True:
        # print("simulate", server.display_name, dt.datetime.now())
        cf = 0.1
        for device in server.devices:
            for functional_unit in device.functional_units:
                for function in functional_unit.functions:
                    if isinstance(function, AnalogControlFunction):
                        acf: AnalogControlFunction = function
                        acf.current_value.value = cf * acf.target_value.value + (1 - cf) * acf.current_value.value + 0.1 * (random.random() - 0.5)
        time.sleep(1)

def create_lads_server(uri: str) -> Server:
    server = Server("My Server", uri)
    device = Device("My Device", server)
    server.devices.append(device)
    for i in range(2):
        functional_unit = FunctionalUnit(f"My Unit {i + 1}", device)    
        functional_unit.functions.append(AnalogControlFunction("Temperature Controller", functional_unit, 37, 25, "°C"))
        functional_unit.functions.append(AnalogControlFunction("Speed Controller", functional_unit, 1000, 0, "rpm"))
        functional_unit.functions.append(AnalogControlFunction("Flow Controller", functional_unit, 10, 0, "sL/h"))
        functional_unit.functions.append(AnalogSensorFunction("XCO2 Sensor", functional_unit, 4.0, "%"))
        device.functional_units.append(functional_unit)
    t = threading.Thread(target=simulate, args=[server], daemon=True, name=f"Simulate {server.display_name}")
    t.start()
    return server

if __name__ == '__main__':
    server = create_lads_server("localhost")
    pass