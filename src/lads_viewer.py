import streamlit as st
import pandas as pd
import time
import random
import datetime as dt
import plotly.graph_objects as go
from typing import Tuple
import lads as lads

class Node():
    def __init__(self, name: str) -> None:
        self.name = name
        
class Device(Node):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.functional_units: list[FunctionalUnit] = []

class FunctionalUnit(Node):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.functions: list[Function] = []

class Function(Node):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.is_enabled = True

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
    def __init__(self, name: str, target_value: float, current_value: float, eu: str) -> None:
        super().__init__(name)
        self.target_value = AnalogItem("Target Value", target_value, eu)
        self.current_value = AnalogItem("Currrent Value", current_value, eu, history=True)

class AnalogSensorFunction(Function):
    def __init__(self, name: str, sensor_value: float, eu: str) -> None:
        super().__init__(name)
        self.sensor_value = AnalogItem("Sensor Value", sensor_value, eu, history=True)

my_device = Device("My Device")
my_unit = FunctionalUnit("My Unit")    
my_device.functional_units.append(my_unit)
my_unit.functions.append(AnalogControlFunction("Temperature Controller", 37, 25, "°C"))
my_unit.functions.append(AnalogControlFunction("Speed Controller", 1000, 0, "rpm"))
my_unit.functions.append(AnalogControlFunction("Flow Controller", 10, 0, "sL/h"))
my_unit.functions.append(AnalogSensorFunction("XCO2 Sensor", 4.0, "%"))


# session state
selectedDeviceKey = "selected_device"
selectedFunctionKey = "selected_function"
devices = [my_unit.name]
functions = list(map(lambda function: function.name, my_unit.functions))

if selectedFunctionKey not in st.session_state:
    st.session_state[selectedFunctionKey] = functions[0]    
if selectedDeviceKey not in st.session_state:
    st.session_state[selectedDeviceKey] = devices[0]    


def format_float(x: float, decis = 1) -> str:
    return "{0:.1f}".format(x)

def update_functions(container):
    with container:        
        with st.container():
            for function in my_unit.functions:
                if function.is_enabled:
                    function_widget = st.container()
                    with function_widget:
                        # st.divider()
                        st.write(f"**{function.name}**")
                        sp_col, pv_col = st.columns(2)
                        if isinstance(function, AnalogControlFunction):
                            acf: AnalogControlFunction = function
                            with sp_col:
                                # target_value = st.number_input("**37.0** °C", on_change=new_target_value, value= 37.0, key=f"{st.session_state[selectedDeviceKey]}.{function}")
                                st.write(f"**{format_float(acf.target_value.value)}** {acf.target_value.eu}")
                            with pv_col: 
                                st.subheader(f":blue[**{format_float(acf.current_value.value)}** {acf.current_value.eu}]")
                        elif isinstance(function, AnalogSensorFunction):
                            asf: AnalogSensorFunction = function
                            with pv_col: 
                                st.subheader(f":blue[**{format_float(asf.sensor_value.value)}** {asf.sensor_value.eu}]")
                        

def update_charts(container, functional_unit: FunctionalUnit, use_plotly=True):
    with container:        
        with st.container():
            # collect analog items
            traces: list[Tuple[Function, AnalogItem]] = []
            for function in functional_unit.functions:
                analog_item: AnalogItem = None
                if isinstance(function, AnalogControlFunction):
                    analog_item = function.current_value
                elif isinstance(function, AnalogSensorFunction):
                    analog_item = function.sensor_value
                if analog_item is not None:
                    if analog_item.history is not None:
                        traces.append((function, analog_item))

            if use_plotly:
                # add traces
                fig = go.Figure()
                layout_dict = {}
                pos_left = 0.1
                pos_right = 0.9
                layout_dict["xaxis"] = dict(domain=[pos_left, pos_right])
                index = 0
                count = len(traces)
                for trace in traces:
                    index += 1
                    function, analog_item = trace
                    df: pd.DataFrame = analog_item.history
                    fig.add_trace(go.Scatter(
                        x = df.index.array,
                        y = df.iloc[:, 0].array,
                        name = function.name,
                        yaxis = f"y{index}"
                    ))
                    color="#404040"
                    yaxis_key = "yaxis" if index <= 1 else f"yaxis{index}"
                    left  = index <= count / 2
                    side = "left" if left else "right"
                    position = pos_left - 0.2 * (index - 1) if left else pos_right + 0.2 * ((index - 1) - count / 2)
                    position = 0 if position < 0 else 1 if position > 1 else position
                    yaxis_dict = dict(
                        title = f"{function.name} [{analog_item.eu}]",
                        titlefont = dict(color=color),
                        tickfont = dict(color=color),
                    )
                    if index > 1:
                        yaxis_dict["anchor"] = "x"
                        yaxis_dict["overlaying"] = "y"
                        yaxis_dict["side"] = side
                        yaxis_dict["position"] = position
                        
                    layout_dict[yaxis_key] = yaxis_dict
                
            
                fig.update_layout(layout_dict)

                fig.update_layout(
                    title_text = functional_unit.name,
                    width = 1000,
                    title_x = 0.1,
                    legend = dict(yanchor = "top", xanchor = "left", x = 0.7, y = 1.35),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                for trace in traces:
                    function, analog_item = trace
                    st.line_chart(data = analog_item.history, height = 100)


def main():
    st.set_page_config(page_title="LADS OPC UA Client", layout="wide")
    container_functions = None
    
    # Title and description
    st.header("LADS OPC UA Client")
    st.write(f"**{st.session_state[selectedDeviceKey]}**")

    # Devices list on the left side
    st.session_state[selectedDeviceKey] = st.sidebar.selectbox("Select a device", devices)
    tab_functions, tab_program_manager, tab_device = st.tabs(["Operation", "Program Management", "Asset Management"])
    container_functional_unit = st.empty()
    with container_functional_unit:
        with tab_functions:
            col1, col2 = st.columns([1, 2])

            # Functions list in the detail view
            with col1:
                container_functions = st.empty()
                update_functions(container_functions)

            # Chart in the detail view
            with col2:
                container_chart = st.empty()

        with tab_program_manager:
            col_templates, col_status, col_results = st.columns([1, 1, 1])
            with col_templates:
                st.write("**Templates**")
            with col_status:
                st.write("**Status**")
            with col_results:
                st.write("**Results**")

        with tab_device:
            st.write("**Nameplate**")
            name_plate = pd.DataFrame({
                "Parameter": ["Manufacturer", "Model", "Serialnumber", "Location"],
                "Value": ["AixEngineers", "AixBox Controller", "08154711", "Turmstrasse, Aachen"],
                "Description": ["Device manufacturer", "Device model", "Device serialnumber", "Location"],
            })
            st.table(name_plate)

        # Display the events table
        with st.container():
            events = pd.DataFrame({
                'Timestamp': ['2022-01-01 10:00:00', '2022-01-02 14:30:00', '2022-01-03 09:45:00'],
                'Severity': [0, 0, 0],
                'Source': ['Temperature Controller', 'Temperature Controller', 'Pressure Controller'],
                'Message': ['Event 1', 'Event 2', 'Event 3'],
            })
            st.divider()
            st.write("**Events**")
            st.table(events)
            
        # update loop
        cf = 0.1
        index = 5
        while(True):
            index += 1
            functional_unit = my_unit
            for function in functional_unit.functions:
                if isinstance(function, AnalogControlFunction):
                    acf: AnalogControlFunction = function
                    acf.current_value.value = cf * acf.target_value.value + (1 - cf) * acf.current_value.value + 0.1 * (random.random() - 0.5)
            update_functions(container_functions)
            if index >= 5:
                index = 0
                update_charts(container_chart, functional_unit, True)
            time.sleep(1)

if __name__ == '__main__':
    main()
