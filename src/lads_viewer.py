import streamlit as st
import datetime as dt
import pandas as pd
import time, math
import plotly.graph_objects as go
from typing import Tuple
# from lads_client_sim import Server, FunctionalUnit, Function, AnalogControlFunction, AnalogSensorFunction, AnalogItem, create_server
from lads_client import BaseControlFunction, CoverFunction, Server, Device, FunctionalUnit, Function, AnalogControlFunction, AnalogSensorFunction, BaseVariable, AnalogItem, StartStopControlFunction, create_connection, DefaultServerUrl
from asyncua import ua

st.set_page_config(page_title="LADS OPC UA Client", layout="wide")

@st.cache_resource
def get_server_connection(url: str) -> Server:
    server: Server = create_connection(url)
    print(f"Created server {url}")
    return server

def format_value(x: float | list[float], decis = 1) -> str:
    result = "NaN"
    try:
        if isinstance(x, list):
            result = f"[{format_number(x[0], decis)} .. {format_number(x[len(x) - 1], decis)}]"
        else:
            result = format_number(x, decis)
    finally:
        return result

def format_number(x: float, decis = 1) -> str:
    result = "NaN"
    try:
        result = "{0:.1f}".format(x)
    finally:
        return result

def state_color(function: BaseControlFunction) -> str:
    return "blue" if "Running" in str(function.current_state.value_str) else "gray"

def update_functions(container, functional_unit: FunctionalUnit,):
    with container:        
        with st.container():
            idx = 0
            for function in functional_unit.functions:
                if function.is_enabled:
                    idx += 1
                    with st.expander(label=f"**{function.display_name}**", expanded=idx < 10):
                        sp_col, pv_col = st.columns([4, 5])
                        if isinstance(function, AnalogControlFunction):
                            with sp_col:
                                # target_value = st.number_input("**37.0** °C", on_change=new_target_value, value= 37.0, key=f"{st.session_state[selectedDeviceKey]}.{function}")
                                st.write(f"**{format_value(function.target_value.value)}** {function.target_value.eu}")
                            with pv_col: 
                                st.write(f":{state_color(function)}[**{format_value(function.current_value.value)}** {function.current_value.eu}]")
                        elif isinstance(function, AnalogSensorFunction):
                            with pv_col: 
                                st.write(f":blue[**{format_value(function.sensor_value.value)}** {function.sensor_value.eu}]")
                        elif isinstance(function, CoverFunction):
                            with pv_col: 
                                st.write(f":blue[**{function.current_state.value_str}**]")
                        elif isinstance(function, StartStopControlFunction):
                            with pv_col: 
                                st.write(f":{state_color(function)}[**{function.current_state.value_str}**]")
                       
def update_charts(container, functional_unit: FunctionalUnit, use_plotly=True):

        
    with container:        
        with st.container():
            # collect analog items with history
            traces: list[Tuple[Function, AnalogItem]] = []
            arrays: list[Tuple[Function, AnalogItem]] = []
            idx = 0


            for function in functional_unit.functions:
                analog_item: AnalogItem = None
                if isinstance(function, AnalogControlFunction):
                    analog_item = function.current_value
                elif isinstance(function, AnalogSensorFunction):
                    analog_item = function.sensor_value
                if analog_item is not None:
                    if isinstance(analog_item.value, list):
                        arrays.append((function, analog_item))
                    elif analog_item.history is not None:
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
                        name = function.display_name,
                        yaxis = f"y{index}"
                    ))
                    color="#404040"
                    yaxis_key = "yaxis" if index <= 1 else f"yaxis{index}"
                    left  = index <= count / 2
                    side = "left" if left else "right"
                    position = pos_left - 0.2 * (index - 1) if left else pos_right + 0.2 * ((index - 1) - count / 2)
                    position = 0 if position < 0 else 1 if position > 1 else position
                    yaxis_dict = dict(
                        title = f"{function.display_name} [{analog_item.eu}]",
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
                    # title_text = functional_unit.display_name,
                    width = 1000,
                    # title_x = 0.1,
                    legend = dict(yanchor = "top", xanchor = "left", x = 0.7, y = 1.35),
                )
                with st.expander("**Chart**", expanded=(idx==0)):
                    st.plotly_chart(fig, use_container_width=True)
                    idx += 1
            else:
                for trace in traces:
                    function, analog_item = trace
                    with st.expander(f"**Chart {function.display_name}**", expanded=(idx==0)):
                        st.line_chart(data = analog_item.history, height = 100)
                        idx += 1

            # tables
            for array in arrays:
                function, analog_item = array
                value = analog_item.value
                i = len(value)
                col_count = int(math.sqrt(3 / 2 * i))
                row_count = int(2 / 3 * col_count)
                cols: dict = {}
                cols["Plate"] = list(range(1, row_count + 1))
                # cols[f"**{function.display_name}**"] = list(range(1, row_count + 1))
                for col_idx in range(col_count):
                    values = []
                    for row_idx in range(row_count):
                        values.append(format_number(value[col_idx * row_count + row_idx]))
                    cols[chr(ord("A") + col_idx)] = values

                df = pd.DataFrame(cols)
                if analog_item.eu_range is not None:
                    eu_range: ua.Range = analog_item.eu_range
                    df.style.background_gradient(
                        axis=None, 
                        vmin=eu_range.Low, 
                        vmax=eu_range.High,
                        cmap="turbo"
                    )

                with st.expander(f"**{function.display_name}**", expanded=(idx==0)):
                    #st.write(f"**{function.display_name}**")
                    st.dataframe(
                        df,
                        use_container_width=True, 
                        hide_index=True,
                    )
                    idx += 1

def update_events(container, functional_unit: FunctionalUnit):
    events = functional_unit.subscription_handler.events
    if events is None:
        return
    last_event_update = functional_unit.subscription_handler.last_event_update
    last_event_list_update = st.session_state[lastEventListUpdateKey]
    if last_event_update == last_event_list_update:
        return
    st.session_state[lastEventListUpdateKey] = last_event_update
    print("updating event list")
    with container:
        with st.container():
            event_columns = events[["Time", "Severity", "SourceName", "Message"]]
            st.dataframe(
                event_columns, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Time": st.column_config.Column(
                        None, 
                        help="Timestamp of the event",
                        disabled=True,
                        width="medium"
                    ),
                    "Severity": st.column_config.Column(
                        None, 
                        help="Severity of the event",
                        disabled=True,
                        width="small"
                    ),
                    "SourceName": st.column_config.Column(
                        None, 
                        help="Source node of the event",
                        disabled=True,
                        width="medium"
                    ),
                    "Message": st.column_config.Column(
                        None, 
                        help="Event message",
                        disabled=True,
                        width="large"
                    ),
                }
            )

def insert_variables_table(variables: list[BaseVariable]):
    names = []
    values = [] 
    descriptions = []
    for variable in variables:
        names.append(variable.display_name if variable.alternate_display_name is None else variable.alternate_display_name)
        values.append(variable.value_str)
        descriptions.append(variable.description.Text if variable.description.Text is not None else "")
    data: pd.DataFrame = {"Name": names, "Value": values, "Description": descriptions, }
    column_config={
        "Name": st.column_config.Column(
            None, 
            help="Variable name",
            disabled=True,
            width="medium"
        ),
        "Value": st.column_config.Column(
            None, 
            help="Variable value",
            disabled=True,
            width="medium"
        ),
        "Description": st.column_config.Column(
            None, 
            help="Variable description",
            disabled=True,
            width="large"
        ),
    }
    st.dataframe(data, use_container_width=True, hide_index=True, column_config=column_config)

def update_device(container, device: Device):
    with container:
        with st.container():
            state_vars = [
                device.state_machine.current_state, 
                device.machinery_item_state.current_state,
                device.machinery_operation_mode.current_state
            ]
            with st.expander(f"**Device Status**", expanded=True):
                insert_variables_table(state_vars)
            with st.expander(f"**Device Nameplate**", expanded=True):
                insert_variables_table(device.name_plate_variables)
            with st.expander(f"**Component A**", expanded=False):
                pass
            with st.expander(f"**Component B**", expanded=False):
                pass

selectedFunctionalUnitKey = "selected_functional_unit"
lastEventListUpdateKey = "last_event_list_update"

def main():
    my_server = get_server_connection(DefaultServerUrl)
    functional_units = my_server.devices[0].functional_units

    # session state
    functional_unit_names = list(map(lambda functional_unit: functional_unit.unique_name, functional_units))
    if selectedFunctionalUnitKey not in st.session_state:
        st.session_state[selectedFunctionalUnitKey] = functional_unit_names[0]    
    # in anyway create a new last_event_update on rerun
    st.session_state[lastEventListUpdateKey] = dt.datetime.now()    

    container_functions = None
    
    # Devices list on the left side
    st.session_state[selectedFunctionalUnitKey] = st.sidebar.selectbox("Select a functional-unit", functional_unit_names)

    # Title and description
    selected_functional_unit = functional_units[0]
    selected_functional_unit_name = st.session_state[selectedFunctionalUnitKey]
    for functional_unit in functional_units:
        if functional_unit.unique_name == selected_functional_unit_name:
            selected_functional_unit = functional_unit

    st.header("LADS OPC UA Client")
    st.write(f"**{selected_functional_unit.unique_name}**")

    tab_functions, tab_program_manager, tab_device = st.tabs(["Operation", "Program Management", "Asset Management"])
    container_functional_unit = st.empty()
    with container_functional_unit:
        with tab_functions:
            col1, col2 = st.columns([1, 2])

            # Functions list in the detail view
            with col1:
                container_functions = st.empty()
                update_functions(container_functions, selected_functional_unit)

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
            container_device = st.empty()
            update_device(container_device, selected_functional_unit.device)

        # Display the events table
        with st.container():
            st.divider()
            st.write("**Events**")
            container_events = st.empty()
            update_events(container_events, selected_functional_unit)
            
        # update loop
        cf = 0.1
        index = 5
        while(True):
            update_functions(container_functions, selected_functional_unit)
            update_events(container_events, selected_functional_unit)
            update_device(container_device, selected_functional_unit.device)
            index += 1
            if index >= 5:
                index = 0
                update_charts(container_chart, selected_functional_unit, True)
            time.sleep(1)

if __name__ == '__main__':
    main()
