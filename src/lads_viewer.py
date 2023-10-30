import streamlit as st
import datetime as dt
import pandas as pd
import time, math
import plotly.graph_objects as go
from typing import Tuple
from lads_client import  BaseStateMachineFunction, FunctionalStateMachine, LADSNode, create_connection, DefaultServerUrl, remove_none, BaseVariable, AnalogItem, BaseControlFunction, Component, CoverFunction, Server, Device, FunctionalUnit, \
    Function, AnalogControlFunction, AnalogSensorFunction, StartStopControlFunction, MulitModeControlFunction, StateMachine, AnalogControlFunctionWithTotalizer
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
    s = str(function.current_state.value_str)
    return "green" if "Running" in s else "red" if "Abort" in s else "gray"

def call_state_machine_method(function: BaseStateMachineFunction):
    if function is None: return
    key = function.unique_name
    if not key in st.session_state:
        st.session_state[key] = None
    method_name = st.session_state[key]
    function.state_machine.call_method_by_name(method_name)

def write_variable_value(variable: BaseVariable):
    if variable is None: 
        return
    key = variable.nodeid
    if not key in st.session_state:
        return
    variable.set_value(st.session_state[key])

def add_variable_value_input(variable: BaseVariable, parent: LADSNode = None):
    if variable.has_write_access:
        help = f"{variable.display_name}: {variable.description.Text}"
        if parent is not None:
            help = f"{parent.display_name}.{help}"
        st.number_input(variable.display_name, on_change = write_variable_value(variable), value = float(variable.value), key=f"{variable.nodeid}", label_visibility="collapsed", help = help)

def show_functions(container, functional_unit: FunctionalUnit) -> dict:
    with container.container():
        idx = 0
        function_containers = {}
        for function in functional_unit.functions:
            if function.is_enabled:
                idx += 1
                label = f"**{function.display_name}**"
                # label = f"**{function.display_name}** :gray[{function.__class__.__name__}]"
                with st.expander(label=label, expanded=idx < 10):
                    col_static, col_sp, col_pv = st.columns([4, 4, 5])
                    with col_static:
                        if isinstance(function, BaseStateMachineFunction):
                            if isinstance(function, AnalogControlFunction):
                                add_variable_value_input(function.target_value, function)
                            elif isinstance(function, MulitModeControlFunction):
                                for controller_parameter in function.controller_parameters:
                                    add_variable_value_input(controller_parameter.target_value, controller_parameter)
                            method_names = function.state_machine.method_names
                            if len(method_names) > 0:
                                cmd = st.selectbox("Command", method_names, label_visibility="collapsed", key=function.unique_name, on_change=call_state_machine_method(function))                                
                    with col_sp:
                        container_sp = st.empty()
                    with col_pv:
                        container_pv = st.empty()
                    function_containers[function] = (container_sp, container_pv)
    update_functions(function_containers)
    return function_containers

def update_functions(function_containers: dict):

    for function, containers in function_containers.items():
        container_sp, container_pv = containers
        sp_col = container_sp.container()
        pv_col = container_pv.container()
        if isinstance(function, AnalogControlFunction):
            color = state_color(function)
            with sp_col:
                st.write(f":{color}[**{format_value(function.target_value.value)}** {function.target_value.eu}]")
                if isinstance(function, AnalogControlFunctionWithTotalizer):
                    st.write(":gray[Totalizer]")
            with pv_col: 
                st.write(f":blue[**{format_value(function.current_value.value)}** {function.current_value.eu}]")
                if isinstance(function, AnalogControlFunctionWithTotalizer):
                    st.write(f":blue[**{format_value(function.totalized_value.value)}** {function.totalized_value.eu}]")
        elif isinstance(function, AnalogSensorFunction):
            with pv_col: 
                st.write(f":blue[**{format_value(function.sensor_value.value)}** {function.sensor_value.eu}]")
        elif isinstance(function, CoverFunction):
            with pv_col: 
                st.write(f":blue[**{function.current_state.value_str}**]")
        elif isinstance(function, StartStopControlFunction):
            with pv_col: 
                st.write(f":{state_color(function)}[**{function.current_state.value_str}**]")
        elif isinstance(function, MulitModeControlFunction):
            for controller_parameter in function.controller_parameters:
                with sp_col:
                    st.write(f":{state_color(function)}[**{format_value(controller_parameter.target_value.value)}** {controller_parameter.target_value.eu}]")
                with pv_col: 
                    st.write(f":blue[**{format_value(controller_parameter.current_value.value)}** {controller_parameter.current_value.eu}]")
        
def update_charts(container, functional_unit: FunctionalUnit, use_plotly=True):
    with container.container():        
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
                    cmap="BlGnRd"
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

def show_variables_table(variables: list[BaseVariable], has_description: bool = False):
    names = []
    values = [] 
    descriptions = []
    for variable in variables:
        names.append(variable.display_name if variable.alternate_display_name is None else variable.alternate_display_name)
        values.append(variable.value_str)
        if has_description:
            descriptions.append(variable.description.Text if variable.description.Text is not None else "")
    data: pd.DataFrame = {"Name": names, "Value": values, "Description": descriptions, } if has_description else {"Name": names, "Value": values}
    column_config={
        "Name": st.column_config.Column(
            None, 
            help="Variable name",
            disabled=True,
            # width="medium"
        ),
        "Value": st.column_config.Column(
            None, 
            help="Variable value",
            disabled=True,
            # width="medium"
        ),
        "Description": st.column_config.Column(
            None, 
            help="Variable description",
            disabled=True,
            # width="large"
        ),
    }
    st.dataframe(data, use_container_width=True, hide_index=True, column_config=column_config)

def update_asset_management(container, device: Device):
    with container.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            state_vars = device.state_machine_variables + device.location_variables
            with st.expander(f"**Overview {device.display_name}**", expanded=True):  
                show_variables_table(state_vars)
        with col2:
            lat = []
            lon = []
            size = []
            color = []
            for dev in device.server.devices:
                location = dev.geographical_location
                if location is not None:
                    lat.append(location[0])
                    lon.append(location[1])
                    size.append(1000 if dev == device else 500)
                    color.append("#ff4400" if dev is device else "#0044ff")
            if len(lat) > 0:
                df = pd.DataFrame({
                    "lat": lat,
                    "lon": lon,
                    "size": size,
                    "color": color,
                    })
                st.map(df, zoom=6, use_container_width=True)

        show_components(device, expanded_count=1)

def show_components(component: Component, expanded_count):
    with st.expander(f"**{component.__class__.__name__} {component.display_name}**", expanded=expanded_count > 0):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Nameplate")
            show_variables_table(component.name_plate_variables)
        with col2:
            if component.operation_counters is not None:
                st.write("Operation Counters")
                show_variables_table(component.operation_counters.variables)
        with col3:
            if len(component.lifetime_counters) > 0:
                st.write("Lifetime Counters")
                show_variables_table(component.lifetime_counters)

    if component.components is not None:
        count = expanded_count
        for sub_component in component.components:
            count = count - 1 
            show_components(sub_component, count)

def update_program_template_set(container, functional_unit: FunctionalUnit):
    with container.container():
        st.write("**Templates**")
        program_manager = functional_unit.program_manager
        if program_manager is None: 
            return
        for program_template in program_manager.program_templates:
            with st.expander(program_template.display_name, expanded=False):
                show_variables_table(program_template.variables)

def show_active_program(container, functional_unit: FunctionalUnit) -> any:
    with container.container():
        st.write("**Active Program**")
        program_manager = functional_unit.program_manager
        if not program_manager is None: 
            with st.form("Start Program"):
                template_id = st.selectbox("Program template", program_manager.program_template_names)
                key_value_df = st.data_editor(pd.DataFrame({"Key": [], "Value": []}, dtype="string"),
                                            column_config={"Key": st.column_config.TextColumn(), 
                                                           "Value": st.column_config.TextColumn()},
                                            num_rows="dynamic", hide_index=True, use_container_width=True)
                job_id = st.text_input("Supervisory job id", value="My Job")
                task_id = st.text_input("Supervisory task id", value="My Task")
                samples_df = st.data_editor(pd.DataFrame({"ContainerId": [], "SampleId": [], "Position": [], "CustomData": []}, dtype="string"),
                                            column_config={"ContainerId": st.column_config.TextColumn(), 
                                                           "SampleId": st.column_config.TextColumn(),
                                                           "Position": st.column_config.TextColumn(),
                                                           "CustomData": st.column_config.TextColumn(),
                                                           },
                                            num_rows="dynamic", hide_index=True, use_container_width=True)
                submitted = st.form_submit_button("Start Program")
    progress_container = st.empty()
    update_active_program(progress_container, functional_unit)
    return progress_container
                        
def update_active_program(container, functional_unit: FunctionalUnit):
    with container.container():
        st.write(f"Current state **{functional_unit.state_machine.current_state.value_str}**")
        program_manager = functional_unit.program_manager
        if program_manager is None: 
            return
        active_program = program_manager.active_program
        if active_program.has_progress:
            st.progress(active_program.current_progress, "Program run progress")
        if active_program.has_step_progress:
            st.progress(active_program.current_step_progress, "Program step progress")
        show_variables_table(active_program.variables)

def update_result_set(container, functional_unit: FunctionalUnit):
    with container.container():
        st.write("**Results**")
        program_manager = functional_unit.program_manager
        if program_manager is None: 
            return
        for result in program_manager.results:
            with st.expander(result.display_name, expanded=False):
                show_variables_table(result.variables)

selectedFunctionalUnitKey = "selected_functional_unit"
lastEventListUpdateKey = "last_event_list_update"

def empty(container):
    container.empty()
    time.sleep(0.02)
    return container

def main():
    my_server = get_server_connection(DefaultServerUrl)
    functional_units = my_server.functional_units

    # session state
    functional_unit_names = list(map(lambda functional_unit: functional_unit.unique_name, functional_units))
    if selectedFunctionalUnitKey not in st.session_state:
        st.session_state[selectedFunctionalUnitKey] = functional_unit_names[0]    
    # in anyway create a new last_event_update on rerun
    st.session_state[lastEventListUpdateKey] = dt.datetime.now()    

    container_functions = None
    
    # functional-unit list on the left side
    st.session_state[selectedFunctionalUnitKey] = st.sidebar.selectbox("Select a functional-unit", functional_unit_names)

    # get selected functional-unit
    selected_functional_unit = functional_units[0]
    selected_functional_unit_name = st.session_state[selectedFunctionalUnitKey]
    for functional_unit in functional_units:
        if functional_unit.unique_name == selected_functional_unit_name:
            selected_functional_unit = functional_unit

    # title
    st.header("LADS OPC UA Client")
    st.write(f"**{selected_functional_unit.unique_name}**")

    tab_functions, tab_program_manager, tab_device = st.tabs(["Operation", "Program Management", "Asset Management"])
    container_functional_unit = st.empty()
    empty(container_functional_unit)
    with container_functional_unit:
        with tab_functions:
            col_functions, col_chart = st.columns([2, 3])

            # Functions list in the detail view
            with col_functions:
                container_functions = empty(st.empty())
                function_containers = show_functions(container_functions, selected_functional_unit)
                # container_functions = st.empty()
                # empty(container_functions)
                # update_functions(container_functions, selected_functional_unit)

            # Chart in the detail view
            with col_chart:
                container_chart = st.empty()

        with tab_program_manager:
            col_templates, col_status, col_results = st.columns([1, 1, 1])
            with col_templates:
                container_templates = empty(st.empty())
                update_program_template_set(container_templates, selected_functional_unit)
            with col_status:
                container_active_program = empty(st.empty())
                progress_container = show_active_program(container_active_program, selected_functional_unit)
            with col_results:
                container_results = empty(st.empty())
                update_result_set(container_results, selected_functional_unit)

        with tab_device:
            container_device = st.empty()
            empty(container_device)
            update_asset_management(container_device, selected_functional_unit.device)

        # Display the events table
        with st.container():
            st.divider()
            st.write("**Events**")
            container_events = st.empty()
            empty(container_events)
            update_events(container_events, selected_functional_unit)
            
        # update loop
        cf = 0.1
        index = 5
        while(True):
            update_functions(function_containers)
            update_events(container_events, selected_functional_unit)
            update_active_program(progress_container, selected_functional_unit)
            update_asset_management(container_device, selected_functional_unit.device)
            index += 1
            if index >= 5:
                index = 0
                update_charts(container_chart, selected_functional_unit, True)
                update_result_set(container_results, selected_functional_unit)
            time.sleep(1)

if __name__ == '__main__':
    main()
