"""Python Streamlit app to visualize LADS OPC UA servers.

This library provides a Streamlit app to visualize the data of LADS OPC UA servers. It uses the LADS OPC UA client to connect to the servers and to read and write data. The app displays the functional units, the functions, the variables, the events, the programs, and the asset management of the servers. The app updates the data in real-time.

Copyright (c) 2023 Dr. Matthias Arnold, AixEngineers, Aachen, Germany.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import streamlit as st
import datetime as dt
import pandas as pd
import time, math
import plotly.graph_objects as go
from typing import Tuple
import lads_opcua_client as lads
from asyncua import ua
import atexit

# For disabling the exit of the threadpool executor
# This is necessary to avoid an error message when the app is closed
# Threads are manually stopped by the atexit in the Connection class
def disable_exit_for_threadpool_executor():
    import concurrent.futures
    atexit.unregister(concurrent.futures.thread._python_exit)
disable_exit_for_threadpool_executor()

st.set_page_config(page_title="LADS OPC UA Client", layout="wide")

@st.cache_resource(show_spinner="Connecting to OPC servers")
def get_server_connections(config_file: str = "config.json") -> lads.Connections:
    connections = lads.Connections(config_file)
    return connections

# MARK: get_initialized_connections
def get_initialized_connections() -> lads.Connections:
    connections = get_server_connections()
    connections.connect()
    urls = ", ".join(connections.urls)
    with st.spinner(f"Initializing OPC connections '{urls}' ..."):
        while not connections.initialized:
            time.sleep(0.1)
        return connections

# MARK: format_value
def format_value(x: float | list[float], decis = 1) -> str:
    result = "NaN"
    try:
        if isinstance(x, list):
            result = f"[{format_number(x[0], decis)} .. {format_number(x[len(x) - 1], decis)}]"
        else:
            result = format_number(x, decis)
    finally:
        return result

# MARK: format_number
def format_number(x: float, decis = 1) -> str:
    result = "NaN"
    try:
        result = "{0:.1f}".format(x)
    finally:
        return result

# MARK: function_state_color
def function_state_color(function: lads.BaseControlFunction) -> str:
    return state_color(function.current_state)

# MARK: variable_status_color
def variable_status_color(variable: lads.BaseVariable, color_good = 'blue') -> str:
    status_code = variable.data_value.StatusCode
    if status_code.is_bad():
        return 'red'
    elif status_code.is_uncertain():
        return 'orange'
    else:
        return color_good

# MARK: state_color
def state_color(current_state: lads.BaseVariable) -> str:
    s = str(current_state.value_str)
    return "green" if "Running" in s else "red" if "Abort" in s else "gray"

# MARK: call_function_state_machine_method
def call_function_state_machine_method(function: lads.BaseStateMachineFunction):
    if function is None: return
    key = function.unique_name
    if not key in st.session_state:
        st.session_state[key] = None
    method_name = st.session_state[key]
    function.state_machine.call_method_by_name(method_name)
    st.session_state[key] = None

# MARK: call_state_machine_method
def call_state_machine_method(state_machine: lads.StateMachine):
    key = state_machine.nodeid
    if not key in st.session_state:
        st.session_state[key] = None
    method_name = st.session_state[key]
    state_machine.call_method_by_name(method_name)
    st.session_state[key] = None

# MARK: write_variable_value
def write_variable_value(variable: lads.BaseVariable):
    if variable is None: 
        return
    key = variable.nodeid
    if not key in st.session_state: 
        return
    variable.set_value(st.session_state[key])
    st.session_state[key] = None

# MARK: write_discrete_variable_value
def write_discrete_variable_value(variable: lads.DiscreteVariable):
    if variable is None: 
        return
    key = variable.nodeid
    if not key in st.session_state: 
        return
    variable.set_value_from_str(st.session_state[key])
    st.session_state[key] = None

# MARK: add_variable_value_input
def add_variable_value_input(variable: lads.BaseVariable, parent: lads.LADSNode = None):
    if variable is None:
        return
    if variable.has_write_access:
        help = f"{variable.display_name}: {variable.description.Text}"
        if parent is not None:
            help = f"{parent.display_name}.{help}"
        st.number_input(variable.display_name, on_change = write_variable_value(variable), value=None, placeholder=str(variable.value), key=variable.nodeid, label_visibility="collapsed", help = help)

# MARK: show_functions
def show_functions(container, functional_unit: lads.FunctionalUnit) -> dict:
    functions_container = container.container()
    function_containers = {}
    show_function_set(functions_container, path="", function_set=functional_unit.function_set, container_dict=function_containers)
    update_functions(function_containers)
    return function_containers

# MARK: show_function_set
def show_function_set(container, path: str, function_set: lads.FunctionSet, container_dict: dict):
     with container:
        index = 0
        for function in function_set.functions:
            if function.is_enabled:
                index += 1
                s = function.display_name if path == "" else "/".join([path, function.display_name])
                label = f"**{s}**"
                with st.expander(label=label, expanded=(path == "") and (index < 10)):
                    col_static, col_sp, col_pv = st.columns([4, 4, 5])
                    with col_static:
                        if isinstance(function, lads.BaseStateMachineFunction):
                            if isinstance(function, lads.AnalogControlFunction):
                                add_variable_value_input(function.target_value, function)
                            elif isinstance(function, lads.MulitModeControlFunction):
                                for controller_parameter in function.controller_parameters:
                                    add_variable_value_input(controller_parameter.target_value, controller_parameter)
                                current_mode = function.current_mode
                                mode = st.selectbox(label=current_mode.display_name, on_change=write_variable_value(current_mode), options=function.modes, label_visibility="collapsed", key=current_mode.nodeid, placeholder="Choose a mode", help=current_mode.dictionary_entries_as_markdown)
                            elif isinstance(function, lads.DiscreteControlFunction):
                                target_value = function.target_value
                                cmd = st.selectbox(label="Command", options=target_value.values_as_str, index=None, label_visibility="collapsed", key=target_value.nodeid, on_change=write_discrete_variable_value(target_value), placeholder="Choose a value", help=target_value.dictionary_entries_as_markdown)
                            method_names = function.state_machine.method_names
                            if len(method_names) > 0:
                                key = function.unique_name
                                cmd = st.selectbox(label="Command", options=method_names, index=None, label_visibility="collapsed", key=key, on_change=call_function_state_machine_method(function), placeholder="Choose a command", help=function.state_machine.dictionary_entries_as_markdown)
                    with col_sp:
                        container_sp = st.empty()
                    with col_pv:
                        container_pv = st.empty()
                    container_dict[function] = (container_sp, container_pv)

            if function.function_set is not None:
                show_function_set(container, s, function_set=function.function_set, container_dict=container_dict)

# MARK: update_functions
def update_functions(function_containers: dict):
    for function, containers in function_containers.items():
        container_sp, container_pv = containers
        sp_col = container_sp.container()
        pv_col = container_pv.container()
        if isinstance(function, lads.TimerControlFunction):
            color = function_state_color(function)
            with sp_col:
                if function.target_value is not None:
                    st.markdown(f":{color}[**{format_value(0.001 * function.target_value.value)}** s]", help=function.target_value.dictionary_entries_as_markdown)
            with pv_col: 
                if function.current_value is not None:
                    st.markdown(f":blue[**{format_value(0.001 * function.current_value.value)}** s]", help=function.current_value.dictionary_entries_as_markdown)
                if function.difference_value is not None:
                    st.markdown(f":blue[**{format_value(0.001 * function.difference_value.value)}** s]", help=function.difference_value.dictionary_entries_as_markdown)
        elif isinstance(function, lads.AnalogControlFunction):
            color = function_state_color(function)
            with sp_col:
                st.markdown(f":{color}[**{format_value(function.target_value.value)}** {function.target_value.eu}]", help=function.target_value.dictionary_entries_as_markdown)
                if isinstance(function, lads.AnalogControlFunctionWithTotalizer):
                    st.markdown(":gray[Totalizer]")
            with pv_col: 
                st.markdown(f":{variable_status_color(function.current_value)}[**{format_value(function.current_value.value)}** {function.current_value.eu}]", help=function.current_value.dictionary_entries_as_markdown)
                if isinstance(function, lads.AnalogControlFunctionWithTotalizer):
                    st.markdown(f":blue[**{format_value(function.totalized_value.value)}** {function.totalized_value.eu}]", help=function.totalized_value.dictionary_entries_as_markdown)
        elif isinstance(function, lads.TwoStateDiscreteControlFunction) or isinstance(function, lads.MultiStateDiscreteControlFunction) :
            color = function_state_color(function)
            with sp_col:
                st.markdown(f":{color}[**{function.target_value.value_str}**]", help=function.target_value.dictionary_entries_as_markdown)
            with pv_col: 
                st.markdown(f":{variable_status_color(function.current_value)}[**{function.current_value.value_str}**]", help=function.current_value.dictionary_entries_as_markdown)
        elif isinstance(function, lads.AnalogScalarSensorFunction):
            if isinstance(function, lads.AnalogScalarSensorFunctionWithCompensation):
                if function.compensation_value is not None:
                    with sp_col:
                        st.markdown(f":gray[{format_value(function.compensation_value.value)} {function.compensation_value.eu}]", help=function.compensation_value.dictionary_entries_as_markdown)
            with pv_col:
                st.markdown(f":{variable_status_color(function.sensor_value)}[**{format_value(function.sensor_value.value)}** {function.sensor_value.eu}]", help=function.sensor_value.dictionary_entries_as_markdown)

        elif isinstance(function, lads.TwoStateDiscreteSensorFunction) or isinstance(function, lads.MultiStateDiscreteSensorFunction):
            with pv_col: 
                st.markdown(f":{variable_status_color(function.sensor_value)}[**{function.sensor_value.value_str}**]", help=function.sensor_value.dictionary_entries_as_markdown)
        elif isinstance(function, lads.CoverFunction):
            with pv_col: 
                st.markdown(f":{variable_status_color(function.current_state)}[**{function.current_state.value_str}**]", help=function.current_state.dictionary_entries_as_markdown)
        elif isinstance(function, lads.StartStopControlFunction):
            with pv_col: 
                st.markdown(f":{function_state_color(function)}[**{function.current_state.value_str}**]", help=function.current_state.dictionary_entries_as_markdown)
        elif isinstance(function, lads.MulitModeControlFunction):
            for controller_parameter in function.controller_parameters:
                with sp_col:
                    st.markdown(f":{function_state_color(function)}[**{format_value(controller_parameter.target_value.value)}** {controller_parameter.target_value.eu}]", help=controller_parameter.target_value.dictionary_entries_as_markdown)
                with pv_col: 
                    st.write(f":{variable_status_color(controller_parameter.current_value)}[**{format_value(controller_parameter.current_value.value)}** {controller_parameter.current_value.eu}]")

# MARK: add_chart_items
def add_chart_items(functions: list[lads.Function], traces: list, arrays: list):
    for function in functions:
        analog_item: lads.AnalogItem = None
        if isinstance(function, lads.AnalogControlFunction):
            analog_item = function.current_value
        elif isinstance(function, lads.AnalogScalarSensorFunction):
            analog_item = function.sensor_value
        if analog_item is not None:
            if isinstance(analog_item.value, list):
                arrays.append((function, analog_item))
            elif analog_item.history is not None:
                traces.append((function, analog_item))
        # recurse sub-functions
        if function.function_set is not None:
            add_chart_items(function.function_set.functions, traces=traces, arrays=arrays)

# MARK: update_charts
def update_charts(container, functional_unit: lads.FunctionalUnit, use_plotly=True):
    with container.container():        
        # collect analog items with history and arrays
        traces: list[Tuple[lads.Function, lads.AnalogItem]] = []
        arrays: list[Tuple[lads.Function, lads.AnalogItem]] = []
        idx = 0

        add_chart_items(functional_unit.functions, traces=traces, arrays=arrays)

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
                    title = dict(
                        text = f"{function.display_name} [{analog_item.eu}]",
                        font = dict(
                            color = color
                        )
                    )
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
                unique_key = f"plotly_chart_{time.time_ns()}"
                st.plotly_chart(fig, use_container_width=True, key=unique_key)
                idx += 1
        else:
            for trace in traces:
                function, analog_item = trace
                with st.expander(f"**Chart {function.display_name}**", expanded=(idx==0)):
                    unique_key = f"line_chart_{time.time_ns()}"
                    st.line_chart(data = analog_item.history, height = 100, key=unique_key)
                    idx += 1

        # tables
        for array in arrays:
            function, analog_item = array
            value = analog_item.value
            i = len(value)
            col_count = int(math.sqrt(3 / 2 * i))
            row_count = int(2 / 3 * col_count)
            isPlate = col_count * row_count == i
            with st.expander(f"**{function.display_name}**", expanded=(idx<=2)):
                if isPlate:
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
                        try:
                            df.style.background_gradient(
                                axis=None, 
                                vmin=eu_range.Low, 
                                vmax=eu_range.High,
                                cmap="jet"
                            )
                        except AttributeError:
                            ' do nothing'
                    st.dataframe(
                        df,
                        use_container_width=True, 
                        hide_index=True,
                    )
                else:
                    eu = analog_item.engineering_units
                    col = eu.DisplayName.Text if eu is not None else "y"
                    df = pd.DataFrame({col: value})
                    unique_key = f"area_chart_{time.time_ns()}"
                    st.area_chart(df, key=unique_key)
            idx += 1

# MARK: update_events
def update_events(container, device: lads.Device):
    events = device.subscription_handler.events
    if events is None:
        return
    if lastEventListUpdateKey not in st.session_state:
        st.session_state[lastEventListUpdateKey] = dt.datetime(2020, 1, 1)    
    last_event_update = device.subscription_handler.last_event_update
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

# MARK: show_variables_table
def show_variables_table(variables: list[lads.BaseVariable], has_description: bool = False):
    names = []
    values = [] 
    descriptions = []
    for variable in variables:
        names.append(variable.display_name)
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

# MARK: show_asset_management
def show_asset_management(container, device: lads.Device):
    device_state_machine = device.device_state
    device_state_methods = device_state_machine.method_names
    operation_mode_state_machine = device.machinery_operation_mode
    operation_mode_methods = [] if operation_mode_state_machine is None else operation_mode_state_machine.method_names

    with container.container():
        col_device, col_map = st.columns([1, 2])
        with col_device:
            if len(device_state_methods) > 0:
                device_method = st.selectbox(label="Device control", options=device_state_methods, index=None, key=device_state_machine.nodeid, on_change=call_state_machine_method(device_state_machine))
            if len(operation_mode_methods) > 0:
                operation_mode_method = st.selectbox(label="Operation mode", options=operation_mode_methods, index=None, key=operation_mode_state_machine.nodeid, on_change=call_state_machine_method(operation_mode_state_machine))
            state_vars = device.state_machine_variables + device.location_variables
            container_device = st.empty()
        with col_map:
            container_map = st.empty()
        container_components = st.empty()

    update_asset_management(container_device, container_map, container_components, device)
    return container_device, container_map, container_components

# MARK: update_asset_management
def update_asset_management(container_device, container_map, container_components, device: lads.Device):
    with container_device:
        state_vars = device.state_machine_variables + device.location_variables
        with st.expander(f"**Status {device.display_name}**", expanded=True): 
            show_variables_table(state_vars)
    with container_map:
        lat = []
        lon = []
        size = []
        color = []
        for dev in device.server.devices:
            location = dev.geographical_location
            if location is not None:
                lat.append(location[0])
                lon.append(location[1])
                size.append(10000 if dev == device else 5000)
                color.append("#ff4400" if dev is device else "#0044ff")
        if len(lat) > 0:
            df = pd.DataFrame({
                "lat": lat,
                "lon": lon,
                "size": size,
                "color": color,
                })
            st.map(df, zoom=8, use_container_width=True)
    with container_components.container():
        show_components(device, expanded_count=1)

# MARK: show_components
def show_components(component: lads.Component, expanded_count):
    with st.expander(f"**{component.__class__.__name__} {component.display_name}**", expanded=expanded_count > 0):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("Nameplate", help=component.dictionary_entries_as_markdown)
            show_variables_table(component.name_plate_variables)
        with col2:
            if component.operation_counters is not None:
                st.write("Operation Counters")
                show_variables_table(component.operation_counters.variables)
        with col3:
            if len(component.lifetime_counters) > 0:
                st.write("Lifetime Counters")
                # show_variables_table(component.lifetime_counters)
                for counter in component.lifetime_counters:
                    try:
                        value = to_float(counter.value, default=0.0)
                        start = to_float(counter.start_value.value, default=0.0)
                        limit = to_float(counter.limit_value.value, default=100.0)
                        eu = counter.eu
                        warning = False if counter.warning_values is None else any(value < float(warning_value) for warning_value in counter.warning_values.value)
                        color = "red" if warning else "green"
                        s = f"{counter.display_name} [{format_number(start)} > :{color}[**{format_number(value)}**] > {format_number(limit)}] {eu}"
                        r = start - limit
                        x = value / r + limit if abs(r) > 0 else 0
                        st.progress(x, s)  
                    except Exception as e:
                        pass

    if component.components is not None:
        count = expanded_count
        for sub_component in component.components:
            count = count - 1 
            show_components(sub_component, count)

# MARK: show_program_template_set
def show_program_template_set(container, functional_unit: lads.FunctionalUnit):
    with container.container():
        st.markdown("**Templates**")
        program_manager = functional_unit.program_manager
        if program_manager is None: 
            return
        for program_template in program_manager.program_templates:
            with st.expander(program_template.display_name, expanded=False):
                definitions = program_template.dictionary_entries_as_markdown
                if len(definitions) > 0:
                    st.markdown(":gray[Definitions]", help=program_template.dictionary_entries_as_markdown)
                show_variables_table(program_template.variables)

# MARK: show_state
def show_state(container, functional_unit: lads.FunctionalUnit) -> any:
    container_state = container.empty()
    update_state(container_state, functional_unit)
    return container_state

# MARK: update_state
def update_state(container, functional_unit: lads.FunctionalUnit) -> bool:
    current_state_var = functional_unit.current_state
    with container:
        st.markdown(f":{state_color(current_state_var)}[**{current_state_var.value_str}**]", help=current_state_var.dictionary_entries_as_markdown)
    return False

# MARK: show_active_program
def show_active_program(container, functional_unit: lads.FunctionalUnit) -> any:
    current_state_var = functional_unit.current_state
    st.session_state[current_state_var.nodeid] = "Init"

    with container.container():
        st.write("**Active Program**")
    program_manager = functional_unit.program_manager
    if not program_manager is None: 
        form_container = empty(st.empty())
        with st.form("Start Program"):
            template_id = st.selectbox("Program template", program_manager.program_template_names)
            with st.expander("Properties", expanded=False):
                key_value_df = st.data_editor(pd.DataFrame({"Key": ["My Property"], "Value": ["42.0"]}, dtype="string"),
                                            column_config={"Key": st.column_config.TextColumn(), 
                                                            "Value": st.column_config.TextColumn()},
                                            num_rows="dynamic", hide_index=True, use_container_width=True)
            job_id = st.text_input("Supervisory job id", value="My Job")
            task_id = st.text_input("Supervisory task id", value="My Task")
            with st.expander("Samples", expanded=False):
                samples_df = st.data_editor(pd.DataFrame({"ContainerId": ["0815", "0815"], "SampleId": ["4711", "4712"], "Position": ["A1", "A2"], "CustomData": ["", ""]}, dtype="string"),
                                            column_config={"ContainerId": st.column_config.TextColumn(), 
                                                            "SampleId": st.column_config.TextColumn(),
                                                            "Position": st.column_config.TextColumn(),
                                                            "CustomData": st.column_config.TextColumn(),
                                                            },
                                            num_rows="dynamic", hide_index=True, use_container_width=True)
            if st.form_submit_button("Start Program", help="**Start's a program run** based on the selected program-template.  \r\nIf you want to manually stop or abort the current run, utilize the functional-unit's commands (drop-down next to the unit's state indicator)."):
                    functional_unit.functional_unit_state.start_program(template_id, key_value_df, job_id, task_id, samples_df)

    progress_container = empty(st.empty())
    update_active_program(progress_container, functional_unit)
    return progress_container

# MARK: to_float
def to_float(value, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except:
        result = default
    return result

# MARK: update_active_program
def update_active_program(progress_container, functional_unit: lads.FunctionalUnit) -> bool:
    current_state_var = functional_unit.current_state
    current_state = current_state_var.value_str
    previous_state = st.session_state[current_state_var.nodeid]
    st.session_state[current_state_var.nodeid] = current_state
    state_changed = current_state != previous_state

    running = "Running" in current_state
    program_manager = functional_unit.program_manager

    with progress_container.container():
        if program_manager is not None: 
            if state_changed or running:
                active_program = program_manager.active_program
                if active_program.has_progress:
                    st.progress(active_program.current_progress, "Program run progress")
                    st.write(f"{format_number(0.001 * to_float(active_program.current_runtime.value))}s / {format_number(0.001 * to_float(active_program.estimated_runtime.value))}s")
                if active_program.has_step_progress:
                    step_name = active_program.current_step_name
                    label = "Program step progress" if step_name is None else f"Program step '{step_name.value_str}' progress" 
                    st.progress(active_program.current_step_progress, text=label)
                    st.write(f"{format_number(0.001 * to_float(active_program.current_step_runtime.value))}s / {format_number(0.001 * to_float(active_program.estimated_step_runtime.value))}s")
                with st.expander("Program run details", expanded=False):
                    show_variables_table(active_program.variables)

    # update result once run is finished
    if False and "Stopped" in current_state and "Running" in previous_state:
        try:
            program_manager.results[-1].update()
        except:
            pass

# MARK: update_result_set
def update_result_set(container, functional_unit: lads.FunctionalUnit):
    with container.container():
        st.write("**Results**")
        program_manager = functional_unit.program_manager
        if program_manager is None: 
            return
        for result in program_manager.results:
            with st.expander(result.display_name, expanded=False):
                definitions = result.dictionary_entries_as_markdown
                if len(definitions) > 0:
                    st.markdown(":gray[Definitions]", help=result.dictionary_entries_as_markdown)

                # show result type specific variables
                show_variables_table(result.variables)

                # show result files, if any
                for result_file in result.result_files:
                    st.markdown(f"Result file {result_file.display_name}", help=result_file.dictionary_entries_as_markdown)
                    show_variables_table(result_file.variables)
                    unique_key = f"button_{time.time_ns()}"
                    if result_file.has_data():
                        file_name = result_file.name.value_str
                        mime_type = result_file.mime_type.value_str
                        data = result_file.data
                        st.download_button("Download", data=data, file_name=file_name, mime=mime_type, key=unique_key)
                    else:
                        st.button("Fetch data", key=unique_key, on_click=result_file.fetch_data())
                
                # show application specific variables, if any
                if len(result.variable_set.variables) > 0:
                    # collect dictionary entries while avoiding duplicates but preserving order
                    entries = list()
                    entry_set = set()
                    for variable in result.variable_set.variables:
                        entry = variable.dictionary_entries_as_markdown
                        if entry not in entry_set:
                            entries.append(entry)
                            entry_set.add(entry)
                    st.markdown("Result data", help="\n\r".join(entries))
                    show_variables_table(result.variable_set.variables)

selectedFunctionalUnitKey = "selected_functional_unit"
lastEventListUpdateKey = "last_event_list_update"

# MARK: empty
def empty(container):
    container.empty()
    time.sleep(0.02)
    return container

# MARK: ConnectionsManager
class ConnectionsManager:
    def __init__(self):
        self.connections = None

    def set_connections(self, connections):
        self.connections = connections

    def disconnect_all(self):
        self.connections.disconnect()

# MARK: Global ConnectionsManager
connections_manager = ConnectionsManager()

# MARK: main
def main():
    # @DrMatthiasArnold at atexit there is an attempt to disconnect all servers
    atexit.register(connections_manager.disconnect_all)

    connections = get_initialized_connections()

    # Set the connections for disconnecting all at exit and possible other uses
    # @DrMatthiasArnold we could also make the ConnectionManager asyncronous
    # to enable adding new connections and in general, to enable runnig the app asyncronously
    # based on an initial configuration file
    connections_manager.set_connections(connections)

    functional_units = connections.functional_units

    # session state
    functional_unit_names = list(map(lambda functional_unit: functional_unit.at_name, functional_units))
    functional_unit_names.sort()
    if selectedFunctionalUnitKey not in st.session_state:
        st.session_state[selectedFunctionalUnitKey] = functional_unit_names[0]    
    # in anyway create a new last_event_update on rerun
    st.session_state[lastEventListUpdateKey] = dt.datetime.now()    

    # functional-unit list on the left side
    st.session_state[selectedFunctionalUnitKey] = st.sidebar.selectbox("Select a functional-unit", functional_unit_names)

    # get selected functional-unit
    selected_functional_unit = functional_units[0]
    selected_functional_unit_name = st.session_state[selectedFunctionalUnitKey]
    for functional_unit in functional_units:
        if functional_unit.at_name == selected_functional_unit_name:
            selected_functional_unit = functional_unit

    # title
    st.subheader("LADS OPC UA Client", help=selected_functional_unit.device.dictionary_entries_as_markdown)
    
    with st.expander(f"**{selected_functional_unit.at_name}**", expanded=True):
        col_cmd, col_state, col_definition = st.columns([2, 3, 2])
        with col_cmd:
            state_machine = selected_functional_unit.functional_unit_state
            methods = list(filter(lambda method: method != "StartProgram", state_machine.method_names ))
            cmd = st.selectbox("Command", options=methods, index=None, label_visibility="collapsed", placeholder="Choose a command")
            if cmd == "Start":
                state_machine.start(pd.DataFrame())
            else:
                state_machine.call_method_by_name(cmd)
        with col_state:
            container_state = show_state(col_state, selected_functional_unit)
        with col_definition:
            definition = selected_functional_unit.dictionary_entries_as_markdown
            if len(definition) > 0:
                st.markdown(":gray[Definitions]", help = definition)

    tab_functions, tab_program_manager, tab_device = st.tabs(["Operation", "Program Management", "Asset Management"])
    container_functional_unit = st.empty()
    empty(container_functional_unit)
    with container_functional_unit:
        with tab_functions:
            col_functions, col_chart = st.columns([2, 3])

            # Functions list in the detail view
            with col_functions:
                function_containers = show_functions(empty(st.empty()), selected_functional_unit)
            # Chart in the detail view
            with col_chart:
                container_chart = st.empty()

        with tab_program_manager:
            col_templates, col_status, col_results = st.columns([1, 1, 1])
            with col_templates:
                show_program_template_set(empty(st.empty()), selected_functional_unit)
            with col_status:
                progress_container = show_active_program(empty(st.empty()), selected_functional_unit)
            with col_results:
                container_results = empty(st.empty())
                update_result_set(container_results, selected_functional_unit)

        with tab_device:
            container_device, container_map, container_components = show_asset_management(empty(st.empty()), selected_functional_unit.device)

        # Display the events table
        with st.container():
            st.divider()
            st.write("**Events**")
            container_events = st.empty()
            empty(container_events)
            update_events(container_events, selected_functional_unit.device)
            
        def update_loop():
            index = 5
            while(True):
                update_state(container_state, selected_functional_unit)
                update_functions(function_containers)
                update_events(container_events, selected_functional_unit)
                update_active_program(progress_container, selected_functional_unit)
                update_asset_management(container_device, container_map, container_components, selected_functional_unit.device)
                index += 1
                if index >= 5:
                    index = 0
                    update_charts(container_chart, selected_functional_unit, True)
                    update_result_set(container_results, selected_functional_unit)
                time.sleep(1)

        update_loop()

if __name__ == '__main__':
    main()
