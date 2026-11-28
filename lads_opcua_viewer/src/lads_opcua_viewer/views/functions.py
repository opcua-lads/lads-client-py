"""View of the functions of a functional unit."""

import lads_opcua_client as lads
import streamlit as st
from lads_opcua_viewer.callbacks import add_variable_value_input, call_function_state_machine_method, modify_target_value_by, write_discrete_variable_value, write_variable_value
from lads_opcua_viewer.colors import function_state_color, variable_status_color
from lads_opcua_viewer.formatting import format_analog_item, format_value, help_markdown


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
                                if isinstance(function, lads.AnalogControlFunctionWithComposedTargetValue):
                                    # the partial target values (TargetValueSet), e.g. Flow and PercentageSolventB of a pump
                                    for partial in function.target_value_set:
                                        add_variable_value_input(partial, function, show_label=True)
                                if isinstance(function, lads.AnalogControlFunctionWithRelativeTargetValue):
                                    # rates (e.g. aspirating / dispensing rate) and the relative change of the target value
                                    for rate in function.rates:
                                        add_variable_value_input(rate, function, show_label=True)
                                    if function.has_modify_target_value_by:
                                        key = f"{function.unique_name}.ModifyTargetValueBy"
                                        st.number_input(f"Change by [{function.target_value.eu}]", value=None, key=key, on_change=modify_target_value_by(function, key),
                                                        help="ModifyTargetValueBy: changes the target value by this amount (positive or negative)")
                            elif isinstance(function, lads.MulitModeControlFunction):
                                for controller_parameter in function.controller_parameters:
                                    add_variable_value_input(controller_parameter.target_value, controller_parameter)
                                current_mode = function.current_mode
                                mode = st.selectbox(label=current_mode.display_name, on_change=write_variable_value(current_mode), options=function.modes, label_visibility="collapsed", key=current_mode.nodeid, placeholder="Choose a mode", help=help_markdown(current_mode, function))
                            elif isinstance(function, lads.DiscreteControlFunction):
                                target_value = function.target_value
                                cmd = st.selectbox(label="Command", options=target_value.values_as_str, index=None, label_visibility="collapsed", key=target_value.nodeid, on_change=write_discrete_variable_value(target_value), placeholder="Choose a value", help=help_markdown(target_value, function))
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
def get_alarm_indicator(function: lads.AnalogScalarSensorFunction | lads.AnalogControlFunction) -> str:
    alarm_indicator = ""
    if function.has_alarm_monitor:
        alarm_indicator = "🔹"
        if function.alarm_active:
            limit: str = function.alarm_limit
            match limit:
                case "High":
                    alarm_indicator = "🔺"
                case "HighHigh":
                    alarm_indicator = "🔺🔺"
                case "Low":
                    alarm_indicator = "🔻"
                case "LowLow":
                    alarm_indicator = "🔻🔻"
                case _:
                    alarm_indicator = "⚠️"
    return alarm_indicator


def get_alarm_limits(function: lads.AnalogScalarSensorFunction | lads.AnalogControlFunction) -> str:


    alarm_monitor: lads.AlarmMonitor = function.alarm_monitor
    if alarm_monitor is None:
        return ""
    current_value_item = function.current_value if isinstance(function, lads.AnalogControlFunction) else function.sensor_value
    target_value_item = function.target_value if isinstance(function, lads.AnalogControlFunction) else None
    target_value = float(target_value_item.value) if target_value_item is not None else None
    eu = current_value_item.eu
    decis = current_value_item.decimals()

    def tryAppend(l: list, variable: lads.BaseVariable):
        if variable is not None:
            value = variable.value
            scale = 0.001 if eu == "s" else 1.0
            if target_value is None:
                l.append(f"{variable.display_name} = {format_value(scale * value, decis)} {eu}")
            else:
                l.append(f"{variable.display_name} = {format_value(scale * value, decis)} {eu} ({format_value(scale * (value + target_value), decis)} {eu})")

    l = []
    tryAppend(l, alarm_monitor.high_high_limit)
    tryAppend(l, alarm_monitor.high_limit)
    tryAppend(l, alarm_monitor.low_limit)
    tryAppend(l, alarm_monitor.low_low_limit)
    if len(l) == 0:
        return ""
    l.insert(0, "**Alarm limits**")
    return  "  \r\n".join(l)


def update_functions(function_containers: dict):
    for function, containers in function_containers.items():
        container_sp, container_pv = containers
        sp_col = container_sp.container()
        pv_col = container_pv.container()
        if isinstance(function, lads.TimerControlFunction):
            color = function_state_color(function)
            with sp_col:
                if function.target_value is not None:
                    st.markdown(f":{color}[**{format_analog_item(function.target_value)}** s]", help=help_markdown(function.target_value, function))
            with pv_col:
                if function.current_value is not None:
                    st.markdown(f":blue[**{format_analog_item(function.current_value)}** s]", help=help_markdown(function.current_value, function))
                if function.difference_value is not None:
                    st.markdown(f":blue[**{format_analog_item(function.difference_value)}** s]", help=help_markdown(function.difference_value, function))
        elif isinstance(function, lads.AnalogControlFunction):
            color = function_state_color(function)
            with sp_col:
                variable = function.target_value
                st.markdown(f":{color}[**{format_value(variable.value, variable.default_decimals)}** {variable.eu}]", help=help_markdown(variable, function))
                if isinstance(function, lads.AnalogControlFunctionWithTotalizer):
                    st.markdown(":gray[Totalizer]")
                if isinstance(function, lads.AnalogControlFunctionWithRelativeTargetValue):
                    for rate in function.rates:
                        st.markdown(f":{color}[{rate.display_name}: **{format_value(rate.value, rate.default_decimals)}** {rate.eu}]", help=help_markdown(rate, function))
                if isinstance(function, lads.AnalogControlFunctionWithComposedTargetValue):
                    for partial in function.target_value_set:
                        eu = getattr(partial, "eu", "")
                        decimals = getattr(partial, "default_decimals", 2)
                        value = format_value(partial.value, decimals) if isinstance(partial.value, (int, float)) else partial.value_str
                        st.markdown(f":{color}[{partial.display_name}: **{value}** {eu}]", help=help_markdown(partial, function))
            with pv_col:
                variable = function.current_value
                help = help_markdown(variable, function, get_alarm_limits(function))
                st.markdown(f":{variable_status_color(variable)}[**{format_value(variable.value, variable.default_decimals)}** {variable.eu} {get_alarm_indicator(function)}]", help=help)
                if isinstance(function, lads.AnalogControlFunctionWithTotalizer):
                    variable = function.totalized_value
                    st.markdown(f":blue[**{format_value(variable.value, variable.default_decimals)}** {variable.eu}]", help=help_markdown(variable, function))
        elif isinstance(function, lads.TwoStateDiscreteControlFunction) or isinstance(function, lads.MultiStateDiscreteControlFunction) :
            color = function_state_color(function)
            with sp_col:
                st.markdown(f":{color}[**{function.target_value.value_str}**]", help=help_markdown(function.target_value, function))
            with pv_col:
                st.markdown(f":{variable_status_color(function.current_value)}[**{function.current_value.value_str}**]", help=help_markdown(function.current_value, function))
        elif isinstance(function, lads.AnalogScalarSensorFunction):
            if isinstance(function, lads.AnalogScalarSensorFunctionWithCompensation):
                if function.compensation_value is not None:
                    with sp_col:
                        variable = function.compensation_value
                        st.markdown(f":gray[{format_value(variable.value, variable.default_decimals)} {variable.eu}]", help=help_markdown(variable, function))
            with pv_col:
                variable = function.sensor_value
                help = help_markdown(variable, function, get_alarm_limits(function))
                st.markdown(f":{variable_status_color(variable)}[**{format_analog_item(variable)}** {variable.eu} {get_alarm_indicator(function)}]", help=help)

        elif isinstance(function, lads.TwoStateDiscreteSensorFunction) or isinstance(function, lads.MultiStateDiscreteSensorFunction):
            with pv_col:
                st.markdown(f":{variable_status_color(function.sensor_value)}[**{function.sensor_value.value_str}**]", help=help_markdown(function.sensor_value, function))
        elif isinstance(function, lads.CoverFunction):
            with pv_col:
                st.markdown(f":{variable_status_color(function.current_state)}[**{function.current_state.value_str}**]", help=help_markdown(function.current_state, function))
        elif isinstance(function, lads.StartStopControlFunction):
            with pv_col:
                st.markdown(f":{function_state_color(function)}[**{function.current_state.value_str}**]", help=help_markdown(function.current_state, function))
        elif isinstance(function, lads.MulitModeControlFunction):
            for controller_parameter in function.controller_parameters:
                with sp_col:
                    variable = controller_parameter.target_value
                    st.markdown(f":{function_state_color(function)}[**{format_value(variable.value, variable.default_decimals)}** {variable.eu}]", help=help_markdown(variable, function))
                with pv_col:
                    variable = controller_parameter.current_value
                    st.write(f":{variable_status_color(variable)}[**{format_value(variable.value, variable.default_decimals)}** {variable.eu}]")
