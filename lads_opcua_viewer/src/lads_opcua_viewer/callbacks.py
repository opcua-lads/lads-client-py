"""Callbacks of Streamlit widgets: calling methods and writing values."""

import lads_opcua_client as lads
import streamlit as st
from lads_opcua_viewer.formatting import to_float


# MARK: call_function_state_machine_method
def call_function_state_machine_method(function: lads.BaseStateMachineFunction):
    if function is None: return
    key = function.unique_name
    if not key in st.session_state:
        st.session_state[key] = None
    method_name = st.session_state[key]
    st.session_state[key] = None
    if method_name is None:
        return
    if method_name.lower().startswith("startwithtarget"):
        variable: lads.AnalogItem = function.target_value
        if variable is not None:
            value = to_float(variable.value)
            st.toast(f"{function.display_name}.StartWitTargetValue({value})")
            function.state_machine.start_with_target_value(value)
            return
    function.state_machine.call_method_by_name(method_name)
    st.session_state[key] = None


# MARK: call_state_machine_method
@st.dialog("Provide target value", dismissible=True)
def get_target_value(name: str):
    st.session_state.target_value = None
    msg = f"Provide target value for function" if name is None else f"Provide target value for function '{name}'"
    st.write(msg)
    value = st.text_input("Target value")
    if st.button("Submit"):
        st.session_state.target_value = value
        st.rerun()


def call_state_machine_method(state_machine: lads.StateMachine):
    key = state_machine.nodeid
    if not key in st.session_state:
        st.session_state[key] = None
    method_name: str = st.session_state[key]
    if method_name is None:
        return
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


# MARK: modify_target_value_by
def modify_target_value_by(function, key: str):
    """on_change of the 'Change by' input: calls ModifyTargetValueBy once with the entered value"""
    value = st.session_state.get(key)
    if value is None:
        return
    function.modify_target_value_by(value)
    st.toast(f"{function.display_name}.ModifyTargetValueBy({value})")
    st.session_state[key] = None


# MARK: add_variable_value_input
def add_variable_value_input(variable: lads.BaseVariable, parent: lads.LADSNode = None, show_label: bool = False):
    if variable is None:
        return
    if variable.has_write_access:
        help = f"{variable.display_name}: {variable.description.Text}"
        if parent is not None:
            help = f"{parent.display_name}.{help}"
        eu = getattr(variable, "eu", "")
        label = f"{variable.display_name} [{eu}]" if eu else variable.display_name
        st.number_input(label, on_change = write_variable_value(variable), value=None, placeholder=str(variable.value), key=variable.nodeid, label_visibility="visible" if show_label else "collapsed", help = help)
