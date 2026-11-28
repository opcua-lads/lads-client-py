"""View of the state, locking and program templates of a functional unit."""

import lads_opcua_client as lads
import streamlit as st
from lads_opcua_viewer.colors import state_color
from lads_opcua_viewer.formatting import format_number
from lads_opcua_viewer.views.tables import show_variables_table


# MARK: show_program_template_set
def show_program_template_set(container, functional_unit: lads.FunctionalUnit):
    with container.container():
        st.markdown("**Templates**")
        program_manager = functional_unit.program_manager
        if program_manager is None:
            return
        for program_template in program_manager.program_templates:
            with st.expander(program_template.unique_name, expanded=False):
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
def update_state(container, functional_unit: lads.FunctionalUnit):
    current_state_var = functional_unit.current_state
    current_state_str = functional_unit.current_state_str
    with container:
        message = f":{state_color(current_state_var)}[**{current_state_str}**]"
        if str(current_state_var.value_str) == "Running":
            program_manager = functional_unit.program_manager
            if program_manager is not None:
                active_program = functional_unit.program_manager.active_program
                if active_program is not None:
                    if active_program.current_program_template is not None:
                        message = message + f"  \nProgram template: {active_program.current_program_template.value_str}"
                        if active_program.current_step_name is not None:
                            message = message + f"  \nProgram step: {active_program.current_step_name.value_str}"
        st.markdown(message, help=current_state_var.dictionary_entries_as_markdown)


# MARK: update_lock
def show_lock_cmd(container_cmd, functional_unit: lads.FunctionalUnit):
    lock = functional_unit.lock
    if lock is None:
        return
    with container_cmd:
        cmd = st.selectbox("Lock command", options=lads.Lock.COMMANDS, index=None, label_visibility="collapsed", placeholder="Choose a lock command")
        lock.call_method_by_name(cmd)


def update_lock_state(container_state, functional_unit: lads.FunctionalUnit):
    lock = functional_unit.lock
    if lock is None:
        return
    locked = lock.locked.value
    with container_state:
        if not locked:
            st.markdown(f"🔓")
        else:
            st.markdown(f"🔐 Remaining {format_number(0.001 * lock.remaining_lock_time.value)}s  \r\n{lock.locking_user.value}@{lock.locking_client.value}")
