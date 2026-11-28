"""View of the active program and the results of a functional unit."""

import lads_opcua_client as lads
import pandas as pd
import streamlit as st
import time
from lads_opcua_viewer.formatting import format_number, to_float
from lads_opcua_viewer.ui import empty
from lads_opcua_viewer.views.tables import show_variables_table


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
                                            num_rows="dynamic", hide_index=True, width="stretch")
            job_id = st.text_input("Supervisory job id", value="My Job")
            task_id = st.text_input("Supervisory task id", value="My Task")
            with st.expander("Samples", expanded=False):
                samples_df = st.data_editor(pd.DataFrame({"ContainerId": ["0815", "0815"], "SampleId": ["4711", "4712"], "Position": ["A1", "A2"], "CustomData": ["", ""]}, dtype="string"),
                                            column_config={"ContainerId": st.column_config.TextColumn(),
                                                            "SampleId": st.column_config.TextColumn(),
                                                            "Position": st.column_config.TextColumn(),
                                                            "CustomData": st.column_config.TextColumn(),
                                                            },
                                            num_rows="dynamic", hide_index=True, width="stretch")
            if st.form_submit_button("Start Program", help="**Start's a program run** based on the selected program-template.  \r\nIf you want to manually stop or abort the current run, utilize the functional-unit's commands (drop-down next to the unit's state indicator)."):
                    functional_unit.functional_unit_state.start_program(template_id, key_value_df, job_id, task_id, samples_df)

    progress_container = empty(st.empty())
    update_active_program(progress_container, functional_unit)
    return progress_container


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
                if active_program.has_runtime_progress:
                    st.progress(active_program.current_runtime_progress, "Program run-time progress")
                    st.write(f"{format_number(0.001 * to_float(active_program.current_runtime.value))}s / {format_number(0.001 * to_float(active_program.estimated_runtime.value))}s")
                if active_program.has_step_number_progress:
                    st.progress(active_program.current_step_number_progress, "Program steps progress")
                    s = f"{active_program.current_step_number.value} / {active_program.estimated_step_numbers.value}"
                    step_name = active_program.current_step_name
                    if step_name is not None:
                        s = s + f" '{step_name.value_str}'"
                    st.write(s)
                if active_program.has_step_runtime_progress:
                    st.progress(active_program.current_step_runtime_progress, "Program step run-time progress")
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
