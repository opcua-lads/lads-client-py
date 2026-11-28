"""Python Streamlit app to visualize LADS OPC UA servers.

This library provides a Streamlit app to visualize the data of LADS OPC UA servers. It uses the LADS OPC UA client to connect to the servers and to read and write data. The app displays the functional units, the functions, the variables, the events, the programs, and the asset management of the servers. The app updates the data in real-time.

Copyright (c) 2023 Dr. Matthias Arnold, AixEngineers, Aachen, Germany.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import datetime as dt
import lads_opcua_client as lads
import pandas as pd
import streamlit as st
import time
from lads_opcua_viewer.connections import connections_manager, get_initialized_connections
from lads_opcua_viewer.keys import lastEventListUpdateKey, selectedFunctionalUnitKey
from lads_opcua_viewer.ui import empty
from lads_opcua_viewer.views.assets import show_asset_management, update_asset_management
from lads_opcua_viewer.views.charts import update_charts
from lads_opcua_viewer.views.documents import show_documents, update_documents
from lads_opcua_viewer.views.events import update_events
from lads_opcua_viewer.views.functions import show_functions, update_functions
from lads_opcua_viewer.views.program import show_active_program, update_active_program, update_result_set
from lads_opcua_viewer.views.state import show_lock_cmd, show_program_template_set, show_state, update_lock_state, update_state


st.set_page_config(page_title="LADS OPC UA Client", layout="wide")


# MARK: main
def main():
    connections = get_initialized_connections()

    # Set the connections for disconnecting all at exit and possible other uses
    # @DrMatthiasArnold we could also make the ConnectionManager asyncronous
    # to enable adding new connections and in general, to enable runnig the app asyncronously
    # based on an initial configuration file
    connections_manager.set_connections(connections)

    # servers which are not reachable are retried in the background and show up on the next rerun
    for url, error in connections.unreachable.items():
        st.sidebar.warning(f"{url} not reachable, retrying: {error}")
    functional_units = connections.functional_units
    if len(functional_units) == 0:
        if len(connections.connections) == 0:
            st.error("No enabled connections found in config.json (check 'enabled' and environment variables, see log).")
        else:
            st.error("No LADS OPC UA server reachable yet. Retrying in the background - reload to refresh.")
        st.button("Reload")
        st.stop()

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
        state_machine = selected_functional_unit.functional_unit_state
        running_state_machine = state_machine.running_state_machine
        # columns = [2, 2, 1, 2, 1.5, 0.2] if running_state_machine is not None else [2, 0.1, 1, 2, 1.5, 0.2]
        columns = [1, 1, 2, 1, 1.5, 0.2] if running_state_machine is not None else [1, 0.1, 2, 1, 1.5, 0.2]
        col_cmd, col_cmd_running, col_state, col_lock_cmd, col_lock_state, col_definition = st.columns(columns)
        with col_cmd:
            # for the command select box extract methods without input arguments (except Start)
            methods = list(filter(lambda method: (len(method.input_arguments) == 0) | (method.display_name == "Start"), state_machine.methods))
            method_names = map(lambda method: method.display_name, methods)
            if False:
                for name in method_names:
                    on_click = state_machine.start(pd.DataFrame()) if name == "Start" else state_machine.call_method_by_name(name)
                    st.button(name, on_click=on_click)
            else:
                cmd = st.selectbox("Command", options=method_names, index=None, label_visibility="collapsed", placeholder="Choose a command", key="Command")
                if cmd == "Start":
                    state_machine.start(pd.DataFrame())
                else:
                    state_machine.call_method_by_name(cmd)

        with col_cmd_running:
            if running_state_machine is not None:
                methods = running_state_machine.methods
                method_names = map(lambda method: method.display_name, methods)
                cmd = st.selectbox("Command", options=method_names, index=None, label_visibility="collapsed", placeholder="Choose a command", key="CommandRunning")
                running_state_machine.call_method_by_name(cmd)
        with col_state:
            container_state = show_state(col_state, selected_functional_unit)

        with col_lock_cmd:
            container_lock_cmd = st.empty()
        with col_lock_state:
            container_lock_state = st.empty()
        show_lock_cmd(container_lock_cmd, selected_functional_unit)
        update_lock_state(container_lock_state, selected_functional_unit)

        with col_definition:
            definition = selected_functional_unit.dictionary_entries_as_markdown
            if len(definition) > 0:
                # st.markdown(":gray[Definition]", help = definition)
                st.markdown(" ", help = definition)

    # eventually create addtional tab for compliance documents
    document_set: lads.LADSSet = None
    try:
        document_set = selected_functional_unit.device.compliance_document_set
    except(Exception):
        pass
    if document_set is not None:
        tab_functions, tab_program_manager, tab_device, tab_documents = st.tabs(["Operation", "Program Management", "Asset Management", "Compliance Documents"])
    else:
        tab_functions, tab_program_manager, tab_device = st.tabs(["Operation", "Program Management", "Asset Management"])
        tab_documents = None
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
            container_device, container_map, container_image, container_components = show_asset_management(empty(st.empty()), selected_functional_unit.device)

        if tab_documents is not None:
            with tab_documents:
                container_documents, container_document_detail = show_documents(empty(st.empty()), document_set)

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
                update_lock_state(container_lock_state, selected_functional_unit)
                update_functions(function_containers)
                update_events(container_events, selected_functional_unit)
                update_active_program(progress_container, selected_functional_unit)
                update_asset_management(container_device, container_map, container_image, container_components, selected_functional_unit.device)
                index += 1
                if index >= 5:
                    index = 0
                    update_charts(container_chart, selected_functional_unit, True)
                    update_result_set(container_results, selected_functional_unit)
                    if tab_documents is not None:
                        update_documents(container_documents, container_document_detail, document_set)
                time.sleep(1)

        update_loop()


if __name__ == '__main__':
    main()
