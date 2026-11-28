"""View of the asset management data and the components of a device."""

import lads_opcua_client as lads
import pandas as pd
import streamlit as st
from lads_opcua_viewer.callbacks import call_state_machine_method
from lads_opcua_viewer.formatting import format_number, to_float
from lads_opcua_viewer.views.tables import show_variables_table


# MARK: show_asset_management
def show_asset_management(container, device: lads.Device):
    device_state_machine = device.device_state
    device_state_methods = device_state_machine.method_names
    operation_mode_state_machine = device.machinery_operation_mode
    operation_mode_methods = [] if operation_mode_state_machine is None else operation_mode_state_machine.method_names

    with container.container():
        col_device, col_map, col_image = st.columns([1, 2, 0.5])
        with col_device:
            if len(device_state_methods) > 0:
                device_method = st.selectbox(label="Device control", options=device_state_methods, index=None, key=device_state_machine.nodeid, on_change=call_state_machine_method(device_state_machine))
            if len(operation_mode_methods) > 0:
                operation_mode_method = st.selectbox(label="Operation mode", options=operation_mode_methods, index=None, key=operation_mode_state_machine.nodeid, on_change=call_state_machine_method(operation_mode_state_machine))
            location_variables = device.location_variables
            if len(location_variables) > 0:
                with st.form("Locations", clear_on_submit=True):
                    input_values = {}
                    for variable in location_variables:
                        name = variable.display_name
                        value = variable.value_str
                        input_values[variable] = st.text_input(label=f"{name}: {value}", value=value, key=variable.nodeid, help=variable.description.Text)
                    if st.form_submit_button("Update Locations"):
                        for variable, value in input_values.items():
                            if value != variable.value_str:
                                variable.set_value(value    )
            container_device = st.empty()
        with col_map:
            container_map = st.empty()
        with col_image:
            container_image = st.empty()
        container_components = st.empty()

    update_asset_management(container_device, container_map, container_image, container_components, device)
    return container_device, container_map, container_image, container_components


# MARK: update_asset_management
def update_asset_management(container_device, container_map, container_image, container_components, device: lads.Device):
    with container_device:
        state_vars = device.state_machine_variables
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
            # st.map(df, zoom=8, use_container_width=True)
            st.map(df, zoom=8, width="stretch")
    with container_image:
        images = device.device_type_images
        if len(images) > 0:
            try:
                st.image(images)
            except:
                print("Error showing device image")
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
                        start = to_float(counter.start_value.value, default=100.0)
                        limit = to_float(counter.limit_value.value, default=0.0)
                        eu = counter.eu
                        # the counter runs from start to limit, downwards (start > limit) or upwards (start < limit)
                        counts_down = start >= limit
                        passed = (lambda v, reference: v < reference) if counts_down else (lambda v, reference: v > reference)
                        warning = False if counter.warning_values is None else any(passed(value, float(warning_value)) for warning_value in counter.warning_values.value)
                        alarm = passed(value, limit)
                        color = "red" if alarm else "orange" if warning else "green"
                        s = f"{counter.display_name} [{format_number(start)} > :{color}[**{format_number(value)}**] > {format_number(limit)}] {eu}"
                        low, high = min(start, limit), max(start, limit)
                        x = (value - low) / (high - low) if high > low else 0       # the bar follows the value between start and limit
                        st.progress(min(max(x, 0.0), 1.0), s)
                    except Exception as e:
                        pass

    if component.components is not None:
        count = expanded_count
        for sub_component in component.components:
            count = count - 1
            show_components(sub_component, count)
