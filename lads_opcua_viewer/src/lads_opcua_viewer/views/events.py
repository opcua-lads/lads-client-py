"""View of the events of a device."""

import datetime as dt
import lads_opcua_client as lads
import streamlit as st
from lads_opcua_viewer.keys import lastEventListUpdateKey


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
                # use_container_width=True,
                width="stretch",
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
