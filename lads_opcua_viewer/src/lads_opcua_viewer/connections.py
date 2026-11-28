"""Connections to the LADS OPC UA servers of the viewer (cached per Streamlit server) and their shutdown."""

import atexit
import lads_opcua_client as lads
import streamlit as st
import time


# For disabling the exit of the threadpool executor
# This is necessary to avoid an error message when the app is closed
# Threads are manually stopped by the atexit in the Connection class
def disable_exit_for_threadpool_executor():
    import concurrent.futures
    atexit.unregister(concurrent.futures.thread._python_exit)


disable_exit_for_threadpool_executor()


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
        # wait until every server has connected or failed once; unreachable ones keep retrying in the background
        while not connections.settled and len(connections.connections) > 0:
            time.sleep(0.1)
    return connections


# MARK: ConnectionsManager
class ConnectionsManager:
    def __init__(self):
        self.connections = None

    def set_connections(self, connections):
        self.connections = connections

    def disconnect_all(self):
        if self.connections is not None:
            self.connections.disconnect()


# MARK: Global ConnectionsManager
connections_manager = ConnectionsManager()

# @DrMatthiasArnold at atexit there is an attempt to disconnect all servers
atexit.register(connections_manager.disconnect_all)
