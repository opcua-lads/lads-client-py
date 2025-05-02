import pytest
import lads_opcua_client as lads
import time
import os
import sys

def test_connections():
    json_file = os.path.join(os.path.dirname(__file__), "config.json")

    print(sys.executable)
    print(dir(lads))

    # Check if the configuration file exists
    try:
        with open(json_file) as file:
            pass
    except FileNotFoundError:
        pytest.fail("Configuration file not found")

    # Initialize connections
    conns = lads.Connections(json_file)
    conns.connect()

    # Wait for initialization
    timeout = 30  # Timeout in seconds
    start_time = time.time()
    while not conns.initialized:
        if time.time() - start_time > timeout:
            pytest.fail("Connection initialization timed out")
        time.sleep(1)

    assert conns.initialized, "Connections were not initialized"

    # Validate server details
    for conn in conns.connections:
        server = conn.server
        assert server.name is not None, "Server name is missing"
        assert len(server.devices) >= 0, "Devices list is invalid"

        for device in server.devices:
            assert device.unique_name is not None, "Device unique name is missing"

        functional_units = server.functional_units
        assert len(functional_units) >= 0, "Functional units list is invalid"

        for fu in functional_units:
            assert fu.unique_name is not None, "Functional unit unique name is missing"
            assert fu.at_name is not None, "Functional unit 'at_name' is missing"

            functions = fu.functions
            assert len(functions) >= 0, "Functions list is invalid"

            for func in functions:
                assert func.unique_name is not None, "Function unique name is missing"

                variables = func.variables
                assert len(variables) >= 0, "Variables list is invalid"

                for var in variables:
                    assert var.display_name is not None, "Variable display name is missing"
                    assert var.value_str is not None, "Variable value string is missing"

    # Disconnect connections
    conns.disconnect()