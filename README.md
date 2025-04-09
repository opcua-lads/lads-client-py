# lads-client-py

Pythonic client and viewer LADS OPC UA libraries.

## Installing

Installation of the *[lads_opcua_client](https://github.com/opcua-lads/lads-client-py/tree/main/lads_opcua_client)* and *[lads_opcua_viewer](https://github.com/opcua-lads/lads-client-py/tree/main/lads_opcua_viewer)* libraries is done by building the libraries and
installing them with pip ideally in a virtual environment. The libraries are planned to be published on PyPi in
the future.

### Client Library

The *[lads_opcua_client](https://github.com/opcua-lads/lads-client-py/tree/main/lads_opcua_client)* library can be installed with:

```bash
cd lads_opcua_client
py -m build
pip install dist/lads_opcua_client-0.0.1.tar.gz
```

### Viewer Library

The *[lads_opcua_viewer](https://github.com/opcua-lads/lads-client-py/tree/main/lads_opcua_viewer)* library can be installed with:

```bash
cd lads_opcua_viewer
py -m build
pip install dist/lads_opcua_viewer-0.0.1.tar.gz
```

## Instructions

The *[lads_opcua_client](https://github.com/opcua-lads/lads-client-py/tree/main/lads_opcua_client)* library provides a Pythonic interface to the LADS OPC UA server. Consult the client [README](https://github.com/opcua-lads/lads-client-py/blob/main/lads_opcua_client/README.md) file
in the package directory for further information. The *[lads_opcua_viewer](https://github.com/opcua-lads/lads-client-py/tree/main/lads_opcua_viewer)* library provides a Streamlit
based viewer for the LADS OPC UA server. Consult the viewer [README](https://github.com/opcua-lads/lads-client-py/blob/main/lads_opcua_viewer/README.md) file in the package directory for further
information. Note that the viewer depends on the client library and requires it to be installed in the same environment.
