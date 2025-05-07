# LADS-CLIENT-PY
Pythonic client and viewer LADS OPC UA libraries.

# Installing

## Python
Python can be downloaded from the official website [python.org](https://www.python.org/downloads/). The libraries are compatible with Python 3.12 and later. If you have an older version of Python installed, it is recommended to install a newer version in order to use the libraries. The libraries are currently not available on PyPi and need to be installed from source. Check if Python is installed on your system with the following command for Windows users:
```bash
py --version
```
for Linux users:
```bash
python3 --version
```
On Windows, if no version is shown but Python is installed, check if Python is added to the [Environmental Variables](https://realpython.com/add-python-to-path/).

## Virtual Environment
Ideally, the libraries should be installed in a virtual environment. Create a virtual environment with the following command for Windows users:
```bash
py -m venv .venv
```
for Linux users:
```bash
python3 -m venv .venv
```
Note that you can choose a different name for the virtual environment than `.venv`. Then, activate the virtual environment with the following command if you are using windows and Command Prompt (cmd):
```bash
.venv\Scripts\activate
```
for Linux users:
```bash
source .venv/bin/activate
```
See the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for more information on creating and managing a Python virtual environment.

## Build Library
The [build](https://pypi.org/project/python-build/) library is required to build the libraries. It can be installed with the following command:
```bash
pip install build
```
Within the /dev folder you will find powershell und bash scripts which help to automate the steps which are outlined in the following.

## Client Library
The *[lads_opcua_client](https://github.com/opcua-lads/lads-client-py/tree/main/lads_opcua_client)* library can be build and installed with the following command for Windows users:
```bash
cd lads_opcua_client
py -m build
pip install dist/lads_opcua_client-0.0.1.tar.gz
```
for Linux users:
```bash
cd lads_opcua_client
python3 -m build
pip install dist/lads_opcua_client-0.0.1.tar.gz
```

## Viewer Library
The *[lads_opcua_viewer](https://github.com/opcua-lads/lads-client-py/tree/main/lads_opcua_viewer)* library requires the client library to be installed in the same environment. The viewer library can be build and installed with the following command for Windows users:
```bash
cd lads_opcua_viewer
py -m build
pip install dist/lads_opcua_viewer-0.0.1.tar.gz
```
for Linux users:
```bash
cd lads_opcua_viewer
python3 -m build
pip install dist/lads_opcua_viewer-0.0.1.tar.gz
```

# Instructions
The *[lads_opcua_client](https://github.com/opcua-lads/lads-client-py/tree/main/lads_opcua_client)* library provides a Pythonic interface to the LADS OPC UA server. Consult the client [README](https://github.com/opcua-lads/lads-client-py/blob/main/lads_opcua_client/README.md) file in the package directory for further information. The *[lads_opcua_viewer](https://github.com/opcua-lads/lads-client-py/tree/main/lads_opcua_viewer)* library provides a Streamlit based viewer for the LADS OPC UA server. Consult the viewer [README](https://github.com/opcua-lads/lads-client-py/blob/main/lads_opcua_viewer/README.md) file in the package directory for further information. Note that the viewer depends on the client library and requires it to be installed in the same environment.
