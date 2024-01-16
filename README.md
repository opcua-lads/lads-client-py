# lads-client-py
Pythonic LADS OPC UA client with optional streamlit based GUI.

## Required Python Packages
* https://pypi.org/project/asyncua/
* https://pypi.org/project/pandas/
* https://pypi.org/project/streamlit/
* https://pypi.org/project/plotly/

## Configuration
For the time being, connections to a LADS OPC UA server a configured via the config.json file.
```
{
    "connections": [
        {
            "url": "opc.tcp://localhost:26543",
            "enabled": true
        },
        {
            "url": "opc.tcp://localhost:4840/LADSServer",
            "enabled": false
        }
}
```
## Starting
The streamlit application is started by envoking streamlit from the command line while providing the path to the streamlit Python apllication as argument:
```
streamlit run src/lads_viewer.py
```
This call start the streamlit web-server and opens a web page pointing to the application at http://localhost:8501.



