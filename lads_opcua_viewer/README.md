## Streamlit Viewer for LADS OPC UA Servers

After installing the lads_opcua_viewer library, the Streamlit all can be started by running:

```bash
lads_opcua_viewer
```

The viewer will open in your default browser and you can start exploring the LADS OPC UA Server.

If a `config.json file` (as examplified below) is present in the current working directory, the viewer will
automatically connect to the server/s specified in the config file. If a connection is not enabled, no connection
will be established.

```json
{
    "connections": [
        {
            "url": "opc.tcp://localhost:XXXX",
            "user": "the_user",
            "password": "the_password",
            "enabled": true
        },
        {
            "url": "opc.tcp://localhost:XXXX",
            "enabled": false
        },
        {
            "url": "opc.tcp://localhost:XXXX",
            "enabled": false
        }
    ]
}
```
