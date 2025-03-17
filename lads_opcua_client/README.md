## Python Client for LADS OPC UA Server
Create a connection to the LADS Server using the server URL:

```python
conn = Connection("your_server_url")
```
<br>

Start the connection to the server:

```python
conn.connect()
```
<br>

Get the server object of the Connection class

```python
serv = conn.server
```
<br>

Get the LADS Device list that is present in the LADS Server DeviceSet

```python
devices = serv.devices
```
<br>

For a Device 'i' the list of Functional Units present inside that Device can be retrieved as

```python
fu = devices[i].functional_units
```
<br>

The Function Set present inside a Functional Unit 'k' can be accessed as

```python
fs = fu[k].function_set
```
<br>

The list of Sensor Functions present inside the Function Set can be retrieved as 

```python
funcs = fs.functions
```
<br>

The Sensor Value of a Sensor Function 'j' can be accessed as 

```python
value = funcs[j].sensor_value.value

```
<br>

The connection to the server can be disconnected as 

```python
conn.disconnect()
```

<br><br>