import lads_opcua_client as lads
import time

def main():
    #conn = lads.Connection (url = "opc.tcp://IUTALADSOPC:26543")
    json_file = "config.json"

    try:
        with open(json_file) as file:
            pass
    except FileNotFoundError:
        print("File not found")
        return

    conns = lads.Connections(json_file)
    conns.connect()
    print("Waiting for connection to be initialized...")
    while not conns.initialized:
        time.sleep(1)
    print(conns.initialized)

    for conn in conns.connections:
        #print(conn.data_types) What is this?
        server = conn.server
        print("Server details: ")
        print(server.name)
        print("Number of devices: ", server.devices.__len__())
        for device in server.devices:
            print("  Device: ", device.unique_name)
        functional_units = server.functional_units
        print("  Number of functional_units: ", functional_units.__len__())
        for fu in functional_units:
            print("    Name of functional_unit: ", fu.unique_name)
            print("    Other name of functional_unit: ", fu.at_name)
            functions = fu.functions
            print("    Number of functions: ", functions.__len__())
            for func in functions:
                print("      Name of function: ", func.unique_name)
                variables = func.variables
                print("      Number of variables: ", variables.__len__())
                for var in variables:
                    print("        Name of variable: ", var.display_name)
                    print("        Value of variable: ", var.value_str)

    conns.disconnect()

if __name__ == '__main__':
    main()
