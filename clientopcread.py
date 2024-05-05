from opcua import Client 

url="opc.tcp://lenovo:62640/IntegrationObjects/ServerSimulator"
nodeId="ns=2;s=Tag7"
client=Client(url)
client.connect()
node=client.get_node(nodeId)
print(node.get_value())
