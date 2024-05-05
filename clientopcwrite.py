from multiprocessing import Value
from opcua import Client
import random as rd

url = "opc.tcp://lenovo:62640/IntegrationObjects/ServerSimulator"
nodeId = "ns=2;s=Tag7"
client=Client(url)
client.connect()
node = client.get_node(nodeId)

while True: 
     value = rd.randint(1000,1000000)
     node.set_value(value)





