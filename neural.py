#coding:utf8
import random

class Node():
    def __init__(self,layer_index,node_index):
	'''
	create a node,locate by layer_index and node_index
	'''
	self.layer_index=layer_index
	self.node_index=node_index
	self.downstream=list()
	self.upstream=list()
	self.output=0
	self.delta=0

    def set_output(self,output):
	'''
	set output,because it will be next node's input
	'''
	self.output=output

    def append_downstream_connection(self,conn):
	'''
	append an downstream to connection
	'''
	self.downstream.append(conn)

    def append_upstream_connection(self,conn):
	'''
	append an upstream to connection
	'''
	self.upstream.append(conn)

    def calc_output(self):
	'''
	calculate out put by $y=sigmoid(\vec w \centerdot \vec x)$
	'''
	output=0
	for one in self.upstream:
	    output+=one.upstream_node.output*one.weight
	self.output=sigmoid(output)
    
    def calc_hidden_layer_delta(self):
	'''
	calculate hidden layer delta by
	\delta_i=a_i(1-a_i)sum_(k \in outputs)w_{ki}\delta_k
	'''
	downstream_delta=0
	for one in self.downstream:
	    downstream_delta+=one.downstream_node.delta*one.weight
	self.delta=self.output*(1-self.output)*downstream_delta

    def calc_output_layer_delta(self,label):
	'''
	calculate output layer delta by
	\delta_i=y_i(1-y_i)(t_i-y_i)
	'''
	self.delta=self.output*(1-self.output)*(label-self.output)

    def __str__(self):
	return "node"

class ConstNode:
    '''
    bias node
    '''
    def __init__(self,layer_index,node_index):
	self.layer_index=layer_index
	self.node_index=node_index
	self.downstream=list()
	self.output=1

    def append_downstream_connection(self,conn):
	self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
	downstream_delta=0
	for one in self.downstream:
	    downstream_delta+=one.downstream_node.delta*one.weight
	self.delta=self.output*(1-self.output)*downstream_delta

    def __str__(self):
	return 'bias node'

class Layer:
    def __init__(self,layer_index,node_count):
	'''
	init a layer
	class Layer include layer index and some nodes
	'''
	self.layer_index=layer_index
	self.nodes=list()
	for i in range(node_count):
	    self.nodes.append(Node(layer_index,i))
	self.nodes.append(ConstNode(layer_index,node_count))

    def set_output(self,data):
	'''
	set the layer's output,using when the layer is input layer
	'''
	for i in range(len(data)):
	    self.nodes[i].set_out(data[i])

    def calc_output(self):
	'''
	calculate the layer's output vector
	'''
	for node in self.nodes[:-1]:
	    node.calc_output()
	
    def dump(self):
	'''
	print the layer's info
	'''
	for node in self.nodes:
	    print (node)

class Connection:
    def __init__(self,upstream_node,downstream_node):
	'''
	init connection,the weight init to a little random
	'''
	self.upstream_node=upstream_node
	self.downstream_node=downstream_node
	self.weight=random.uniform(-0.1,0.1)
	self.gradient=0.0

    def calc_gradient(self):
	'''
	calculate the gradient
	'''
	self.gradient=self.downstream_node.delta*self.upstream_node.output

	def get_gradient(self):
	    return self.gradient

	def update_weight(self,rate):
	    '''
	    update weight by gradient descent
	    '''
	    self.calc_gradient()
	    self.weight+=rate*self.gradient

	def __str__(self):
	    return 'connection'

class Connections:
    def __init__(self):
	self.connections=list()

    def add_connection(self,connection):
	self.connections.append(connection)

    def dump(self):
	for conn in self.connections:
	    print(conn)

class Network:
    def __init__(self,layers):
	'''
	init a fully connected neural network
	'''
	self.connections=Connections()
	self.layers=list()
	layer_count=len(layers)
	node_count=0
	for i in range(layer_count):
	    self.layers.append(Layer(i,layers[i]))
	for layer in range(layer_count-1):
	    connections=[Connection(upstream_node,downstream_node)
			for upstream_node in self.layers[layer].nodes 
			for downstream_node in self.layers[layer+1].nodes[:-1]]

	    for conn in connections:
		self.connections.add_connection(conn)
		conn.downstream_node.append_upstream_connection(conn)
		conn.upstream_node.append_downstream_connection(conn)

    def train(self,labels,data_set,rate,iteration):
	'''
	train the neural network
	'''
	for i in range(iteration):
	    for d in range(len(data_set)):
		self.train_one_sample(labels[d],data_set[d],rate)

    def train_one_sample(self,label,sample,rate):
	'''
	train one samlpe
	'''
	self.predict(sample)
	self.calc_delta(label)
	self.update_weight(rate)

    def calc_delta(self,label):
	'''
	calculate delta
	'''
	output_nodes=self.layers[-1].nodes
	for i in range(len(label)):
	    output_nodes[i].calc_output_layer_delta(label[i]_)
	for layer in self.layers[-2::-1]:
	    for node in layer.nodes:
		node.calc_hidden_layer_delta()

    def update_weight(self,rate):
	'''
	update weight
	'''
	for layer in self.layers[:-1]:
	    for node in layer.nodes:
		for conn in node.downstream:
		    conn.update_weight(rate)

    def calc_gradient(self):
	'''
	calculate gradient
	'''
	for layer in self.layers[:-1]:
	    for node in layer.nodes:
		for conn in node.downstream:
		    conn.calc_gradient()
    
    def get_gradient(self,label,sample):
	'''
	get gradient
	'''
	self.perdict(sample)
	self.calc_delta(label)
	self.calc_gradient()

    def predict(self,sample):
	'''
	predict output by input
	'''
	self.layers[0].set_output(sample)
	for i in range(1,len(self.layers)):
	    self.layers[i].calc_output()
	return map(lambda node:node.output,self.layers[-1].nodes[:-1])

    def dump(self):
	'''
	print neural info
	'''
	for layer in self.layers:
	    layer.dump()
