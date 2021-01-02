import numpy as np

HIDDEN_DIM = 4

class Sigmoid:
    # DO NOT DELETE
	def __init__(self):
		pass
	def forward(self, x):
		self.res = 1/(1+np.exp(-x))
		return self.res
	def backward(self):
		return self.res * (1-self.res)
	def __call__(self, x):
		return self.forward(x)


class Tanh:
    # DO NOT DELETE
	def __init__(self):
		pass
	def forward(self, x):
		self.res = np.tanh(x)
		return self.res
	def backward(self):
		return 1 - (self.res**2)
	def __call__(self, x):
		return self.forward(x)

class Linear():
    # DO NOT DELETE
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)
        
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx

class GRU_Cell:
	"""docstring for GRU_Cell"""
	def __init__(self, in_dim, hidden_dim):
		self.d = in_dim
		self.h = hidden_dim
		h = self.h
		d = self.d
		self.x_t=0

		self.Wzh = np.random.randn(h,h)
		self.Wrh = np.random.randn(h,h)
		self.Wh  = np.random.randn(h,h)

		self.Wzx = np.random.randn(h,d)
		self.Wrx = np.random.randn(h,d)
		self.Wx  = np.random.randn(h,d)

		self.dWzh = np.zeros((h,h))
		self.dWrh = np.zeros((h,h))
		self.dWh  = np.zeros((h,h))

		self.dWzx = np.zeros((h,d))
		self.dWrx = np.zeros((h,d))
		self.dWx  = np.zeros((h,d))

		self.z_act = Sigmoid()
		self.r_act = Sigmoid()
		self.h_act = Tanh()

        # Define other variables to store forward results for backward here
        
		self.z1 = np.zeros(h)
		self.z2 = np.zeros(h)
		self.z3 = np.zeros(h)
		self.zt = np.zeros(h)
        
		self.z4 = np.zeros(h)
		self.z5 = np.zeros(h)
		self.z6 = np.zeros(h)
		self.rt = np.zeros(h)
        
		self.z7 = np.zeros(h)
		self.z8 = np.zeros(h)
		self.z9 = np.zeros(h)
		self.z10 = np.zeros(h)
		self.htt = np.zeros(h)
        
		self.z11 = np.zeros(h)
		self.z12 = np.zeros(h)
		self.z13 = np.zeros(h)
		self.ht = np.zeros(h)


	def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
		self.Wzh = Wzh
		self.Wrh = Wrh
		self.Wh = Wh
		self.Wzx = Wzx
		self.Wrx = Wrx
		self.Wx  = Wx

	def __call__(self, x, h):
		return self.forward(x,h)

	def forward(self, x, h):
		# input:
		# 	- x: shape(input dim),  observation at current time-step
		# 	- h: shape(hidden dim), hidden-state at previous time-step
		# 
		# output:
		# 	- h_t: hidden state at current time-step
        
		''' 
		zt = self.z_act(np.dot(self.Wzh, h) + np.dot(self.Wzx, x))
		rt = self.r_act(np.dot(self.Wrh, h) + np.dot(self.Wrx, x))
		htt = self.h_act(np.dot(self.Wh, rt*h) + np.dot(self.Wx, x))
		h_t = ((1 - zt) * h) + (zt * htt)
        
		return h_t
    
		'''
        ## h here is actually h_t-1
		self.x = x
		self.ht_1 = h
        
		self.z1 = np.dot(self.Wzh, h)
		self.z2 = np.dot(self.Wzx, x)
		self.z3 = self.z1 + self.z2
		self.zt = self.z_act(self.z3)
        
		self.z4 = np.dot(self.Wrh, h)
		self.z5 = np.dot(self.Wrx, x)
		self.z6 = self.z4 + self.z5
		self.rt = self.r_act(self.z6)
        
		self.z7 = self.rt * h
		self.z8 = np.dot(self.Wh, self.z7)
		self.z9 = np.dot(self.Wx, x)
		self.z10 = self.z8 + self.z9
		self.htt = self.h_act(self.z10)
        
		self.z11 = (1 - self.zt)
		self.z12 = (self.z11 * h)
		self.z13 = (self.zt * self.htt)
		self.ht = self.z12 + self.z13
        
		return self.ht
  


    # This  must calculate the gradients wrt the parameters and returns the derivative wrt the inputs, xt and ht, to the cell.
	def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h
        #raise NotImplementedError
        
		d = self.d
		h = self.h
        
		dht = delta
		dz13 = dht
		dz12 = dht
		dzt = dz13 * self.htt
		dhtt = dz13 * self.zt
		dz11 = dz12 * np.transpose(self.ht_1)
		dht_1 = (dz12 * np.transpose(self.z11))
		dzt = dzt + (-1 * dz11)
        
        
		dz10 = dhtt * self.h_act.backward()
		dz9 = dz10
		dz8 = dz10
		self.dWx = np.dot(dz9.reshape((h,1)), np.transpose(self.x.reshape((d,1))))
		dx = np.dot(dz9, self.Wx)
		self.dWh = np.dot(dz8.reshape((h,1)), np.transpose(self.z7.reshape((h,1))))
		dz7 = np.dot(dz8, self.Wh)
		dht_1 = dht_1 + (dz7 * self.rt)
		drt = dz7 * self.ht_1

		dz6 = drt * self.r_act.backward()
		dz4 = dz6
		dz5 = dz6
		self.dWrx = np.dot(dz5.reshape((h,1)), np.transpose(self.x.reshape((d,1))))
		dx = dx + np.dot(dz5, self.Wrx)
		self.dWrh = np.dot(dz4.reshape((h,1)),np.transpose(self.ht_1.reshape((h,1))))
		dht_1 = dht_1 + np.dot(dz4, self.Wrh)
        
		dz3 = dzt * self.z_act.backward()
		dz1 = dz3
		dz2 = dz3
		self.dWzx = np.dot(dz2.reshape((h,1)), np.transpose(self.x.reshape((d,1))))
		dx = dx + np.dot(dz2, self.Wzx)
		self.dWzh = np.dot(dz1.reshape((h,1)),np.transpose(self.ht_1.reshape((h,1))))
		dht_1 = dht_1 + np.dot(dz1, self.Wzh)
        
        
        
		return dx, dht_1

# This is the neural net that will run one timestep of the input 
# You only need to implement the forward method of this class. 
# This is to test that your GRU Cell implementation is correct when used as a GRU.	
class CharacterPredictor(object):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        # The network consists of a GRU Cell and a linear layer   
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.rnn = GRU_Cell(input_dim, hidden_dim)
        self.linear = Linear(hidden_dim, num_classes)

    def init_rnn_weights(self, w_hi, w_hr, w_hn, w_ii, w_ir, w_in):
        # DO NOT MODIFY
        self.rnn.init_weights(w_hi, w_hr, w_hn, w_ii, w_ir, w_in) 

    def __call__(self, x, h):
        return self.forward(x, h)        

    def forward(self, x, h):
        # A pass through one time step of the input 
        ht = self.rnn.forward(x,h)
        output = self.linear.forward(ht)
        
        return ht, output

# An instance of the class defined above runs through a sequence of inputs to generate the logits for all the timesteps. 
def inference(net, inputs):
    # input:
    #  - net: An instance of CharacterPredictor
    #  - inputs - a sequence of inputs of dimensions [seq_len x feature_dim]
    # output:
    #  - logits - one per time step of input. Dimensions [seq_len x num_classes]

    seq_len = inputs.shape[0]
    
    ht = np.zeros(net.hidden_dim)
    
    logits = []
    for i in range(seq_len):
        ht, out = net.forward(inputs[i,:], ht)
        logits.append(out)
        
    logits = np.array(logits)
    
    return logits

'''
gru = GRU_Cell(4,5)
x = np.arange(4)
h = np.arange(5)
y = gru.forward(x,h)

delta = np.array([0.1,0.2,0.3,0.1,0.3])

dx, dh = gru.backward(delta)
print(y)
print(dx)
print(dh)

x = np.arange(4*5).reshape((4,5))
gru_net = CharacterPredictor(5,10,13)
logit = inference(gru_net,x)
print(logit.shape)
'''