from layers import *

class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        
        conv1 = Conv1D(24, 8, 8, 4)
        conv2 = Conv1D(8, 16, 1, 1)
        conv3 = Conv1D(16, 4, 1, 1)
        rel1 = ReLU()
        rel2 = ReLU()
        Flat = Flatten()
        
        self.layers = [conv1,rel1,  conv2, rel2, conv3, Flat]
        
    def __call__(self, x):

        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given

        Clayers = []
        
        for i in range(len(self.layers)):
            if i%2 == 0:
                Clayers.append(self.layers[i])
            
        
        for l in range(len(Clayers)):
            for i in range(Clayers[l].out_channel):
                out_channels = Clayers[l].out_channel
                in_channels = Clayers[l].in_channel
                k_size = Clayers[l].kernel_size
                Weights_per_out_channel = k_size * in_channels               
                Clayers[l].W[i,:,:] = np.transpose(np.transpose(weights[l][0:Weights_per_out_channel,0:out_channels])[i,:].reshape((k_size,in_channels)))
 
        
        return None

    def forward(self, x):
        # You do not need to modify this method
        out = x
        
        for layer in self.layers:
            out = layer(out)
        
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta




class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        conv1 = Conv1D(24, 2, 2, 2)
        conv2 = Conv1D(2, 8, 2, 2)
        conv3 = Conv1D(8, 4, 2, 1)
        rel1 = ReLU()
        rel2 = ReLU()
        Flat = Flatten()
        
        self.layers = [conv1, rel1, conv2, rel2, conv3, Flat]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # Load the weights for your CNN from the MLP Weights given

        Clayers = []
        
        for i in range(len(self.layers)):
            if i%2 == 0:
                Clayers.append(self.layers[i])
            
        
        for l in range(len(Clayers)):
            for i in range(Clayers[l].out_channel):
                out_channels = Clayers[l].out_channel
                in_channels = Clayers[l].in_channel
                k_size = Clayers[l].kernel_size
                Weights_per_out_channel = k_size * in_channels               
                Clayers[l].W[i,:,:] = np.transpose(np.transpose(weights[l][0:Weights_per_out_channel,0:out_channels])[i,:].reshape((k_size,in_channels)))
 
        return None

    def forward(self, x):
        # You do not need to modify this method
        out = x
        
        for layer in self.layers:
            out = layer(out)
        
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
