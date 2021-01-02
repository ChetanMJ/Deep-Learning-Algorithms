import numpy as np
import math


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

        

class Conv1D():
    def __init__(self, in_channel, out_channel, 
                 kernel_size, stride):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        
        
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        ## Your codes here
        self.x = x

        self.batch, __ , self.width = x.shape
        assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)
        dilation = 1
        Lout = math.floor(((self.width - dilation * (self.kernel_size - 1) - 1)/self.stride) + 1)
        
       
        output = np.zeros((self.batch,self.out_channel,Lout))
        
        i = 0
       
        for L in range(Lout):
            for o in range(self.out_channel):
               output[:,o,L] = np.sum((x[:,:,i:(i + self.kernel_size)] * self.W[o,:,:]), axis=(1,2)) + self.b[o]
            i = i + self.stride

        return output


    def backward(self, delta):
        
        ## Your codes here
        dx = np.zeros(self.x.shape)
        
        self.db = np.sum(delta, axis=(0,2))
        
        i = 0
       
        Lout = delta.shape[2]
        
        for L in range(Lout):
            for o in range(self.out_channel):
                delta_curr = delta[:,o,L]
                
                x_curr = self.x[:,:,i:(i + self.kernel_size)]
                
                delta_repeat = np.repeat(delta_curr,x_curr.shape[1]*x_curr.shape[2]).reshape(x_curr.shape)
                
                ##sum over all the bacthes of data which is axis = 0 of x
                self.dW[o,:,:] += np.sum(delta_repeat * x_curr, axis = 0)
                
                dx[:,:,i:(i + self.kernel_size)] += self.W[o,:,:] * delta_repeat
                
            i = i + self.stride
                
        return dx




class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        ## Your codes here

        self.batch_size = x.shape[0]
        self.in_channel = x.shape[1]
        self.in_width = x.shape[2]

        return x.reshape((self.batch_size, self.in_channel*self.in_width))

    def backward(self, x):
        # Your codes here
        return x.reshape((self.batch_size, self.in_channel,self.in_width))


class ReLU():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.dy = (x>=0).astype(x.dtype)
        return x * self.dy

    def backward(self, delta):
        return self.dy * delta
    

    