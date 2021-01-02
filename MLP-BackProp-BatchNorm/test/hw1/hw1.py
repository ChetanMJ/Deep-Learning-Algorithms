"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os



class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):

        # Might we need to store something before returning?
        
        self.state = 1 / (1 + np.exp(-1 * x))

        return self.state

    def derivative(self):

        # Maybe something we need later in here...
        forward_x = self.state
        return  forward_x * (1 - forward_x)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        #return (math.exp(x) - math.exp(-1 * x))/(math.exp(x) + math.exp(-1 * x))
        return self.state

    def derivative(self):
        
        forward_x = self.state
        
        return (1 - forward_x**2)


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.maximum(0,x)
        return self.state

    def derivative(self):
        
        return 1.0 * ( self.state > 0.0 )

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):

        self.logits = x
        self.labels = y

        # ...
        
        c = np.max(x, axis = 1)
        c1 = c.reshape(len(c), 1)       
        x_c = np.add(x, -1*c1)
        
        ## log of softmax
        log_sm = x_c - (np.log(np.sum(np.exp(x_c), axis = 1))).reshape(len(c), 1)  
        self.sm = np.exp(log_sm) 
        forward = -1 * np.sum(np.multiply(y, log_sm), axis = 1)
        return forward

    def derivative(self):

        # self.sm might be useful here...

        return self.sm - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        if eval:
            norm_out = (x - self.running_mean) / ((self.running_var + self.eps)**0.5)
            outx = (np.multiply(norm_out,self.gamma) + self.beta) 
            return outx
        else :
            self.x = x
            self.mean = np.mean(x, axis = 0)
            self.var = np.var(x, axis = 0)
            self.norm = (x - self.mean) / ((self.var + self.eps)**0.5)
            self.out = np.multiply(self.norm,self.gamma) + self.beta
            self.running_mean = (self.running_mean * self.alpha) + ((1 - self.alpha)*self.mean)
            self.running_var = (self.running_var * self.alpha) + ((1 - self.alpha)*self.var)
            
            # update running batch statistics


        # ...

        return self.out

    def backward(self, delta):

        m = self.x.shape[0]
        dLdXci = np.multiply(delta,self.gamma)
        self.dbeta = np.sum(delta, axis = 0) * m
        self.dgamma = np.sum(np.multiply(delta, self.norm), axis = 0) * m
        dLdVar = np.multiply((-0.5*((self.var + self.eps)**-1.5)), np.sum(np.multiply(dLdXci,(self.x - self.mean)), axis = 0))
        dLdMu = (np.sum((-1 * np.multiply(dLdXci, ((self.var + self.eps)**-0.5))), axis=0) + \
                  np.multiply(((-2/m) * dLdVar) , np.sum((self.x - self.mean), axis = 0)))
        
        dLdXi = (np.multiply(dLdXci, ((self.var + self.eps)**-0.5)) + \
                np.multiply(dLdVar, ((2/m) * (self.x - self.mean))) + \
                (dLdMu * (1/m)))
        
        return np.transpose(dLdXi)


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.normal(size = (d0, d1))


def zeros_bias_init(d):
    return np.zeros((1,d))


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        

        self.W = []
        self.dW = []
        self.b = []
        self.db = []
        ##self.layer_input = []
        self.W_running_average = []
        self.b_running_average = []
        
        for i in range(self.nlayers):
            if (i == 0) and (i == (self.nlayers - 1)):
                self.W.append(random_normal_weight_init(self.input_size, self.output_size))
                self.dW.append(np.zeros((self.input_size, self.output_size)))
                self.W_running_average.append(np.zeros((self.input_size, self.output_size)))
                self.b.append(zeros_bias_init(self.output_size))
                self.b_running_average.append(zeros_bias_init(self.output_size))
                self.db.append(np.zeros((1, self.output_size)))
                break
            
            
            if i == 0:
                self.W.append(random_normal_weight_init(self.input_size, hiddens[i]))
                self.dW.append(np.zeros((self.input_size, hiddens[i])))
                self.W_running_average.append(np.zeros((self.input_size, hiddens[i])))
                self.b.append(zeros_bias_init(hiddens[i]))
                self.b_running_average.append(zeros_bias_init(hiddens[i]))
                self.db.append(np.zeros((1, hiddens[i])))
                continue
            
            if i == (self.nlayers - 1):
                self.W.append(random_normal_weight_init(hiddens[i-1], self.output_size))
                self.dW.append(np.zeros((hiddens[i-1], self.output_size)))
                self.W_running_average.append(np.zeros((hiddens[i-1], self.output_size)))
                self.b.append(zeros_bias_init(self.output_size))
                self.b_running_average.append(zeros_bias_init(self.output_size))
                self.db.append(np.zeros((1, self.output_size)))
                break
            
            self.W.append(random_normal_weight_init(hiddens[i-1], hiddens[i]))
            self.dW.append(np.zeros((hiddens[i-1], hiddens[i])))
            self.W_running_average.append(np.zeros((hiddens[i-1], hiddens[i])))
            self.b.append(zeros_bias_init(hiddens[i]))
            self.b_running_average.append(zeros_bias_init(hiddens[i]))
            self.db.append(np.zeros((1, hiddens[i])))
            
        
        self.bn_layers = []
        
        if self.bn:
            for i in range(num_bn_layers):
                self.bn_layers.append(BatchNorm(hiddens[i]))
                
         # Feel free to add any other attributes useful to your implementation (input, output, ...)
        
        self.x = None
        self.batch_size = None

    def forward(self, x):
        
        self.x = x
        self.batch_size = float(x.shape[0])
        activation_output = x
        bn_layer_count = self.num_bn_layers
        
        
        for i in range(self.nlayers):
           
            linear = np.matmul(activation_output,self.W[i]) + self.b[i]  
            
            
            if bn_layer_count > 0 and i < bn_layer_count:
                
                bn_output = self.bn_layers[i-1].forward(linear, not(self.train_mode))
                               
                bn_layer_count = bn_layer_count - 1
                              
            else:
                bn_output = linear

            activation_output = self.activations[i](bn_output)
            


            
            

        return activation_output

    def zero_grads(self):
        
        for i in range(len(self.dW)):
            self.dW[i] = np.zeros(self.dW[i].shape)
            self.db[i] = np.zeros(self.db[i].shape)
            
        '''
            if self.bn:
                if i < self.num_bn_layers:
                    self.bn_layers[i].dgamma = np.zeros(self.bn_layers[i].dgamma.shape)
                    self.bn_layers[i].dbeta = np.zeros(self.bn_layers[i].dbeta.shape)
        '''
        
        return True

    def step(self):
        
        for i in range(len(self.W)):
            if (self.momentum > 0):
                self.W_running_average[i] = np.multiply(self.momentum,self.W_running_average[i]) - np.multiply(self.lr, self.dW[i])
                self.W[i] = self.W[i] + self.W_running_average[i]
                self.b_running_average[i] = np.multiply(self.momentum,self.b_running_average[i]) - np.multiply(self.lr, self.db[i])
                self.b[i] = self.b[i] + self.b_running_average[i]            
                
            else :

                self.W[i] = self.W[i] - (self.lr*self.dW[i])
                self.b[i] = self.b[i] - (self.lr*self.db[i])

                if i == 0 and self.num_bn_layers > 1:
                    self.bn_layers[i].gamma = self.bn_layers[i].gamma - (self.lr*self.bn_layers[i].dgamma)
                    self.bn_layers[i].beta = self.bn_layers[i].beta - (self.lr*self.bn_layers[i].dbeta)
 
        return True

      
    def backward(self, labels):
        
        loss = self.criterion.forward(self.activations[-1].state, labels)
        dLdy = self.criterion.derivative() / self.batch_size
        
        for i in range(self.nlayers-1, -1, -1):
                
            dLdL1 = np.multiply(dLdy, self.activations[i].derivative())
            
            if i < self.num_bn_layers:
                dLdB = dLdL1
                dBdy = self.bn_layers[i].backward(dLdB)
                #dLdL1 = np.transpose(np.transpose(dLdB) * dBdy)
                dLdL1 = np.transpose(dBdy)
            
            dLdy = np.matmul(dLdL1,np.transpose(self.W[i]))

            if i > 0:
                self.dW[i] = np.matmul(np.transpose(self.activations[i-1].state),dLdL1)       
            else:
                self.dW[i] = np.matmul(np.transpose(self.x),dLdL1)
            
            self.db[i] = np.mean(dLdL1, axis=0) * self.batch_size
        
        return loss
        

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []
    
    # Setup ...

    train_index_array = np.arange(trainx.shape[0])

    for e in range(nepochs):

        # Per epoch setup ...
        np.random.shuffle(train_index_array)
        
        print("epoch:"+str(e))
        
        start = 0
        end = 0
        loss = 0
        errors = 0
        for b in range(0, len(trainx), batch_size):

            # Train ...
            end = start + batch_size
            index_batch = train_index_array[start:end]
            train_batch_x = trainx[index_batch]
            train_batch_y = trainy[index_batch]
            
            predictions = np.argmax(mlp.forward(train_batch_x), axis = 0)
            labels = np.argmax(train_batch_y, axis = 0)
            errors = errors + ((len(predictions) - np.sum((predictions == labels)*1))/batch_size)
            
            loss = loss + (sum(mlp.backward(train_batch_y))/batch_size)
            
            mlp.step()
            
        
            start = start + batch_size
            
            mlp.zero_grads()
            
            
        
        avarage_epoch_loss = loss / (trainx.shape[0]/float(batch_size))
        training_losses.append(avarage_epoch_loss)       
        training_errors.append(errors/(trainx.shape[0]/float(batch_size)))
        
        val_loss = 0
        val_errors = 0
        val_start = 0
        val_end = 0
        
        for b in range(0, len(valx), batch_size):
            # Val ...
            val_end = val_start + batch_size
            predictions = np.argmax(mlp.forward(valx[val_start:val_end]), axis = 0)
            labels = np.argmax(valy[val_start:val_end], axis = 0)
            val_errors = val_errors + (len(predictions) - np.sum((predictions == labels)*1))
            val_loss = val_loss + (sum(mlp.backward(valy[val_start:val_end]))/batch_size)
            val_start = val_start + batch_size
         
        avarage_val_epoch_loss = val_loss / (valx.shape[0]/float(batch_size))
        validation_losses.append(avarage_val_epoch_loss)
        validation_errors.append(val_errors/len(valx))
        print(val_errors/len(valx))
        # Accumulate data...

    # Cleanup ...

    start = 0
    end = 0
    test_loss = 0
    test_errors = 0

    for b in range(0, len(testx), batch_size):
        
        end = start + batch_size
        predictions = np.argmax(mlp.forward(testx[start:end]), axis = 0)
        labels = np.argmax(testy[start:end], axis = 0)
        test_errors = test_errors + (len(predictions) - np.sum((predictions == labels)*1))
        test_loss = test_loss + (sum(mlp.backward(testy[start:end]))/batch_size)
            
        start = start + batch_size
         
    avarage_test_loss = test_loss / (testx.shape[0]/float(batch_size))

    print("test loss: " + str(avarage_test_loss))
    print("test erros: " + str(test_errors))
        

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)






