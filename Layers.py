import numpy
import sys

null = lambda x: 0
identity = lambda x: x

class BaseLayer():
    def backwardPass(self, y):
        gradient = self.output - y + self.w_constraint(self.W)
        #sys.stdout.write( "\rError: %4.4f"%numpy.mean((y - self.output)**2) )
        print "%s %d: %4.4f"%(self.name, self.epoch, numpy.mean((y - self.output)**2))
        return gradient
        
    def update(self, transfer):
        gradient =  transfer * self.gradOutput
        transfer = numpy.dot(gradient, self.W[1:].T)
        self.W -= self.alpha * numpy.dot(self.input.reshape(self.n_input, 1), gradient.reshape(1, self.n_output))
        self.W = self.W * self.weight_decay
        self.epoch += 1
        return transfer

class SigmoidLayer(BaseLayer):
    def __init__(self, n_input, n_output, bias=True, alpha=0.01, weight_decay=1, w_constraint=null, noise=identity, W=None):
        if W==None:
            self.W = numpy.random.uniform(-0.1,0.1, (n_input + int(bias),n_output))
        else:
            self.W = W
        self.bias = bias
        self.alpha = alpha
        self.n_input = n_input + int(bias)
        self.n_output = n_output
        self.w_constraint = w_constraint
        self.noise_regularizer = noise
        self.weight_decay = weight_decay
        self.epoch = 0
        self.name = str(n_output) + "x" + str(n_input)
        
    def activate(self, X):
        X = self.noise_regularizer(X)
        if self.bias:
            X = numpy.column_stack([numpy.ones((X.shape[0],1)), X])
        self.input = X
        self.output = numpy.tanh(numpy.dot(self.input, self.W)) 
        self.gradOutput = 1 - self.output**2
        return self.output
        
     
class SoftmaxLayer(BaseLayer):
    def __init__(self, n_input, n_output, bias=True, alpha=0.01, weight_decay=1, w_constraint=null, noise=identity):
        self.W = numpy.random.uniform(-0.1,0.1, (n_input + int(bias),n_output))
        self.bias = bias
        self.alpha = alpha
        self.n_input = n_input + int(bias)
        self.n_output = n_output
        self.w_constraint = w_constraint
        self.noise_regularizer = noise
        self.weight_decay = weight_decay
        self.epoch = 0
        self.name = str(n_output) + "x" + str(n_input)
        
    def activate(self, X):
        X = self.noise_regularizer(X)
        if self.bias:
            X = numpy.column_stack([numpy.ones((X.shape[0],1)), X])
        self.input = X
        v = numpy.exp(numpy.dot(self.input, self.W))
        self.output = v/numpy.sum(v)
        self.gradOutput = self.output*(1 - self.output)
        return self.output

    def update(self, transfer):
        gradient =  transfer  
        transfer = numpy.dot(gradient, self.W[1:].T)
        self.W -= self.alpha * numpy.dot(self.input.reshape(self.n_input, 1), gradient.reshape(1, self.n_output))
        self.W = self.W * self.weight_decay
        self.epoch += 1
        return transfer
        