import numpy
from dA import dA
from Loss import Loss
import Noise

class StackedDA():
    def __init__(self, structure, alpha=0.01, bias=True):
        self.alpha = alpha
        self.structure = structure
        self.W = []
        self.bias = bias
        
    def __repr__(self):
        print "\nStacked Denoising Autoencoder Structure:\t%s"%(" -> ".join([str(x) for x in self.structure]))
        
    def pre_train(self, X, step=1, bias=True):
        structure = numpy.concatenate([[X.shape[1]], self.structure])
        self.structure = structure
        self.V = X
        for idx in xrange(len(structure) - 1):
            print "\nPre-trainning %d-th layer with %d hidden units"%(idx + 1, structure[idx + 1])
            # trainning bottom sigmoid layers
            layer = dA(structure[idx], structure[idx + 1], self.V.shape[1], alpha=0.01, bias=True); 
            layer.train(self.V, self.V, step)
            Z = layer.predict(self.V)
            self.V = layer.hidden
            self.W.append(layer.W)
         
    def fine_tune(self, X, y, learning_layer=100, n_iters=1, alpha=0.01):
        self.n_output = len(y[0])
        self.structure[0] += int(self.bias)
        self.structure = numpy.concatenate([self.structure, [learning_layer], [self.n_output]])
        
        # adding softmax layer on top of stack
        layer = dA(self.V.shape[1], learning_layer, self.n_output, alpha=alpha, bias=False, isSoftmax=True)
        layer.train(self.V, y, 1)
        
        self.W.append(layer.W)
        self.W.append(layer.K)
        
        data = numpy.arange(len(X))
        for i in numpy.arange(n_iters):
            numpy.random.shuffle(data)
            for idx in data:
                self.forwardPass(X[idx])
                self.backwardPass(y[idx])
            
    def forwardPass(self, X):
        count = 1
        X = numpy.concatenate([[1], X]) 
        self.activations = [X]
        self.change = []
        for W in self.W: 
            self.output = numpy.tanh(numpy.dot(X, W))
            self.activations.append(self.output)
            count +=1
            X = self.output.ravel()
        
    def backwardPass(self, y):
        self.output = numpy.exp(self.output)/numpy.sum(numpy.exp(self.output))
        L = Loss().Cross_Entropy(self.output, y)
        print L
        gradient = L.dE_dY*(1 - self.output**2)
        k = len(self.W)
        while (k>0):
            k = k - 1
            self.W[k] -= self.alpha * numpy.dot(self.activations[k].reshape(self.structure[k], 1), gradient.reshape(1, self.structure[k+1]))
            transfer = numpy.dot(gradient, self.W[k].T) 
            gradient =  transfer * (1 - self.activations[k]**2) 