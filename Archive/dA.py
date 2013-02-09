"""
    Stochastic Denoising Autoencoder with dropouts and gaussian noise for learning high level representation of the feature space in an unsupervised fashion
    Author: Mendrikra Ramarlina
    email: m.ramarlina@gmail.com
    
    References: 
        Bengio, Y. (2007). Learning deep architectures for AI (Technical Report 1312).
        Universite de Montreal, dept. IRO.
        
        Bengio, Y., Vincent, P., Manzagol, P-A., & Larochelle, H (2008, February). 
        Extracting and Composing Robust Features with Denoising Autoencoders (Technical Report 1316). 
        Universite de Montreal, dept. IRO.
"""

import numpy
from Loss import Loss
from Noise import Noise    
import sys
        
class dA():
    def __init__(self, n_in, n_hidden, self_out, alpha = 0.000015, bias = True):
        self.n_hidden = n_hidden
        self.n_output = self_out
        self.n_input = n_in
        
        self.W = numpy.random.uniform(-0.1, 0.1, (n_in + int(bias), n_hidden))
        self.K = numpy.random.uniform(-0.1, 0.1, (n_hidden, self_out))
        
        self.alpha = alpha
        self.alpha_decay = 1-1e-3
        self.bias = bias        
        
        self.hidden = 0
        self.output = 0
        self.E = 0
        
        self.epoch = 0
        self.bestW = []
        self.errors = [1e6,1e6,1e6,1e6,1e6]
        self.bestScore = 1e6
        
    def predict(self, X):
        if self.bias:
            X = numpy.hstack([numpy.ones((len(X),1)), X])
        self.X = X  
        # activation of the noisy backprobagation hidden layer
        self.hidden = numpy.tanh(numpy.dot(self.X, self.W)) 
        self.output = numpy.tanh(numpy.dot(self.hidden, self.K))
        return self.output
        
    def train(self, X, y, n_iters=5, batch_size=1):
        self.batch_size = batch_size
        rows = numpy.arange(X.shape[0])
        self.stop = False
        ranges = numpy.arange(len(X))
        for i in range(n_iters):
            numpy.random.shuffle(ranges)
            for j in numpy.arange(len(X)):
                if not self.stop:                    
                    self.activate(X[ranges[j*batch_size:(j+1)*batch_size]])
                    self.update(y[ranges[j*batch_size:(j+1)*batch_size]])
                    self.epoch = self.epoch + 1
        
        #print "\nBest Score: ", self.bestScore, " at epoch ", self.bestEpoch
        #self.W = self.bestW
            
    def activate(self, X):
        # Salt and pepper noise
        X = Noise().SaltAndPepper(X, 0.5)
        
        if self.bias:
            X = numpy.hstack([numpy.ones((self.batch_size,1)), X])
           
        self.X = X
        
        # activation of the noisy backprobagation hidden layer
        self.hidden = numpy.tanh(numpy.dot(self.X, self.W)) 
                
        self.output = numpy.tanh(numpy.dot(self.hidden, self.K))
        
        return self.output      
        
    def update(self, y):
    
        L = Loss().MSE(self.output, y)
        
        # stopping criteria
        self.errors[self.epoch%5] =  numpy.mean(L.E**2)**0.5
        score = numpy.mean(self.errors)    
            
        # stop when error starts to diverge too much
        print " " , self.bestScore
        self.stop = score/self.bestScore > 1e60
        
        # save the best weights
        if score < self.bestScore:
            self.bestW = self.W
            self.bestScore = score
            self.bestEpoch = self.epoch
        norm_W = numpy.linalg.norm(self.W)
        sys.stdout.write( "\rEpoch %d: RMSE: %2.3f, Norm(W): %2.2f"%(self.epoch, numpy.mean((y-self.output)**2)**0.5, norm_W) )
        sys.stdout.flush()
        
        # gradients
        grad_outputs = L.dE_dY*(1 - self.output**2)
        dE_dK = numpy.dot(self.hidden.reshape(self.n_hidden, 1), grad_outputs.reshape(1, self.n_output))
        
        transfer = numpy.dot(grad_outputs, self.K.T)        
               
        # hidden layer
        grad_hidden =  transfer * (1 - self.hidden**2) 
        dE_dW = numpy.dot(self.X.T , grad_hidden)

        # updating weights
        self.K -= 1.2*self.alpha*dE_dK
        
        self.W -= self.alpha*dE_dW
        
        # weight decay
        #self.W = self.W*(1-1e-2)
        # linearly decaying learning rate
        #self.alpha = self.alpha * self.alpha_decay
    
 

        
        




    

