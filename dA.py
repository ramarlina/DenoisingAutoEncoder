"""
    Denoising Autoencoder with dropouts and gaussian noise for learning high level representation of the feature space in an unsupervised fashion
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
import sys


class dA():
    def __init__(self, n_in, n_hidden, self_out, alpha = 0.000015, bias = True, stochastic=False):
        self.n_hidden = n_hidden
        
        self.W = numpy.random.uniform(-0.1, 0.1, (n_in + int(bias), n_hidden))
        self.K = numpy.random.uniform(-0.1, 0.1, (n_hidden, self_out))
        
        self.alpha = alpha
        self.alpha_decay = 1-1e-7
        self.isStochastic = stochastic
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
        
        # activation of the hidden layer
        self.hidden = numpy.tanh(numpy.dot(self.X, self.W))
        if self.isStochastic:
            self.hidden = (self.hidden > 0).astype("i4")
        
        # half the outgoing weights to approximate the geometric mean of neural network models trained with dropout
        self.output = numpy.dot(self.hidden, 0.5*self.K)
        return self.output
        
    def train(self, X, y, n_iters=5, batch_size=50):
        self.batch_size = batch_size
        rows = numpy.arange(X.shape[0])
        self.stop = False
        ranges = numpy.arange(len(X))
        for i in range(n_iters):
            for j in numpy.arange(len(X)/batch_size - batch_size):
                numpy.random.shuffle(ranges)
                if not self.stop:
                    self.activate(X[ranges[j*batch_size:(j+1)*batch_size]])
                    self.update(y[ranges[j*batch_size:(j+1)*batch_size]])
                    self.epoch = self.epoch + 1
        
        print "\nBest Score: ", self.bestScore, " at epoch ", self.bestEpoch
        self.W = self.bestW
            
    def activate(self, X):
        if self.bias:
            X = numpy.hstack([numpy.ones((self.batch_size,1)), X])
            
        self.X = X #+ numpy.random.normal(0,0.1, X.shape)
        
        # activation of the noisy backprobagation hidden layer
        self.hidden = numpy.tanh(numpy.dot(self.X, self.W)) 
        self.hidden += numpy.random.normal(0,1, self.hidden.shape)
        
        # random dropouts in the hidden layer
        self.drop = (numpy.random.uniform(0,1, self.n_hidden)>0.5).astype("i4")
        self.hidden = self.drop * (self.hidden) 
        
        # if stochastic units
        #if self.isStochastic:
        #self.hidden = (self.hidden > 0).astype("i4")
        
        
        self.output = numpy.dot(self.hidden, self.K)
        #print " -> ", self.output        
        return self.output
        
    def update(self, y):
        self.E = numpy.mean(numpy.abs(y - self.output), axis=0)      
        
        # stopping criteria
        self.errors[self.epoch%5] =  numpy.mean(self.E**2)**0.5
        score = numpy.mean(self.errors)    
            
        # stop when error starts to diverge too much
        print " " , self.bestScore
        self.stop = score/self.bestScore > 5
        
        # save the best weights
        if score < self.bestScore:
            self.bestW = self.W
            self.bestScore = score
            self.bestEpoch = self.epoch
        
        sys.stdout.write( "\rEpoch %d: RMSE: %4.4f"%(self.epoch, numpy.mean(self.E**2)**0.5) )
        sys.stdout.flush()
        
        # compute gradients
        dE_dY = self.E
        
        # output layer
        dY_dK = numpy.dot(self.hidden.T,(numpy.ones((self.batch_size,1))*dE_dY))
        
        # hidden layer
        dY_dZ = numpy.hstack(numpy.sum(self.K,axis=1))
        gradSigmoid = self.hidden*(1 - self.hidden)
        dY_dW = numpy.dot(self.X.T, numpy.ones((len(self.X),1))*dY_dZ*gradSigmoid)
        
        # updating weights
        self.K = self.K + 0.001*self.alpha*dY_dK
        self.W = self.W + 1.1*self.drop*self.alpha*dY_dW
        #self.W = self.W + 1.1*self.alpha*dY_dW
        self.W = numpy.abs(self.W)/(numpy.sum(numpy.abs(self.W)))
        
        # linearly decaying learning rate
        self.alpha = self.alpha * self.alpha_decay
    
 

import gzip
import cPickle
from PIL import Image
from tools import tile_raster_images
from matplotlib import pyplot as plt
plt.ion()

def load_mnist():   
    f = gzip.open("mnist/mnist.pkl.gz", 'rb')
    train, valid, test = cPickle.load(f)
    f.close()  
    return train[0]
    
def demo(structure = [25**2, 23**2, 21**2,19**2,16**2, 15**2]):
    # Get data
    X = load_mnist()
    structure=[10**2]
    structure = numpy.concatenate([[X.shape[1]], structure])
    
    stackedDA = []
    V = X
    
    print "\nStacked Denoising Autoencoder Structure:\n\n\t%s"%(" -> ".join([str(x) for x in structure]))
    
    for idx in xrange(len(structure) - 1):
        print "\nPre-trainning %d-th layer with %d hidden units"%(idx + 1, structure[idx + 1])
        autoencoder = dA(structure[idx], structure[idx + 1], X.shape[1], alpha=0.0001, bias=True, stochastic=False); 
        autoencoder.train(V,X,2)
        #Z = autoencoder.predict(V)
        V = autoencoder.hidden
        #stackedDA.append(autoencoder.W)

    W = autoencoder.W.T[:, 1:]
    W = tile_raster_images(W, img_shape= (28,28), tile_shape=(10,10))
    plt.plot(W)
    plt.show()
    img = Image.fromarray(W)
    img.save("c:/Git/dA.png")
    return V
    
if __name__ == '__main__':
    demo()
