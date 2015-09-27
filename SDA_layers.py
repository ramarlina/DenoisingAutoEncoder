import numpy  
import Noise
import Layers

class Trainer():
    def train(self, layers, X, y, n_iters=1):
        rows = numpy.arange(X.shape[0])
        for iters in range(n_iters):
            numpy.random.shuffle(rows)
            for idx in rows:
                tmp = X[[idx]]
                for i in range(len(layers)):
                    tmp = layers[i].activate(tmp)
                grad = layers[-1].backwardPass(y[[idx]])
                for i in range(len(layers)-1, -1, -1):
                    grad = layers[i].update(grad)
        return layers
        
class StackedDA():
    def __init__(self, structure, alpha=0.01):
        self.alpha = alpha
        self.structure = structure
        self.Layers = []
        print "Call \"pre_train(n_iters)\""
        
    def __repr__(self):
        return "\nStacked Denoising Autoencoder Structure:\t%s"%(" -> ".join([str(x) for x in self.structure]))
        
    def pre_train(self, X, epochs=1, noise_rate=0.3):
        self.structure = numpy.concatenate([[X.shape[1]], self.structure])
        self.X = X
        trainer = Trainer()
        print "Pre-training: "#, self.__repr__()
        for i in range(len(self.structure) - 1):
            print "Layer: %dx%d"%( self.structure[i], self.structure[i+1])
            s1 = Layers.SigmoidLayer(self.structure[i], self.structure[i+1], noise=Noise.SaltAndPepper(noise_rate))
            s2 = Layers.SigmoidLayer(self.structure[i+1], self.X.shape[1])
            s1, s2 = trainer.train([s1, s2], self.X, self.X, epochs)
            self.X = s1.activate(self.X)
            self.Layers.append(s1)
    
    def finalLayer(self, X, y, epochs=1, n_neurons=200):
        print "Final Layer" 
        V = self.predict(X)
        softmax = Layers.SoftmaxLayer(self.Layers[-1].W.shape[1], y.shape[1]) 
        softmax = Trainer().train([softmax], V, y, epochs)[0]
        self.Layers.append(softmax)
     
    def fine_tune(self, X, y, epochs=1):
        print "Fine Tunning" 
        self.Layers = Trainer().train(self.Layers, X, y, epochs)
     
    def predict(self, X):
        #print self
        tmp = X
        for L in self.Layers:
            tmp = L.activate(tmp)
        return tmp
               