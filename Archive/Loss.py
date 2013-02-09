import numpy

class Loss():
    def __init__(self):
        self.label = "Uninitialized Loss Function"
        
    def MSE(self, x, y):
        self.dE_dY = -(y - x)
        self.E = numpy.sum(self.dE_dY**2)
        self.label = "MSE: "
        return self
        
    def __repr__(self):
        return self.label+ str(self.E)
        
    def Cross_Entropy(self, x, y):
        def loss(x, o):
            return 0.5*(x - o)**2
        zeros = numpy.where(y==0)[0]
        ones = numpy.where(y==1)[0]
        other = numpy.where(numpy.sin(y*numpy.pi)>0.01)[0]
        self.E = numpy.zeros(y.shape)
        self.E[zeros] = - numpy.log(1 - x[zeros]) 
        self.E[ones] = - numpy.log(x[ones]) 
        self.E[other] = loss(y[other], x[other])
        self.E = numpy.sum(self.E**2)
        
        # compute gradients
        self.dE_dY = - (y - x)
        self.label = "Log Loss: "
        return self