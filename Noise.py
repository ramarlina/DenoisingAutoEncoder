import numpy

class Noise():
    def SaltAndPepper(self, X, rate=0.3):
        # Salt and pepper noise
        
        drop = numpy.arange(X.shape[1])
        numpy.random.shuffle(drop)
        sep = int(len(drop)*rate)
        drop = drop[:sep]
        X[:, drop[:sep/2]]=0
        X[:, drop[sep/2:]]=1
        return X
        
    def GaussianNoise(self, X, sd=0.5):
        # Injecting small gaussian noise
        X += numpy.random.normal(0, sd, X.shape)
        return X
        
    def MaskingNoise(self, X, rate=0.5):
        mask = (numpy.random.uniform(0,1, X.shape)<rate).astype("i4")
        X = mask*X
        return X
        
def SaltAndPepper(rate=0.3):
    # Salt and pepper noise
    def func(X):
        drop = numpy.random.uniform(0,1, X.shape)
        z = numpy.where(drop < 0.5*rate)
        o = numpy.where(numpy.abs(drop - 0.75*rate) < 0.25*rate)
        X[z]=0
        X[o]=1   
        return X
    return func
    
def GaussianNoise(self, sd=0.5):
    # Injecting small gaussian noise
    def func(X):
        X += numpy.random.normal(0, sd, X.shape)
        return X
    return func