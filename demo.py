import utils
#from SDA import StackedDA
from SDA_layers import StackedDA
import Noise
import Layers
import numpy

def demo(structure = [25**2, 23**2, 21**2,19**2,16**2, 15**2]):
    # Getting the data
    X,y = utils.load_mnist()
    
    
    autoencoder = StackedDA([100], alpha=0.01)
    autoencoder.pre_train(X[:1000], 10)
    
    y = utils.makeMultiClass(y)
    autoencoder.fine_tune(X[:1000], y[:1000], learning_layer=200, n_iters=20, alpha=0.01)

    W = autoencoder.W[0].T[:, 1:]
    W = utils.saveTiles(W, img_shape= (28,28), tile_shape=(10,10), filename="Results/res_dA.png")
    
def useLayers():
    X,y = utils.load_mnist()
    y = utils.makeMultiClass(y)
    
    # Layers
    sDA = StackedDA([100])
    sDA.pre_train(X[:1000], rate=0.5, n_iters=500)
    sDA.finalLayer(y[:1000], learner_size=200, n_iters=1)
    sDA.fine_tune(X[:1000], y[:1000], n_iters=1)
    pred = sDA.predict(X)
    
    W = sDA.Layers[0].W.T[:, 1:]
    W = utils.saveTiles(W, img_shape= (28,28), tile_shape=(10,10), filename="Results/res_dA.png")
    return pred, y
    
    
if __name__ == '__main__':
    useLayers()