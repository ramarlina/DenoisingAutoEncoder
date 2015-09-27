import utils 
from SDA_layers import StackedDA  
 
def demo():
    X,y = utils.load_mnist()
    y = utils.makeMultiClass(y)
    
    # building the SDA
    sDA = StackedDA([100])

    # pre-trainning the SDA
    sDA.pre_train(X[:100], noise_rate=0.3, epochs=1)

    # saving a PNG representation of the first layer
    W = sDA.Layers[0].W.T[:, 1:]
    utils.saveTiles(W, img_shape= (28,28), tile_shape=(10,10), filename="results/res_dA.png")

    # adding the final layer
    sDA.finalLayer(X[:37500], y[:37500], epochs=2)

    # trainning the whole network
    sDA.fine_tune(X[:37500], y[:37500], epochs=2)

    # predicting using the SDA
    pred = sDA.predict(X[37500:]).argmax(1)

    # let's see how the network did
    y = y[37500:].argmax(1)
    e = 0.0
    for i in range(len(y)):
        e += y[i]==pred[i]

    # printing the result, this structure should result in 80% accuracy
    print "accuracy: %2.2f%%"%(100*e/len(y))

    return sDA
    
    
if __name__ == '__main__':
    demo()