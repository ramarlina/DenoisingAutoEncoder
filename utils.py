import gzip
import cPickle
from PIL import Image
import numpy
import sys

def makeMultiClass(y):
    u = numpy.unique(y)
    coords = {}
    for idx in range(len(u)):
        coords[str(u[idx])] = idx
    V = numpy.zeros((len(y), len(u)))
    for idx in range(len(y)):
        V[idx, coords[str(y[idx])]] = 1
    return V

def load_mnist():   
    f = gzip.open("mnist/mnist.pkl.gz", 'rb')
    train, valid, test = cPickle.load(f)
    f.close()  
    return train

def scale(X, eps=1e-8):
  return (X - X.min())/ (X.max() + eps)

def saveTiles(X, img_shape, tile_shape, tile_spacing=(0, 0), filename="Results/res_DA.png"):
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

    H, W = img_shape
    Hs, Ws = tile_spacing

    out_array = numpy.zeros(out_shape, dtype='uint8')


    for tile_row in xrange(tile_shape[0]):
      for tile_col in xrange(tile_shape[1]):
          if tile_row * tile_shape[1] + tile_col < X.shape[0]:
              img = scale(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
              out_array[
                  tile_row * (H+Hs): tile_row * (H + Hs) + H,
                  tile_col * (W+Ws): tile_col * (W + Ws) + W
                  ] \
                  = img * 255
    
    img = Image.fromarray(out_array)
    img.save(filename) 