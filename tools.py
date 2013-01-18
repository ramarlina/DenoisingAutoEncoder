import numpy

def scale(X, eps=1e-8):
  return (X - X.min())/ (ndar.max() + eps)


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0)):
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
    return out_array