from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv3D
from scipy.signal import gaussian
import numpy as np

class FoveationLayer(Layer):

    def __init__(self, **kwargs):
        super(FoveationLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        #assert input_shape[-1] == 2

        #self.the_output_shape = (input_shape[0],input_shape[1],input_shape[2],input_shape[3],input_shape[4]//2)
        self.the_output_shape = input_shape
        ####

        gauss_inner_1d = gaussian(7,0.5)
        gauss_inner_2d = np.outer(gauss_inner_1d,gauss_inner_1d)
        gauss_inner_3d = np.repeat(gauss_inner_2d.reshape((1,7,7)), 7, 0)

        gauss_outer_1d = gaussian(7,7.0)
        gauss_outer_2d = np.outer(gauss_outer_1d,gauss_outer_1d)
        gauss_outer_3d = np.repeat(gauss_outer_2d.reshape((1,7,7)), 7, 0)

        for j in range(7):
            for i in range(7):
                gauss_inner_3d[:,j,i] = np.multiply(gauss_inner_3d[:,j,i],gauss_inner_1d)
                gauss_outer_3d[:,j,i] = np.multiply(gauss_outer_3d[:,j,i],gauss_outer_1d)

        gauss_inner_3d = gauss_inner_3d / np.sum(gauss_inner_3d)
        gauss_outer_3d = gauss_outer_3d / np.sum(gauss_outer_3d)

        weights = np.concatenate([gauss_inner_3d.reshape((1,7,7,7,1,1)), gauss_outer_3d.reshape((1,7,7,7,1,1))],5)

        print(input_shape)

        self.conv3d_layer = Conv3D(2,
                                   (7, 7, 7),
                                   padding='same',
                                   input_shape=(input_shape[0],
                                                input_shape[1],
                                                input_shape[2],
                                                1),
                                   trainable = False,
                                   use_bias = False,
                                   weights = weights)

        ####

        cx = input_shape[2+1] // 2
        cy = input_shape[1+1] // 2
        cz = input_shape[0+1] // 2

        blend_weights = np.empty((1,input_shape[0+1],
                                  input_shape[1+1],
                                  input_shape[2+1],
                                  2), dtype=np.float32)

        for z in range(input_shape[0+1]):
            for y in range(input_shape[1+1]):
                for x in range(input_shape[2+1]):
                    blend_weights[0,z,y,x,0] = np.exp(-(((x-cx)/20.0)**2+((y-cy)/20.0)**2+((z-cz)/20.0)**2))**2
                    blend_weights[0,z,y,x,1] = 1 - blend_weights[0,z,y,x,0]

        self.W = K.variable(blend_weights)

        super(FoveationLayer, self).build(input_shape)

    def call(self, x):

        w = self.the_output_shape[1]
        h = self.the_output_shape[2]
        d = self.the_output_shape[3]

        y = self.conv3d_layer(x)

        inner = y[...,0] * self.W[...,0]
        outer = y[...,1] * self.W[...,1]

        return K.reshape(inner + outer, (-1, w, h, d, 1))

    def compute_output_shape(self, input_shape):
        output_shape = self.the_output_shape
        return output_shape

