from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv3D
from scipy.signal import gaussian
import numpy as np
import math

class InvariantDifferentialOperatorLayer(Layer):

    def __init__(self, scales, orders, padding='valid', **kwargs):

        self._scales = scales
        self._orders = orders
        self._padding = padding

        super(InvariantDifferentialOperatorLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        #assert input_shape[-1] == 2

        self.the_output_shape = input_shape
        print(input_shape, self._scales)
        k_size = input_shape[1]//2

        weight_list = []

        for j, sigma in enumerate(self._scales):

            gauss_inner_1d = gaussian(k_size, sigma)
            gauss_inner_2d = np.outer(gauss_inner_1d,gauss_inner_1d)
            gauss_inner_3d = np.repeat(gauss_inner_2d.reshape((1, k_size, k_size)), k_size, 0)

            for i in range(k_size):
                gauss_inner_3d[:,j,i] = np.multiply(gauss_inner_3d[:,j,i],gauss_inner_1d)

            gauss_inner_3d = gauss_inner_3d / np.sum(gauss_inner_3d)

            weight_list.append(gauss_inner_3d.reshape((1,k_size,k_size,k_size,1,1)))

        weights = np.concatenate(weight_list, 5)

        self.conv3d_layer = Conv3D(len(self._scales),
                                   (k_size, k_size, k_size),
                                   padding=self._padding,
                                   input_shape=(input_shape[0],
                                                input_shape[1],
                                                input_shape[2],
                                                1),
                                   trainable = False,
                                   use_bias = False,
                                   weights = weights)



        super(InvariantDifferentialOperatorLayer, self).build(input_shape)

    def call(self, x):

        w = self.the_output_shape[1]
        h = self.the_output_shape[2]
        d = self.the_output_shape[3]

        y = self.conv3d_layer(x)

        xp = y[:, 2:  , 1:-1, 1:-1, :]
        xm = y[:,  :-2, 1:-1, 1:-1, :]
        xc = y[:, 1:-1, 1:-1, 1:-1, :]

        yp = y[:, 1:-1, 2:  , 1:-1, :]
        ym = y[:, 1:-1,  :-2, 1:-1, :]
        yc = y[:, 1:-1, 1:-1, 1:-1, :]

        zp = y[:, 1:-1, 1:-1, 2:  , :]
        zm = y[:, 1:-1, 1:-1,  :-2, :]
        zc = y[:, 1:-1, 1:-1, 1:-1, :]

        xpyp = y[:, 2:  , 2:  , 1:-1, :]
        xpzp = y[:, 2:  , 1:-1, 2:  , :]
        ypzp = y[:, 1:-1, 2:  , 2:  , :]

        xmyp = y[:,  :-2, 2:  , 1:-1, :]
        xmzp = y[:,  :-2, 1:-1, 2:  , :]
        ymzp = y[:, 1:-1,  :-2, 2:  , :]

        xpym = y[:, 2:  ,  :-2, 1:-1, :]
        xpzm = y[:, 2:  , 1:-1,  :-2, :]
        ypzm = y[:, 1:-1, 2:  ,  :-2, :]

        xmym = y[:,  :-2,  :-2, 1:-1, :]
        xmzm = y[:,  :-2, 1:-1,  :-2, :]
        ymzm = y[:, 1:-1,  :-2,  :-2, :]

        fx  = (xp - xm) / 2.
        fy  = (yp - ym) / 2.
        fz  = (zp - zm) / 2.

        fxx = xp - xc + xm
        fyy = yp - yc + ym
        fzz = zp - zc + zm

        fxy = (xpyp - xmyp - xpym + xmym) / 4.
        fxz = (xpzp - xmzp - xpzm + xmzm) / 4.
        fyz = (ypzp - ymzp - ypzm + ymzm) / 4.

        fxfx = fx*fx
        fyfy = fy*fy
        fzfz = fz*fz

        fxfy = fx*fy
        fxfz = fx*fz
        fyfz = fy*fz

        result = [fxfx + fyfy + fzfz, fxx + fyy + fzz]
        #val = K.expand_dims(val, axis=???)
        return K.concatenate(result, axis=-1)

    def compute_output_shape(self, input_shape):
        return self.the_output_shape

'''
InvariantDifferentialOperatorLayer()

input: 45*45*45

scala 2.5, 5, 10 (3)
op til 3. ordens rot. inv. poly (N styks)

dvs. 45*45*45 -> 3*N x 1

exp 1) rot inv filt
exp 2) gauss afledte
'''
