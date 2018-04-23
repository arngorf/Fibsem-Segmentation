from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda
from scipy.signal import gaussian
import numpy as np
import tensorflow as tf

class GeometricTransformationLayer(Layer):

    def __init__(self,
                 linear_deformation=True,
                 rotation=True,
                 non_linear_resampling=True,
                 target_shape=None,
                 force_use_in_test_phase=False,
                 **kwargs):
        super(GeometricTransformationLayer, self).__init__(**kwargs)

        self.linear_deformation = linear_deformation
        self.rotation = rotation
        self.non_linear_resampling = non_linear_resampling
        self.target_shape = target_shape

        self.force_use_in_test_phase = force_use_in_test_phase

    def build(self, input_shape):

        self.source_shape = input_shape

        self.ci = (input_shape[1] - 1.0) / 2.0
        self.cj = (input_shape[2] - 1.0) / 2.0
        self.ck = (input_shape[3] - 1.0) / 2.0

        i,j,k = np.meshgrid(np.linspace(-self.ci, self.ci, input_shape[1]),
                            np.linspace(-self.cj, self.cj, input_shape[2]),
                            np.linspace(-self.ck, self.ck, input_shape[3]))

        ones = np.ones(i.size)

        self.points = K.variable(np.vstack([j.flatten(), i.flatten(), k.flatten(), ones]))

        if self.target_shape:
            target_shape = self.target_shape
        else:
            target_shape = ((2*input_shape[1])//3, (2*input_shape[2])//3, (2*input_shape[3])//3)

        d0 = input_shape[1] / 2
        d1 = target_shape[0] / 2
        dd = d0 - d1

        a = dd / (d1**3)

        ct = self.points[:3,:]

        ct = a*ct**3.0

        ct = K.concatenate([ct, K.zeros((1, i.size))],
                           axis=0)

        self.coordinate_transform = ct

        super(GeometricTransformationLayer, self).build(input_shape)

    def call(self, inputs, training=None):

        if not (self.rotation or
                self.linear_deformation or
                self.non_linear_resampling):
            return inputs

        input_shape = self.source_shape
        batch_size = input_shape[0]

        def rotate(points):

            rv = np.random.random(3)

            theta = rv[0] * np.pi * 2
            phi   = rv[1] * np.pi * 2
            z     = rv[2] * 2.0

            r  = np.sqrt(z)
            Vx = np.sin(phi) * r
            Vy = np.cos(phi) * r
            Vz = np.sqrt(2.0 - z)

            st = np.sin(theta)
            ct = np.cos(theta)

            Sx = Vx * ct - Vy * st
            Sy = Vx * st + Vy * ct

            R00 = Vx * Sx - ct
            R01 = Vx * Sy - st
            R02 = Vx * Vz

            R10 = Vy * Sx + st
            R11 = Vy * Sy - ct
            R12 = Vy * Vz

            R20 = Vz * Sx
            R21 = Vz * Sy
            R22 = 1.0 - z

            R = K.variable(np.zeros((4,4), dtype=np.float32))

            R = K.variable([[R00, R01, R02, 0],
                            [R10, R11, R12, 0],
                            [R20, R21, R22, 0],
                            [  0,   0,   0, 1]])

            return K.dot(R, points)

        def linear_warp(points):

            w_diagonal = 0.15
            w_off_diagonal = 0.15

            rv_diag = (np.random.random(3) - 0.5) * 2.0 * w_diagonal + 1.0
            rv_offd = (np.random.random(6) - 0.5) * 2.0 * w_off_diagonal
            sign = (-1)**np.random.randint(0,2,3) # flips axes direction randomly
            rv_diag = rv_diag * sign

            S = K.variable([[rv_diag[0], rv_offd[0], rv_offd[1], 0],
                            [rv_offd[3], rv_diag[1], rv_offd[2], 0],
                            [rv_offd[4], rv_offd[5], rv_diag[2], 0],
                            [0,          0,          0,          1]])

            return K.dot(S, points)

        def non_linear_warp(points):

            return points + self.coordinate_transform

        def correct_center(points):

            T = K.variable([[1, 0, 0, self.ci],
                            [0, 1, 0, self.cj],
                            [0, 0, 1, self.ck],
                            [0, 0, 0, 1       ]])

            return K.dot(T, points)

        def nearest_neighbour_interpolation(points):
            points = K.round(points)

            w, h, d = input_shape[1], input_shape[2], input_shape[3]

            #if self.target_shape:
            #    w, h, d = self.target_shape[1:4]

            i = K.clip(points[0,:], 0, w-1)
            j = K.clip(points[1,:], 0, h-1)
            k = K.clip(points[2,:], 0, d-1)

            i = K.cast(i, dtype='int32')
            j = K.cast(j, dtype='int32')
            k = K.cast(k, dtype='int32')

            #indices = k*h*w + j*w + i
            indices = i*h*w + j*w + k

            y = tf.gather(K.reshape(inputs, (-1, w*h*d)), indices, axis=1)

            return K.reshape(y, (-1, w, h, d, 1))

        def linear_interpolation(points):

            w, h, d = input_shape[1], input_shape[2], input_shape[3]

            #if self.target_shape:
            #    w, h, d = self.target_shape[1:4]

            x = K.reshape(inputs, (-1, w*h*d))

            values = K.zeros_like(x)
            w_total = K.zeros((w*h*d), dtype=np.float32)

            for k_fun in [tf.floor, lambda z: tf.floor(z) + 1]:
                for j_fun in [tf.floor, lambda z: tf.floor(z) + 1]:
                    for i_fun in [tf.floor, lambda z: tf.floor(z) + 1]:

                        i = points[0,:]
                        j = points[1,:]
                        k = points[2,:]

                        i = i_fun(i)
                        j = j_fun(j)
                        k = k_fun(k)

                        i = K.clip(i, 0.0, w-1)
                        j = K.clip(j, 0.0, h-1)
                        k = K.clip(k, 0.0, d-1)

                        w1 = 1.0 - K.abs(K.clip(points[0,:], 0.0, w-1) - i)
                        w2 = 1.0 - K.abs(K.clip(points[1,:], 0.0, h-1) - j)
                        w3 = 1.0 - K.abs(K.clip(points[2,:], 0.0, d-1) - k)

                        i = K.cast(i, dtype='int32')
                        j = K.cast(j, dtype='int32')
                        k = K.cast(k, dtype='int32')

                        #indices = k*h*w + j*w + i
                        indices = i*h*w + j*w + k

                        y = tf.gather(x, indices, axis=1)

                        weight = w1*w2*w3
                        y = weight * y

                        w_total = w_total + weight

                        values = values + y

            w_total = K.clip(w_total, 1e-8, 8.0) # Just to be sure

            values = values / w_total

            return K.reshape(values, (-1, w, h, d, 1))

        def train_augmented_inputs(inputs):
            # THIS CODE HAS BEEN ALTERED TO NOT DO ANYTHING OTHER THAN SIMPLE PIXEL TO PIXEL REARRANGEMENTS

            '''
            points = K.variable(self.points)

            if self.rotation:
                points = rotate(points)

            if self.linear_deformation:
                points = linear_warp(points)

            if self.non_linear_resampling:
                points = non_linear_warp(points)

            points = correct_center(points)

            interpolant = linear_interpolation(points) #nearest_neighbour_interpolation(points)'''
            rv = np.random.random(4)

            if rv[0] > 0.5:
                inputs = K.reverse(inputs,axes=2)
            if rv[1] > 0.5:
                inputs = K.reverse(inputs,axes=3)
            if rv[2] > 0.5:
                inputs = K.permute_dimensions(inputs, (0,2,1,3,4))
            if rv[3] > 0.5:
                inputs = K.reverse(inputs,axes=1)

            return inputs

        def test_augmented_inputs(inputs):
            # THIS CODE HAS BEEN ALTERED TO NOT DO ANYTHING OTHER THAN SIMPLE PIXEL TO PIXEL REARRANGEMENTS

            '''
            points = K.variable(self.points)

            if self.non_linear_resampling:
                points = non_linear_warp(points)

            points = correct_center(points)

            interpolant = linear_interpolation(points) #nearest_neighbour_interpolation(points)'''

            return inputs # interpolant

        if self.force_use_in_test_phase:
            return train_augmented_inputs(inputs)

        return K.in_train_phase(train_augmented_inputs(inputs), test_augmented_inputs(inputs),
                                training=training)

    def compute_output_shape(self, input_shape):

        #if self.target_shape:
        #    target_shape = (input_shape[0], target_shape[0], target_shape[1], target_shape[2], input_shape[4])
        #    return self.target_shape

        return input_shape

    def get_config(self):

        config = {'linear_deformation': self.linear_deformation,
                  'rotation': self.rotation,
                  'non_linear_resampling': self.non_linear_resampling,
                  'force_use_in_test_phase': self.force_use_in_test_phase,
                  'target_shape': self.target_shape
                  }

        base_config = super(GeometricTransformationLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
