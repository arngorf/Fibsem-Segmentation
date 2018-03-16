from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda
from scipy.signal import gaussian
import numpy as np
import tensorflow as tf

class RotationLayer(Layer):

    def __init__(self, linear_deformation=True, force_use_in_test_phase=False, **kwargs):
        super(RotationLayer, self).__init__(**kwargs)

        self.linear_deformation = linear_deformation
        self.force_use_in_test_phase = force_use_in_test_phase
        print('self.force_use_in_test_phase',self.force_use_in_test_phase)

    def build(self, input_shape):

        self.the_output_shape = input_shape

        ci = input_shape[1] / 2.
        cj = input_shape[2] / 2.
        ck = input_shape[3] / 2.

        i,j,k = np.meshgrid(np.linspace(-ci, ci, input_shape[1]),
                            np.linspace(-cj, cj, input_shape[2]),
                            np.linspace(-ck, ck, input_shape[3]))

        ones = np.ones(i.size)

        self.points = K.variable(np.vstack([j.flatten(), i.flatten(), k.flatten(), ones]))

        self.center_correct_column = K.variable([ci, cj, ck, 1])

        super(RotationLayer, self).build(input_shape)

    def call(self, inputs, training=None):

        def augmented_inputs():

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

            R = K.variable([[R00, R01, R02, self.center_correct_column[0]],
                            [R10, R11, R12, self.center_correct_column[1]],
                            [R20, R21, R22, self.center_correct_column[2]],
                            [0,     0,   0, self.center_correct_column[3]]])

            if self.linear_deformation:

                rv_diag = (np.random.random(3) - 0.5) * 0.3 + 1
                rv_offd = (np.random.random(6) - 0.5) * 0.3

                S = K.variable([[rv_diag[0], rv_offd[0], rv_offd[1], 0],
                                [rv_offd[3], rv_diag[1], rv_offd[2], 0],
                                [rv_offd[4], rv_offd[5], rv_diag[2], 0],
                                [0,          0,          0,          1]])

                new_points = K.dot(S, self.points)
                new_points = K.dot(R, new_points)

            else:

                new_points = K.dot(R, self.points)

            new_points = K.round(new_points)

            w = self.the_output_shape[1]
            h = self.the_output_shape[2]
            d = self.the_output_shape[3]

            i = K.cast(K.clip(new_points[0,:], 0, w-1), dtype='int32')
            j = K.cast(K.clip(new_points[1,:], 0, h-1), dtype='int32')
            k = K.cast(K.clip(new_points[2,:], 0, d-1), dtype='int32')

            indices = k*h*w + j*w + i

            y = tf.gather(K.reshape(inputs, (-1, w*h*d)), indices, axis=1)

            return K.reshape(y, (-1, w, h, d, 1))

        if self.force_use_in_test_phase:
            return augmented_inputs()

        return K.in_train_phase(augmented_inputs(), inputs,
                                training=training)

    def compute_output_shape(self, input_shape):
        output_shape = self.the_output_shape
        return output_shape

