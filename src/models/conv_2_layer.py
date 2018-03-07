from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.models import Sequential
from layers.FoveationLayer import FoveationLayer
from layers.RotationLayer import RotationLayer
import keras
import keras.backend as K

def make_model(num_classes,
               name='conv_2_layer',
               conv_dropout_p=0.75,
               dense_dropout_p=0.5):

    name = name + '_' + str(conv_dropout_p) + '_' + str(dense_dropout_p)
    input_shape = (25, 25, 25)

    model = Sequential()

    model = preprocessing(model, 'all')

    model.add(Conv3D(48, (5, 5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(conv_dropout_p))

    model.add(Conv3D(96, (3, 3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(conv_dropout_p))

    model.add(Flatten())
    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(dense_dropout_p))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model, name, input_shape