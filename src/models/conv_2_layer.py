from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.models import Sequential
import keras
import keras.backend as K
from preprocessing import all_preprocessing

def make_model(num_classes,
               conv_dropout_p=0.75,
               dense_dropout_p=0.5,
               name='conv_2_layer',
               **kwargs):

    name = name + '_' + str(conv_dropout_p) + '_' + str(dense_dropout_p)
    input_shape = (25, 25, 25)
    k_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    model = Sequential()

    model = all_preprocessing(model,
                              ['normalize', 'rotation', 'noise'],
                              input_shape=k_input_shape,
                              **kwargs,
                              )

    model.add(Conv3D(48, (5, 5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(conv_dropout_p))

    model.add(Conv3D(64, (4, 4, 4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(conv_dropout_p))

    model.add(Flatten())
    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(dense_dropout_p))

    model.add(Dense(num_classes))
    model.add(Activation('tanh')) #softmax

    return model, name, input_shape
