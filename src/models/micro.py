from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential

from preprocessing import all_preprocessing

def make_model(num_classes, name='micro', **kwargs):

    input_shape = (25, 25, 25)
    k_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    model = Sequential()

    #model = all_preprocessing(model, 'all', input_shape=k_input_shape, **kwargs)
    model = all_preprocessing(model, ['normalize', 'noise'], input_shape=k_input_shape, **kwargs)

    model.add(Flatten())

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model, name, input_shape