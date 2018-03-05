from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten
from models.preprocessing.normalize import normalize
from models.preprocessing.data_augmentation import data_augmentation


def make_model(num_classes, name='micro', **kwargs):

    input_shape = (9, 9, 9)
    keras_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    model = Sequential()

    model = normalize(model, input_shape=keras_input_shape, **kwargs)

    model = data_augmentation(model, **kwargs)

    model.add(Flatten())

    model.add(Dense(num_classes))

    model.add(Activation('softmax'))

    return model, name, input_shape