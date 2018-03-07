from keras.layers import Activation, Dense, Flatten, Dropout
from keras.models import Sequential

from models.preprocessing.data_augmentation import data_augmentation
from models.preprocessing.normalize import normalize

def make_model(num_classes, name='midi', **kwargs):

    input_shape = (1, 1, 1)
    keras_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    model = Sequential()

    model = normalize(model, input_shape=keras_input_shape, **kwargs)

    model = data_augmentation(model, **kwargs)

    model.add(Flatten())

    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))

    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model, name, input_shape