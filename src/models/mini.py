from keras.layers import Activation, Dense, Flatten, Dropout
from keras.models import Sequential

from preprocessing import all_preprocessing

def make_model(num_classes, **model_params):

    input_shape = (1, 1, 1)
    k_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    if 'normalize' in model_params:
        normalize = model_params['normalize']
    else:
        normalize = True

    if 'dense_dropout_p' in model_params:
        dense_dropout_p = model_params['dense_dropout_p']
    else:
        dense_dropout_p = 0.25

    model = Sequential()

    model = all_preprocessing(model, normalize=True, input_shape=k_input_shape, **model_params)

    model.add(Flatten())

    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(dense_dropout_p))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model, input_shape