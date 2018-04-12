from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.models import Sequential
import keras
import keras.backend as K
from preprocessing import all_preprocessing

def make_model(num_classes,
               name='conv_2_layer',
               **model_params):

    if not 'normalize' in model_params:
        model_params['normalize'] = True

    if not 'rotation' in model_params:
        model_params['rotation'] = False

    if not 'foveation' in model_params:
        model_params['foveation'] = False

    if not 'noise' in model_params:
        model_params['noise'] = False

    if not 'linear_deformation' in model_params:
        model_params['linear_deformation'] = False

    if not 'non_linear_resampling' in model_params:
        model_params['non_linear_resampling'] = False

    if 'conv_dropout_p' in model_params:
        conv_dropout_p = model_params['conv_dropout_p']
    else:
        conv_dropout_p = 0.0

    if 'dense_dropout_p' in model_params:
        dense_dropout_p = model_params['dense_dropout_p']
    else:
        dense_dropout_p = 0.0


    name = name + '_' + str(conv_dropout_p) + '_' + str(dense_dropout_p)
    input_shape = (25, 25, 25)
    k_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    model = Sequential()

    model = all_preprocessing(model,
                              input_shape=k_input_shape,
                              **model_params)

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
    model.add(Activation('softmax')) #softmax

    return model, input_shape
