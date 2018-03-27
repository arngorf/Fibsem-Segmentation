from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.models import Sequential
import keras
import keras.backend as K
from preprocessing import all_preprocessing
from keras.layers import Cropping3D

def make_model(num_classes,
               rotation,
               foveation,
               noise,
               linear_deformation,
               non_linear_resampling,
               name='conv_2_layer_conf',
               **kwargs):

    conv_dropout_p = 0.5
    dense_dropout_p = 0.5

    if non_linear_resampling:
        input_shape = (25, 25, 25)
    else:
        input_shape = (15, 15, 15)
    target_shape = (15, 15, 15)

    k_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    model = Sequential()

    if rotation:
        name += '_rotation'
    if foveation:
        name += '_foveation'
    if noise:
        name += '_noise'
    if linear_deformation:
        name += '_linear_deformation'
    if non_linear_resampling:
        name += '_non_linear_resampling'

    model = all_preprocessing(model,
                              normalize=True,
                              rotation=rotation,
                              foveation=foveation,
                              noise=noise,
                              linear_deformation=linear_deformation,
                              non_linear_resampling=non_linear_resampling,
                              input_shape=k_input_shape,
                              target_shape=target_shape,
                              **kwargs)
    if non_linear_resampling:
        model.add(Cropping3D((5,5,5)))

    model.add(Conv3D(32, (7, 7, 7), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(conv_dropout_p))

    model.add(Conv3D(32, (7, 7, 7), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(conv_dropout_p))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(dense_dropout_p))

    model.add(Dense(num_classes))
    model.add(Activation('softmax')) #softmax

    return model, name, input_shape