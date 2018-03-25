from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.models import Sequential
import keras
import keras.backend as K
from preprocessing import all_preprocessing
from keras.layers import Cropping3D

def make_model(num_classes,
               name='conv_2_layer_non_linear',
               **kwargs):

    conv_dropout_p = 0.5
    dense_dropout_p = 0.5

    input_shape = (25, 25, 25)
    target_shape = (15, 15, 15)
    k_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    model = Sequential()

    model = all_preprocessing(model,
                              normalize=True,
                              rotation=True,
                              foveation=False,
                              noise=True,
                              linear_deformation=False,
                              non_linear_resampling=True,
                              input_shape=k_input_shape,
                              target_shape=target_shape,
                              force_use_in_test_phase=False,
                              **kwargs)

    model.add(Cropping3D((5,5,5)))

    model.add(Conv3D(16, (7, 7, 7), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(conv_dropout_p))

    model.add(Conv3D(16, (7, 7, 7), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(conv_dropout_p))

    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(dense_dropout_p))

    model.add(Dense(num_classes))
    model.add(Activation('softmax')) #softmax

    return model, name, input_shape