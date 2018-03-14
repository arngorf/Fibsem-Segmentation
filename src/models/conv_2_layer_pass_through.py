from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, Cropping3D, Concatenate, Input
from keras.models import Model
import keras
import keras.backend as K
from preprocessing import all_preprocessing

def make_model(num_classes,
               conv_dropout_p=0.75,
               dense_dropout_p=0.5,
               name='conv_2_layer_pass_through',
               **kwargs):

    name = name + '_' + str(conv_dropout_p) + '_' + str(dense_dropout_p)
    input_shape = (25, 25, 25)
    k_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    inputs = Input(shape=(input_shape[0],input_shape[1],input_shape[2],1))

    processed = all_preprocessing(inputs, 'all', functional_api=True, **kwargs)

    conv_1 = Conv3D(48, (5, 5, 5), padding='valid')(processed)
    conv_1 = Activation('relu')(conv_1)

    pool_1 = MaxPooling3D(pool_size=(3, 3, 3))(conv_1)
    drop_1a = Dropout(conv_dropout_p)(pool_1)

    crop_1 = Cropping3D((7, 7, 7))(conv_1)
    drop_1b = Dropout(conv_dropout_p)(crop_1)

    conc_1 = Concatenate(axis=4)([drop_1a, drop_1b])

    conv_2 = Conv3D(96, (3, 3, 3), padding='valid')(conc_1)
    conv_2 = Activation('relu')(conv_2)

    pool_2 = MaxPooling3D(pool_size=(3, 3, 3))(conv_2)
    drop_2a = Dropout(conv_dropout_p)(pool_2)

    crop_2 = Cropping3D((2, 2, 2))(conv_2)
    drop_2b = Dropout(conv_dropout_p)(crop_2)

    conc_2 = Concatenate(axis=4)([drop_2a, drop_2b])

    flat = Flatten()(conc_2)
    fc_1 = Dense(150)(flat)
    fc_1 = Activation('relu')(fc_1)
    drop_3 = Dropout(dense_dropout_p)(fc_1)

    fc_2 = Dense(num_classes)(drop_3)
    predictions = Activation('softmax')(fc_2)

    model = Model(inputs=inputs, outputs=predictions)

    return model, name, input_shape
