def make_model(name = 'rf_2_layer', conv_dropout_p = 0.75, dense_dropout_p = 0.5):

    name = name + '_' + str(conv_dropout_p) + '_' + str(dense_dropout_p)

    model = Sequential()

    model.add(RotationLayer(input_shape=(output_feature_shape[0],
                                         output_feature_shape[1],
                                         output_feature_shape[2],
                                         1)))
    model.add(FoveationLayer())

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

    return model, name