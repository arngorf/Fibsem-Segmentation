from keras.layers import Lambda

def normalize(model, **kwargs):

    if 'norm_params' in kwargs:

        mean, std = kwargs['norm_params']
        a = 1./std

        if 'input_shape' in kwargs:
            input_shape = kwargs['input_shape']
            model.add(Lambda(lambda x: (x - mean)*a,
                             input_shape=input_shape))
        else:
            model.add(Lambda(lambda x: (x - mean)*a))

    return model