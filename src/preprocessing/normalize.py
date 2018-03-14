from keras.layers import Lambda

def normalize(input_param, **kwargs):

    mean, std = kwargs['norm_params']
    a = 1./std

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
        layer = Lambda(lambda x: (x - mean)*a, input_shape=input_shape)
    else:
        layer = Lambda(lambda x: (x - mean)*a)

    if 'functional_api' in kwargs and kwargs['functional_api'] == True:
        result = layer(input_param)
    else:
        input_param.add(layer)

    return input_param