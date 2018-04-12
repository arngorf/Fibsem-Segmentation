from keras.layers import Lambda

def normalize(x, **kwargs):

    mean, std = kwargs['norm_params']
    a = 1./std

    layer = Lambda(lambda x: (x - mean)*a, name='pp_normalize')

    if 'functional_api' in kwargs and kwargs['functional_api'] == True:
        x = layer(x)
    else:
        x.add(layer)

    return x