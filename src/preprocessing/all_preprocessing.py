from .affine_augmentation import affine_augmentation
from .foveation import foveation
from .normalize import normalize
from .noise import noise
from keras.layers import Reshape

def all_preprocessing(x, parameters, **kwargs):

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
        reshape_layer = Reshape(input_shape, input_shape=input_shape)
        if 'functional_api' in kwargs and kwargs['functional_api'] == True:
            x = reshape_layer(x)
        else:
            x.add(reshape_layer)

    if parameters == 'all' or 'rotation' in parameters:
        x = affine_augmentation(x, **kwargs)

    if parameters == 'all' or 'normalize' in parameters:
        x = normalize(x, **kwargs)

    if parameters == 'all' or 'foveation' in parameters:
        x = foveation(x, **kwargs)

    if parameters == 'all' or 'noise' in parameters:
        x = noise(x, **kwargs)

    return x