from .geometric_transformations import geometric_transformations
from .foveation import foveation
from .normalize import normalize
from .noise import noise
from keras.layers import Reshape

def all_preprocessing(x, **kwargs):

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
        reshape_layer = Reshape(input_shape, input_shape=input_shape)
        if 'functional_api' in kwargs and kwargs['functional_api'] == True:
            x = reshape_layer(x)
        else:
            x.add(reshape_layer)

    if 'rotation' in kwargs or \
        'linear_deformation' in kwargs or \
        'non_linear_resampling' in kwargs:

        x = geometric_transformations(x, **kwargs)

    if 'normalize' in kwargs:
        x = normalize(x, **kwargs)

    if 'foveation' in kwargs:
        x = foveation(x, **kwargs)

    if 'noise' in kwargs:
        x = noise(x, **kwargs)

    return x