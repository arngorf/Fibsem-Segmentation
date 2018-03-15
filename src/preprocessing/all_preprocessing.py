from .affine_augmentation import affine_augmentation
from .foveation import foveation
from .normalize import normalize
from keras.layers import GaussianNoise

def all_preprocessing(model, parameters, **kwargs):

    if parameters == 'all' or 'rotation' in parameters:
        model = affine_augmentation(model, **kwargs)
    if parameters == 'all' or 'normalize' in parameters:
        model = normalize(model, **kwargs)
    if parameters == 'all' or 'foveation' in parameters:
        model = foveation(model, **kwargs)
    if parameters == 'all' or 'noise' in parameters:
        stddev = 2.
        model.add(GaussianNoise(stddev))

    return model