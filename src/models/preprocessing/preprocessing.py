from models.preprocessing.affine_augmentation import affine_augmentation
from models.preprocessing.foveation import foveation
from models.preprocessing.normalize import normalize

def preprocessing(model, parameters, **kwargs):

    if parameters == 'all' or 'rotation' in parameters:
        model = affine_augmentation(model, **kwargs)
    if parameters == 'all' or 'normalize' in parameters:
        model = normalize(model, **kwargs)
    if parameters == 'all' or 'foveation' in parameters:
        model = foveation(model, **kwargs)

    return model