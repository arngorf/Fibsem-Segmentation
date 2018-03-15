from layers.RotationLayer import RotationLayer

def affine_augmentation(x, **kwargs):

    layer = RotationLayer()

    if 'functional_api' in kwargs and kwargs['functional_api'] == True:
        result = layer(x)
    else:
        x.add(layer)

    return x