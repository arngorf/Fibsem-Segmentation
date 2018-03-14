from layers.RotationLayer import RotationLayer

def affine_augmentation(input_param, **kwargs):

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
        layer = RotationLayer(input_shape=input_shape)
    else:
        layer = RotationLayer()

    if 'functional_api' in kwargs and kwargs['functional_api'] == True:
        result = layer(input_param)
    else:
        input_param.add(layer)

    return input_param