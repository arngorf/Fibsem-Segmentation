from layers.RotationLayer import RotationLayer

def affine_augmentation(model, **kwargs):

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
        model.add(RotationLayer(input_shape=input_shape))
    else:
        model.add(RotationLayer())

    return model