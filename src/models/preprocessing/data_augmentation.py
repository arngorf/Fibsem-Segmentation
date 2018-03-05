from layers.FoveationLayer import FoveationLayer
from layers.RotationLayer import RotationLayer

def data_augmentation(model, **kwargs):

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
        model.add(RotationLayer(input_shape=input_shape))
    else:
        model.add(RotationLayer())

    model.add(FoveationLayer())

    return model