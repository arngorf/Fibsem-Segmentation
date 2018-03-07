from layers.FoveationLayer import FoveationLayer

def foveation(model, **kwargs):

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
        model.add(FoveationLayer(input_shape=input_shape))
    else:
        model.add(FoveationLayer())

    return model