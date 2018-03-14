from layers.FoveationLayer import FoveationLayer

def foveation(input_param, **kwargs):

    layer = FoveationLayer()

    if 'functional_api' in kwargs and kwargs['functional_api'] == True:
        result = layer(input_param)
    else:
        input_param.add(layer)

    return input_param