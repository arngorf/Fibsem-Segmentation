from layers.FoveationLayer import FoveationLayer

def foveation(x, **kwargs):

    layer = FoveationLayer()

    if 'functional_api' in kwargs and kwargs['functional_api'] == True:
        result = layer(x)
    else:
        x.add(layer)

    return x