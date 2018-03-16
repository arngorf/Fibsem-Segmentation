from layers.RotationLayer import RotationLayer

def affine_augmentation(x, **kwargs):

    if 'force_use_in_test_phase' in kwargs:
        force_use_in_test_phase = kwargs['force_use_in_test_phase']
        print('here 1', force_use_in_test_phase)
        layer = RotationLayer(True, force_use_in_test_phase)
    else:
        print('here 2')
        layer = RotationLayer(True)

    if 'functional_api' in kwargs and kwargs['functional_api'] == True:
        result = layer(x)
    else:
        x.add(layer)

    return x