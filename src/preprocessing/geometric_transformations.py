from layers.GeometricTransformationLayer import GeometricTransformationLayer

def geometric_transformations(x, **kwargs):

    linear_deformation=False
    rotation=False
    non_linear_resampling=False
    target_shape=None
    force_use_in_test_phase=False

    if 'linear_deformation' in kwargs:
        linear_deformation = kwargs['linear_deformation']

    if 'rotation' in kwargs:
        rotation = kwargs['rotation']

    if 'non_linear_resampling' in kwargs:
        non_linear_resampling = kwargs['non_linear_resampling']

    if 'target_shape' in kwargs:
        target_shape = kwargs['target_shape']

    if 'force_use_in_test_phase' in kwargs:
        force_use_in_test_phase = kwargs['force_use_in_test_phase']

    layer = GeometricTransformationLayer(linear_deformation,
                                         rotation,
                                         non_linear_resampling,
                                         target_shape,
                                         force_use_in_test_phase,
                                         )

    if 'functional_api' in kwargs and kwargs['functional_api'] == True:
        x = layer(x)
    else:
        x.add(layer)

    return x