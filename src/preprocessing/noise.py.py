from keras.layers import GaussianNoise

def noise(x, **kwargs):

    if 'noise_stddev' in kwargs:
        stddev = kwargs['stddev']
    else:
        stddev = 2.

    noise_layer = GaussianNoise(stddev)

    if 'functional_api' in kwargs and kwargs['functional_api'] == True:
        x = noise_layer(x)
    else:
        x.add(noise_layer)

    return x