from keras.models import Sequential
import keras
import keras.backend as K
from preprocessing import all_preprocessing

def make_model(num_classes,
               name='test_affine',
               **kwargs):

    input_shape = (25, 25, 25)
    k_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    model = Sequential()

    model = all_preprocessing(model, ['normalize', 'rotation'], input_shape=k_input_shape, force_use_in_test_phase=True, **kwargs)

    return model, name, input_shape