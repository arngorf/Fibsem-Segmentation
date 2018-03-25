from keras.models import Sequential
import keras
import keras.backend as K
from preprocessing import all_preprocessing
from keras.layers import Cropping3D

def make_model(num_classes,
               name='test_all_layers',
               **kwargs):

    input_shape = (25, 25, 25)
    target_shape = (15, 15, 15)
    k_input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

    model = Sequential()

    model = all_preprocessing(model,
                              normalize=True,
                              rotation=True,
                              #foveation=True,
                              #linear_deformation=True,
                              non_linear_resampling=True,
                              input_shape=k_input_shape,
                              target_shape=target_shape,
                              force_use_in_test_phase=True,
                              **kwargs)

    model.add(Cropping3D((5,5,5)))

    return model, name, input_shape