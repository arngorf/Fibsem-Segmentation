from keras.layers import Activation
from keras.models import Sequential
import keras
import keras.backend as K
from preprocessing import all_preprocessing

def make_model(name='test_softmax',
               **kwargs):

    input_shape = (30,)
    k_input_shape = (30,)

    model = Sequential()

    model.add(Activation('softmax', input_shape=k_input_shape))

    return model, name, input_shape
