from tests.dataset_tests import *
from tests.layer_tests import *
from PatchDataset import PatchDataset
from test import test_model
import keras
from layers.GeometricTransformationLayer import GeometricTransformationLayer
from layers.FoveationLayer import FoveationLayer
#from tests.invariant_tests import *

'''dataset_dir = '../data/lausanne/'
batch_size = 32
feature_shape = (45, 45, 45)
img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1, 2]]
dataset_mean = 142.1053396892233
dataset_std = 30.96410819657719

dataset = PatchDataset(dataset_dir,
                        batch_size,
                        feature_shape,
                        img_class_map,
                        norm_params=(dataset_mean, dataset_std))

custom_objects={'GeometricTransformationLayer':GeometricTransformationLayer,
                'FoveationLayer':FoveationLayer}'''

if __name__ == '__main__':

    validate_output_manually()
    #validate_test_output_manually()
    #benchmark_dataset(25000)
    #benchmark_test_dataset(25000)
    #validate_train_test()
    #test_normalization_layer(4)
    #test_foveation_layer(4)
    #test_affine_layer(4)
    #test_rotation_layer(4)
    #test_non_linear_layer(4)
    #test_all_layers(8)
    #test_noise_layer()
    #test_softmax_layer()

