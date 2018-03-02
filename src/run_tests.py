#from tests.dataset_tests import *
from PatchDataset import PatchDataset
from test import test_model
import keras
from layers.RotationLayer import RotationLayer
from layers.FoveationLayer import FoveationLayer
from tests.invariant_tests import *

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

custom_objects={'RotationLayer':RotationLayer, 'FoveationLayer':FoveationLayer}'''

if __name__ == '__main__':

    #validate_output_manually()
    #validate_test_output_manually()
    #benchmark_dataset(25000)
    #benchmark_test_dataset(25000)
    pass

