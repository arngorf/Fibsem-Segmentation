import os
from train import train_model
from test import test_model
from predict import predict_image
from itertools import product
from tqdm import tqdm
import time
import pickle

from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

limit_memory = True

if limit_memory:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    set_session(tf.Session(config=config))

from ModelsManager import ModelsManager
from PatchDataset import PatchDataset

def get_image_stack(cidx, depth):
    assert depth % 2 == 1

    s = depth // 2

    left = []
    right = []

    for idx in range(1,s+1):

        left_idx = cidx - idx
        left_idx = abs(left_idx)

        #left_path = '/home/dith/Dropbox/Fibsem-Segmentation/data/lausanne/image_dir/lausanne_' + str(cidx - idx) + '.png'
        left_path = '/scratch1/xkv467/lausanne_unregistered/image_dir/lausanne_' + str(left_idx) + '.png'
        left.append(np.array(Image.open(left_path), dtype=np.float32))

        right_idx = cidx + idx
        if right_idx > 1064:
            how_much_above = right_idx - 1064
            right_idx = 1064 - how_much_above

        #right_path = '/home/dith/Dropbox/Fibsem-Segmentation/data/lausanne/image_dir/lausanne_' + str(cidx + idx) + '.png'
        right_path = '/scratch1/xkv467/lausanne_unregistered/image_dir/lausanne_' + str(right_idx) + '.png'
        right.append(np.array(Image.open(right_path), dtype=np.float32))

    #center_path = '/home/dith/Dropbox/Fibsem-Segmentation/data/lausanne/image_dir/lausanne_' + str(cidx) + '.png'
    center_path = '/scratch1/xkv467/lausanne_unregistered/image_dir/lausanne_' + str(cidx) + '.png'

    center_img = np.array(Image.open(center_path), dtype=np.float32)
    result = left
    result.reverse()
    result.append(center_img)
    result.extend(right)

    return np.stack(result)


def mini_test():
    dataset_path = '../data/lausanne'
    results_path = '../results'
    batch_size = 32
    #img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]] # other, membrane + synapse
    #img_class_map = [[0, 7], [1], [2], [3, 4, 8], [6], [5]] # 6 groups
    img_class_map = [[0, 1, 2, 5, 6, 7], [3, 4, 8]] # other, vesicles
    output_size = len(img_class_map)
    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    model_type = 'mini'
    model_id = 'mini'

    model_params = {'norm_params': norm_params,
                    'rotation':True,
                    'output_size': output_size,
                    'lr': 0.01,
                    }

    model_manager.new_model(model_type,
                            model_id,
                            **model_params,
                            )

    model_class = model_manager.get_model(model_id)
    input_shape = model_class.input_shape

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           )

    iterations_per_epoch = 131072//2
    max_epochs = 9

    train_model(dataset,
                model_class,
                batch_size,
                iterations_per_epoch,
                max_epochs,
                avg_grad_stop=False,
                )

    #model_class.summary()

def dropbox_and_preprocessing_effect():
    #dataset_path = '../data/lausanne'
    dataset_path = '/scratch1/xkv467/lausanne'
    results_path = '../results'
    batch_size = 32
    #img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    img_class_map = [[0, 7], [1], [2], [3, 4, 8], [6], [5]]
    img_class_map = [[0, 7], [1, 2, 3, 4, 8, 6, 5]]
    output_size = len(img_class_map)
    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    dropout_p_list = [0.25] #[0.35, 0.5, 0.65]

    for dropout_p in dropout_p_list:
        for preprocessing_idx in [1]:

            pp = [False, False, False, False]
            pp[preprocessing_idx] = True

            noise = False
            none, rot, fovea, linear = pp
            pp_affix = ['none', 'rot2', 'fovea', 'linear'][preprocessing_idx]

            model_type = 'conv_2_layer'
            model_id = 'conv_2_layer_' + pp_affix + '_' + str(dropout_p)

            model_params = {'norm_params': norm_params,
                            'output_size': output_size,
                            'lr': 0.01,
                            'rotation':rot,
                            'foveation':fovea,
                            'linear_deformation':linear,
                            }

            model_manager.new_model(model_type,
                                    model_id,
                                    **model_params,
                                    )

            model_class = model_manager.get_model(model_id)
            input_shape = model_class.input_shape

            dataset = PatchDataset(dataset_path,
                                   batch_size,
                                   input_shape,
                                   img_class_map,
                                   )

            iterations_per_epoch = 565000//2
            max_epochs = 32

            train_model(dataset,
                        model_class,
                        batch_size,
                        iterations_per_epoch,
                        max_epochs,
                        avg_grad_stop=False,
                        avg_grad_n=16
                        )

def train_single():
    #dataset_path = '../data/lausanne'
    dataset_path = '/scratch1/xkv467/lausanne'
    results_path = '../results'
    batch_size = 32
    #img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    img_class_map = [[0, 7], [1], [2], [3, 4, 8], [6], [5]]
    #img_class_map = [[0, 7], [1, 2, 3, 4, 8, 6, 5]]
    output_size = len(img_class_map)
    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    dropout_p = 0.35

    model_type = 'conv_2_layer'
    model_id = 'conv_2_layer_07_1_2_348_6_5'

    model_params = {'norm_params': norm_params,
                    'output_size': output_size,
                    'lr': 0.01,
                    'rotation':True,
                    'foveation':True,
                    'linear_deformation':False,
                    'conv_dropout_p':dropout_p,
                    'dense_dropout_p':dropout_p,
                    }

    model_manager.new_model(model_type,
                            model_id,
                            **model_params,
                            )

    model_class = model_manager.get_model(model_id)
    input_shape = model_class.input_shape

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           )

    iterations_per_epoch = 565000//4
    max_epochs = 32*4

    train_model(dataset,
                model_class,
                batch_size,
                iterations_per_epoch,
                max_epochs,
                avg_grad_stop=False,
                avg_grad_n=16
                )

def predict_single_image(img_number):

    #dataset_path = '../data/lausanne'
    #dataset_path = '/scratch1/xkv467/lausanne'
    dataset_path = '/scratch1/xkv467/lausanne_unregistered'
    #dataset_path = '../data/test_dataset'
    results_path = '../results'
    batch_size = 128
    #img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    img_class_map = [[0, 7], [1], [2], [3, 4, 8], [6], [5]]
    #img_class_map = [[0, 1, 2, 5, 6, 7], [3, 4, 8]] # other, vesicles
    #img_class_map = [[0, 7], [1, 2, 3, 4, 8, 6, 5]] # cytosol, rest


    output_size = len(img_class_map)

    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    saved_model = model_manager.get_model('conv_2_layer_07_1_2_348_6_5_more_reg')

    model = saved_model.load_model('best')

    epoch, train, test = saved_model.session_stats()

    input_shape = saved_model.input_shape

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           predict_only=True
                           )

    # consistency test:
    # Does model produce the same test acc as expected two times in a row?

    #test_acc = test_model(dataset, model)
    #print('test_acc:', test_acc)

    image = get_image_stack(img_number, input_shape[0])

    output = predict_image(dataset, model, image)
    #print(output.shape)

    #plt.imshow(image[image.shape[0]//2,:,:], cmap='gray')

    #segments = np.argmax(output, axis=2)

    #colors = ['red', 'blue', 'g', 'purple', 'cyan', 'gold']

    #for class_idx in range(6):
    #    if class_idx == 1:
    #        pass
            #plt.contour((segments==class_idx).astype(int), levels=[0.5], colors = [colors[class_idx]], linewidths=1)

    #plt.show()

    return output

def predict_range(start_idx, end_idx):
    save_path = '../prediction_results/07_1_2_348_6_5_final'
    save_path = '/scratch1/xkv467/07_1_2_348_6_5_final'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for idx in tqdm(range(start_idx, end_idx)):
        output = predict_single_image(idx)
        filename = os.path.join(save_path, 'lausanne_' + str(idx) + '.p')
        pickle.dump(output, open(filename, 'wb'))

if __name__ == '__main__':

    #mini_test()
    #dropbox_effect()
    #preprocessing_effect()
    #train_n_time(3)
    #predict_single_image(400)
    #train_single()
    #predict_range(0, 532 + 1)
    #predict_range(532, 1064 + 1)
    predict_range(107, 270 + 1)
    #predict_range(270, 532 + 1)
    #predict_range(645, 1064 + 1)

    #time.sleep(10)
    #dropbox_and_preprocessing_effect()
