import os
from models import micro, mini, midi, conv_2_layer, conv_2_layer_non_linear, conv_2_layer_non_linear_2, conv_2_layer_pass_through, conv_2_layer_conf
from train import train_model
from test import test_model
from itertools import product
from tqdm import tqdm
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="3"

from ModelsManager import ModelsManager
from PatchDataset import PatchDataset

def mini_test():
    dataset_path = '../data/lausanne'
    results_path = '../results'
    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    num_classes = len(img_class_map)
    norm_params = (142.1053396892233, 30.96410819657719)

    model_manager = ModelsManager(results_path)


    train_params = mini.make_model(num_classes,
                                   norm_params=norm_params,
                                   )

    model, model_name, input_shape = train_params

    model_manager.new_model(model,
                            model_name,
                            input_shape,
                            num_classes,
                            lr = 0.0001,
                            )

    model_class = model_manager.get_model(model_name)
    input_shape = model_class.input_shape

    model_class.set_session('default')

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           norm_params=norm_params,
                           )

    iterations_per_epoch=131072
    max_epochs=32

    train_model(dataset,
                model_class,
                batch_size,
                iterations_per_epoch,
                max_epochs,
                avg_grad_stop=False,
                )

    model_class.summary()

def dropbox_effect():
    #dataset_path = '../data/lausanne'
    dataset_path = '/scratch/xkv467/lausanne'
    results_path = '../results'
    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    num_classes = len(img_class_map)
    #norm_params = (142.1053396892233, 30.96410819657719)
    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    conv_dropout_p_list = [0.25, 0.5, 0.75]
    dense_dropout_p_list = [0.25, 0.5, 0.75]

    for conv_dropout_p in conv_dropout_p_list:
        for dense_dropout_p in dense_dropout_p_list:

            train_params = conv_2_layer.make_model(num_classes,
                                                   conv_dropout_p=conv_dropout_p,
                                                   dense_dropout_p=dense_dropout_p,
                                                   norm_params=norm_params,
                                                   )

            model, model_name, input_shape = train_params

            model_manager.new_model(model,
                                    model_name,
                                    input_shape,
                                    num_classes,
                                    lr = 0.001,
                                    )

            model_class = model_manager.get_model(model_name)
            input_shape = model_class.input_shape

            model_class.set_session('default')

            dataset = PatchDataset(dataset_path,
                                   batch_size,
                                   input_shape,
                                   img_class_map,
                                   norm_params=norm_params,
                                   )

            iterations_per_epoch=524288
            max_epochs=64

            train_model(dataset,
                        model_class,
                        batch_size,
                        iterations_per_epoch,
                        max_epochs,
                        avg_grad_stop=True,
                        )

def dropbox_and_preprocessing_effect():
    #dataset_path = '../data/lausanne'
    dataset_path = '/scratch/xkv467/lausanne'
    results_path = '../results'
    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    num_classes = len(img_class_map)
    #norm_params = (142.1053396892233, 30.96410819657719)
    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    conv_dropout_p_list = [0.0] #[0.35, 0.5, 0.65]
    dense_dropout_p_list = [0.0] #[0.35, 0.5, 0.65]

    for conv_dropout_p in conv_dropout_p_list:
        for dense_dropout_p in dense_dropout_p_list:
            for preprocessing_idx in range(4):

                pp = [False, False, False, False]
                pp[preprocessing_idx] = True

                noise = False
                none, rot, fovea, linear = pp
                pp_affix = ['none', 'rot', 'fovea', 'linear'][preprocessing_idx]

                train_params = conv_2_layer.make_model(num_classes,
                                                       rot,
                                                       fovea,
                                                       noise,
                                                       linear,
                                                       conv_dropout_p,
                                                       dense_dropout_p,
                                                       norm_params=norm_params,
                                                       )

                model, model_name, input_shape = train_params

                model_name = 'conv_2_layer_' + pp_affix + '_' + str(conv_dropout_p) + '_' + str(dense_dropout_p)

                model_manager.new_model(model,
                                        model_name,
                                        input_shape,
                                        num_classes,
                                        lr = 0.001,
                                        )

                model_class = model_manager.get_model(model_name)
                input_shape = model_class.input_shape

                model_class.set_session('default')

                dataset = PatchDataset(dataset_path,
                                       batch_size,
                                       input_shape,
                                       img_class_map,
                                       norm_params=norm_params,
                                       )

                iterations_per_epoch=565000
                max_epochs=32

                train_model(dataset,
                            model_class,
                            batch_size,
                            iterations_per_epoch,
                        max_epochs,
                        avg_grad_stop=False,
                        avg_grad_n=16
                        )

                print(model_name)

def single_train():
    #dataset_path = '../data/lausanne'
    dataset_path = '/scratch/xkv467/lausanne'
    #dataset_path = '../data/test_dataset'
    results_path = '../results'
    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    #img_class_map = [[0], [1], [2]]
    num_classes = len(img_class_map)
    #norm_params = (142.1053396892233, 30.96410819657719)
    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    conv_dropout_p = 0.5
    dense_dropout_p = 0.5

    '''train_params = micro.make_model(num_classes,
                                    norm_params=norm_params,
                                    )'''

    '''train_params = conv_2_layer_pass_through.make_model(num_classes,
                                                        conv_dropout_p=conv_dropout_p,
                                                        dense_dropout_p=dense_dropout_p,
                                                        norm_params=norm_params,
                                                        )'''

    '''train_params = conv_2_layer.make_model(num_classes,
                                           conv_dropout_p,
                                           dense_dropout_p,
                                           norm_params=norm_params,
                                           )'''

    '''train_params = conv_2_layer_non_linear_2.make_model(num_classes,
                                                      norm_params=norm_params,
                                                      )'''

    rot = True
    fovea = False
    noise = False
    linear = True
    non_linear = False

    train_params = conv_2_layer_conf.make_model(num_classes,
                                                rot,
                                                fovea,
                                                noise,
                                                linear,
                                                non_linear,
                                                norm_params=norm_params,
                                                )

    model, model_name, input_shape = train_params
    model_name = 'long_conv_2_layer'

    #model_name = model_name+'_sigmoid_activations'

    model_manager.new_model(model,
                            model_name,
                            input_shape,
                            num_classes,
                            lr = 0.001, #0.001
                            )

    model_class = model_manager.get_model(model_name)

    #model_class.session_summary()

    model_class.summary()

    input_shape = model_class.input_shape

    model_class.set_session('default')

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           norm_params=norm_params,
                           )

    iterations_per_epoch= 524288//2
    max_epochs=64*2

    train_model(dataset,
                model_class,
                batch_size,
                iterations_per_epoch,
                max_epochs,
                avg_grad_stop=False,
                avg_grad_n=16,
                )

def preprocessing_effect():
    #dataset_path = '../data/lausanne'
    dataset_path = '/scratch/xkv467/lausanne'
    #dataset_path = '../data/test_dataset'
    results_path = '../results'
    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    #img_class_map = [[0], [1], [2]]
    num_classes = len(img_class_map)
    #norm_params = (142.1053396892233, 30.96410819657719)
    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    for rot, fovea, noise, linear, non_linear in tqdm(product([False, True], repeat=5)):

        train_params = conv_2_layer_conf.make_model(num_classes,
                                                    rot,
                                                    fovea,
                                                    noise,
                                                    linear,
                                                    non_linear,
                                                    norm_params=norm_params,
                                                    )

        model, model_name, input_shape = train_params

        model_manager.new_model(model,
                                model_name,
                                input_shape,
                                num_classes,
                                lr = 0.001, #0.001
                                )

        model_class = model_manager.get_model(model_name)

        #model_class.session_summary()

        model_class.summary()

        input_shape = model_class.input_shape

        model_class.set_session('default')

        dataset = PatchDataset(dataset_path,
                               batch_size,
                               input_shape,
                               img_class_map,
                               norm_params=norm_params,
                               )

        iterations_per_epoch=524288//2 #4096
        max_epochs=16

        train_model(dataset,
                    model_class,
                    batch_size,
                    iterations_per_epoch,
                    max_epochs,
                    avg_grad_stop=False,
                    avg_grad_n=32,
                    )

def train_n_time(n):
    #dataset_path = '../data/lausanne'
    dataset_path = '/scratch/xkv467/lausanne'
    results_path = '../results'
    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    num_classes = len(img_class_map)
    norm_params = (142.1053396892233, 30.96410819657719)

    model_manager = ModelsManager(results_path)

    conv_dropout_p = 0.5
    dense_dropout_p = 0.5

    session_names = ['run_' + str(i+1) + '_new_data' for i in range(n)]

    train_params = conv_2_layer.make_model(num_classes,
                                           conv_dropout_p=conv_dropout_p,
                                           dense_dropout_p=dense_dropout_p,
                                           name='conv_2_layer_new_data',
                                           norm_params=norm_params,
                                           )

    model, model_name, input_shape = train_params

    model_manager.new_model(model,
                            model_name,
                            input_shape,
                            num_classes,
                            lr = 0.001,
                            )

    model_class = model_manager.get_model(model_name)
    input_shape = model_class.input_shape

    for session_name in session_names:

        model_class.set_session(session_name)

        dataset = PatchDataset(dataset_path,
                               batch_size,
                               input_shape,
                               img_class_map,
                               norm_params=norm_params,
                               )

        iterations_per_epoch=524288
        max_epochs=32

        train_model(dataset,
                    model_class,
                    batch_size,
                    iterations_per_epoch,
                    max_epochs,
                    avg_grad_stop=True,
                    avg_grad_n=16,
                    )

def predict_single_image(img_number):

    #dataset_path = '../data/lausanne'
    dataset_path = '/scratch/xkv467/lausanne'
    #dataset_path = '../data/test_dataset'
    results_path = '../results'
    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]

    num_classes = len(img_class_map)

    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    saved_model = model_manager.get_model('conv_2_layer_fovea_0.5_0.35')
    saved_model.load_model('latest', 'best')
    model = saved_model.model
    input_shape = saved_model.input_shape

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           norm_params=norm_params,
                           )

    # consistency test:
    # Does model produce the same test acc as expected two times in a row?

    for i in range(2):
        test_acc = test_model(dataset, model)
        print('test_acc:', test_acc)

    #predict_image(dataset, model, image)

if __name__ == '__main__':

    #mini_test()
    #dropbox_effect()
    #single_train()
    #preprocessing_effect()
    #train_n_time(3)
    predict_single_image(400)
    time.sleep(10)
    dropbox_and_preprocessing_effect()
