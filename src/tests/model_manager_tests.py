from train import train_model
from test import test_model
from tqdm import tqdm

from ModelsManager import ModelsManager
from PatchDataset import PatchDataset

def dataset_real():
    dataset_dir = '../data/lausanne'
    batch_size = 32
    feature_shape = (1, 1, 1)
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1, 2]]

    dataset = PatchDataset(dataset_dir,
                           batch_size,
                           feature_shape,
                           img_class_map)

    return dataset

def dataset_synth():
    dataset_dir = '../data/test_dataset'
    batch_size = 32
    feature_shape = (25, 25, 25)
    img_class_map = [[0], [1], [2]]

    dataset = PatchDataset(dataset_dir,
                           batch_size,
                           feature_shape,
                           img_class_map)

    return dataset

def test_train_model_again():
    results_path = '../results'
    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    output_size = len(img_class_map)
    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    dataset = dataset_real()

    model_name = 'test_mini'

    model_manager.new_model('mini',
                            model_name,
                            output_size,
                            norm_params,
                            lr = 0.01,
                            )

    model_class = model_manager.get_model(model_name)
    input_shape = model_class.input_shape

    iterations_per_epoch=128
    max_epochs=16

    train_model(dataset,
                model_class,
                batch_size,
                iterations_per_epoch,
                max_epochs,
                avg_grad_stop=False,
                )

    max_epochs=32

    train_model(dataset,
                model_class,
                batch_size,
                iterations_per_epoch,
                max_epochs,
                avg_grad_stop=False,
                )

def test_acc_matches_stored_test_acc():
    results_path = '../results'
    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    output_size = len(img_class_map)
    norm_params = (126.04022903600975, 29.063149797089494)

    model_manager = ModelsManager(results_path)

    dataset = dataset_real()

    model_type = 'mini'
    model_id = 'test_mini'
    model_params = {'norm_params': norm_params,
                    'output_size': output_size,
                    'lr': 0.01}

    model_manager.new_model(model_type,
                            model_id,
                            **model_params,
                            )

    model_class = model_manager.get_model(model_id)

    input_shape = model_class.input_shape

    iterations_per_epoch=256
    max_epochs=2

    train_model(dataset,
                model_class,
                batch_size,
                iterations_per_epoch,
                max_epochs,
                avg_grad_stop=False,
                )

    epochs, train_accs, test_accs = model_class.session_stats()

    for i, epoch in enumerate(epochs):
        epoch_model = model_class.load_model(epoch)
        test_acc = test_model(dataset, epoch_model)
        print('test_acc:', test_accs[i], test_acc)
