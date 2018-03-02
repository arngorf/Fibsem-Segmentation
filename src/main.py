import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="3"

from ModelConfigs import ModelConfigs

import CNN.FibsemDataset as input_data


if __name__ '__main__':
    dataset_path = '../data/lausanne'
    #dataset_path = '/scratch/xkv467/lausanne'
    batch_size = 32
    feature_shape = (25, 25, 25)
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]

    dataset = FibsemDataset(dataset_path,
                            batch_size,
                            feature_shape,
                            img_class_map,
                            )
    train_model(dataset_id, model)