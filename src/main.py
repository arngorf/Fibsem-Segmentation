import os
from models import micro
from train import train_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="3"

from ModelsManager import ModelsManager

from PatchDataset import PatchDataset

if __name__ == '__main__':
    dataset_path = '../data/lausanne'
    #dataset_path = '/scratch/xkv467/lausanne'
    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    num_classes = len(img_class_map)
    norm_params = (142.1053396892233, 30.96410819657719)

    model_manager = ModelsManager()

    model, model_name, input_shape = micro.make_model(num_classes,
                                                      norm_params=norm_params)

    model_manager.new_model(model, model_name, input_shape, num_classes)

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           norm_params=norm_params
                           )

    model_class = model_manager.get_model(model_name)
    model_class.set_session('session_a')

    train_model(dataset, model_class, batch_size)

    model_class.session_summary('session_a')