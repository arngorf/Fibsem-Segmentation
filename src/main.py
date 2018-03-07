import os
from models import micro, mini, midi, conv_2_layer
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

    #model, model_name, input_shape = micro.make_model(num_classes,
    #                                                  norm_params=norm_params)
    #model, model_name, input_shape = mini.make_model(num_classes,
    #                                                  norm_params=norm_params) #0.63
    #model, model_name, input_shape = midi.make_model(num_classes,


    conv_dropout_p_list = [0.1, 0.25, 0.5, 0.75, 0.9]
    dense_dropout_p_list = [0.1, 0.25, 0.5, 0.75, 0.9]

    for conv_dropout_p in conv_dropout_p_list:
        for dense_dropout_p in dense_dropout_p_list:

            train_params = conv_2_layer.make_model(num_classes,
                                                   norm_params=norm_params,
                                                   conv_dropout_p=conv_dropout_p,
                                                   dense_dropout_p=dense_dropout_p,
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

            iterations_per_epoch=1048576
            max_epochs=100

            train_model(dataset,
                        model_class,
                        batch_size,
                        iterations_per_epoch,
                        max_epochs,
                        avg_grad_stop=True,
                        )

            #model_class.session_summary()