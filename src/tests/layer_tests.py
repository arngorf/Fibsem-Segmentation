from models.layer_test_models import test_normalization_model, test_foveation_model, test_affine_model
from ModelsManager import ModelsManager
from PatchDataset import PatchDataset
import matplotlib.pyplot as plt
import numpy as np

def test_normalization_layer():
    #dataset_path = '../data/lausanne'
    #dataset_path = '/scratch/xkv467/lausanne'
    dataset_path = '../data/test_dataset'
    results_path = '../results'
    batch_size = 32
    #img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    img_class_map = [[0], [1], [2]]
    num_classes = len(img_class_map)
    norm_params = (142.1053396892233, 30.96410819657719)

    model_manager = ModelsManager(results_path)

    conv_dropout_p = 0.5
    dense_dropout_p = 0.5

    train_params = test_normalization_model.make_model(num_classes,
                                                       conv_dropout_p=conv_dropout_p,
                                                       dense_dropout_p=dense_dropout_p,
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

    model_class.summary()

    input_shape = model_class.input_shape

    model_class.set_session('default')

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           norm_params=norm_params,
                           )
    d, h, w = model_class.input_shape

    inputs = []
    outputs = []

    for i in range(100):
        x, _ = dataset.next_batch()
        x = x.reshape(batch_size, d, h, w, 1)
        output = model_class.model.predict(x)
        inputs.append(x)
        outputs.append(output)

    mu_before = np.mean(np.concatenate(inputs))
    mu_after = np.mean(np.concatenate(outputs))
    stddev_before = np.std(np.concatenate(inputs))
    stddev_after = np.std(np.concatenate(outputs))

    print("Before mean: {:04.2f}, stddev: {:04.2f}".format(mu_before, stddev_before))
    print("After mean: {:04.2f}, stddev: {:04.2f}".format(mu_after, stddev_after))

    x = inputs[50]
    z = outputs[50]

    for i in range(batch_size):

        for j in range(14):

            plt.subplot(4,7,j+1)
            plt.title('original '+str(j+5))
            plt.imshow(x[i,j+5,:,:,0], cmap='gray')


            plt.subplot(4,7,2*7+j+1)
            plt.title('normed '+str(j+5))
            plt.imshow(z[i,j+5,:,:,0], cmap='gray')

        plt.show()

def test_foveation_layer():
    #dataset_path = '../data/lausanne'
    #dataset_path = '/scratch/xkv467/lausanne'
    dataset_path = '../data/test_dataset'
    results_path = '../results'
    batch_size = 32
    #img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    img_class_map = [[0], [1], [2]]
    num_classes = len(img_class_map)
    norm_params = (142.1053396892233, 30.96410819657719)

    model_manager = ModelsManager(results_path)

    conv_dropout_p = 0.5
    dense_dropout_p = 0.5

    train_params = test_foveation_model.make_model(num_classes,
                                                   conv_dropout_p=conv_dropout_p,
                                                   dense_dropout_p=dense_dropout_p,
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

    model_class.summary()

    input_shape = model_class.input_shape

    model_class.set_session('default')

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           norm_params=norm_params,
                           )
    d, h, w = model_class.input_shape

    inputs = []
    outputs = []

    for i in range(1):
        x, _ = dataset.next_batch()
        x = x.reshape(batch_size, d, h, w, 1)
        output = model_class.model.predict(x)
        inputs.append(x)
        outputs.append(output)

    mu_before = np.mean(np.concatenate(inputs))
    mu_after = np.mean(np.concatenate(outputs))
    stddev_before = np.std(np.concatenate(inputs))
    stddev_after = np.std(np.concatenate(outputs))

    print("Before mean: {:04.2f}, stddev: {:04.2f}".format(mu_before, stddev_before))
    print("After mean: {:04.2f}, stddev: {:04.2f}".format(mu_after, stddev_after))

    x = inputs[0]
    z = outputs[0]

    for i in range(batch_size):

        for j in range(14):

            plt.subplot(4,7,j+1)
            plt.title('original '+str(j+5))
            plt.imshow(x[i,j+5,:,:,0], cmap='gray')


            plt.subplot(4,7,2*7+j+1)
            plt.title('foveated '+str(j+5))
            plt.imshow(z[i,j+5,:,:,0], cmap='gray')

        plt.show()

def test_affine_layer():
    #dataset_path = '../data/lausanne'
    #dataset_path = '/scratch/xkv467/lausanne'
    dataset_path = '../data/test_dataset'
    results_path = '../results'
    batch_size = 32
    #img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    img_class_map = [[0], [1], [2]]
    num_classes = len(img_class_map)
    norm_params = (142.1053396892233, 30.96410819657719)

    model_manager = ModelsManager(results_path)

    conv_dropout_p = 0.5
    dense_dropout_p = 0.5

    train_params = test_affine_model.make_model(num_classes,
                                                conv_dropout_p=conv_dropout_p,
                                                dense_dropout_p=dense_dropout_p,
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

    model_class.summary()

    input_shape = model_class.input_shape

    model_class.set_session('default')

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           norm_params=norm_params,
                           )
    d, h, w = model_class.input_shape

    inputs = []
    outputs = []

    for i in range(1):
        x, _ = dataset.next_batch()
        x = x.reshape(batch_size, d, h, w, 1)
        output = model_class.model.predict(x)
        inputs.append(x)
        outputs.append(output)

    mu_before = np.mean(np.concatenate(inputs))
    mu_after = np.mean(np.concatenate(outputs))
    stddev_before = np.std(np.concatenate(inputs))
    stddev_after = np.std(np.concatenate(outputs))

    print("Before mean: {:04.2f}, stddev: {:04.2f}".format(mu_before, stddev_before))
    print("After mean: {:04.2f}, stddev: {:04.2f}".format(mu_after, stddev_after))

    x = inputs[0]
    z = outputs[0]

    for i in range(batch_size):

        for j in range(14):

            plt.subplot(4,7,j+1)
            plt.title('original '+str(j+5))
            plt.imshow(x[i,j+5,:,:,0], cmap='gray')


            plt.subplot(4,7,2*7+j+1)
            plt.title('affined '+str(j+5))
            plt.imshow(z[i,j+5,:,:,0], cmap='gray')

        plt.show()

def test_noise_layer():
    #dataset_path = '../data/lausanne'
    #dataset_path = '/scratch/xkv467/lausanne'
    dataset_path = '../data/test_dataset'
    results_path = '../results'
    batch_size = 32
    #img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    img_class_map = [[0], [1], [2]]
    num_classes = len(img_class_map)
    norm_params = (142.1053396892233, 30.96410819657719)

    model_manager = ModelsManager(results_path)

    conv_dropout_p = 0.5
    dense_dropout_p = 0.5

    train_params = test_normalization_model.make_model(num_classes,
                                                       conv_dropout_p=conv_dropout_p,
                                                       dense_dropout_p=dense_dropout_p,
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

    model_class.summary()

    input_shape = model_class.input_shape

    model_class.set_session('default')

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           norm_params=norm_params,
                           )
    d, h, w = model_class.input_shape

    inputs = []
    outputs = []

    for i in range(1):
        x, _ = dataset.next_batch()
        x = x.reshape(batch_size, d, h, w, 1)
        output = model_class.model.predict(x)
        inputs.append(x)
        outputs.append(output)

    mu_before = np.mean(np.concatenate(inputs))
    mu_after = np.mean(np.concatenate(outputs))
    stddev_before = np.std(np.concatenate(inputs))
    stddev_after = np.std(np.concatenate(outputs))

    print("Before mean: {:04.2f}, stddev: {:04.2f}".format(mu_before, stddev_before))
    print("After mean: {:04.2f}, stddev: {:04.2f}".format(mu_after, stddev_after))

    x = outputs[0]
    noise = np.random.normal(0, 1.0, x.shape)
    z = outputs[0] + noise

    for i in range(batch_size):

        for j in range(14):

            plt.subplot(4,7,j+1)
            plt.title('original '+str(j+5))
            plt.imshow(x[i,j+5,:,:,0], cmap='gray')


            plt.subplot(4,7,2*7+j+1)
            plt.title('normed '+str(j+5))
            plt.imshow(z[i,j+5,:,:,0], cmap='gray')

        plt.show()