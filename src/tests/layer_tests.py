from models.layer_test_models import test_normalization_model, test_foveation_model, test_affine_model, test_rotation_model, test_non_linear_model, test_softmax, test_all
from ModelsManager import ModelsManager
from PatchDataset import PatchDataset
import matplotlib.pyplot as plt
import numpy as np

def test_normalization_layer(stop_after_n_examples):
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

        if i >= stop_after_n_examples:
            break

        for j in range(14):

            plt.subplot(4,7,j+1)
            plt.title('original '+str(j+5))
            plt.imshow(x[i,j+5,:,:,0], cmap='gray')


            plt.subplot(4,7,2*7+j+1)
            plt.title('normed '+str(j+5))
            plt.imshow(z[i,j+5,:,:,0], cmap='gray')

        plt.show()

def test_foveation_layer(stop_after_n_examples):
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

        if i >= stop_after_n_examples:
            break

        for j in range(14):

            plt.subplot(4,7,j+1)
            plt.title('original '+str(j+5))
            plt.imshow(x[i,j+5,:,:,0], cmap='gray')


            plt.subplot(4,7,2*7+j+1)
            plt.title('foveated '+str(j+5))
            plt.imshow(z[i,j+5,:,:,0], cmap='gray')

        plt.show()

def test_affine_layer(stop_after_n_examples):
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

        if i >= stop_after_n_examples:
            break

        for j in range(14):

            plt.subplot(4,7,j+1)
            plt.title('original '+str(j+5))
            plt.imshow(x[i,j+5,:,:,0], cmap='gray')

            plt.subplot(4,7,2*7+j+1)
            plt.title('affined '+str(j+5))
            plt.imshow(z[i,j+5,:,:,0], cmap='gray')

        plt.show()

def test_rotation_layer(stop_after_n_examples):
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

    train_params = test_rotation_model.make_model(num_classes,
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

        if i >= stop_after_n_examples:
            break

        for j in range(14):

            plt.subplot(4,7,j+1)
            plt.title('original '+str(j+5))
            plt.imshow(x[i,j+5,:,:,0], cmap='gray')

            plt.subplot(4,7,2*7+j+1)
            plt.title('affined '+str(j+5))
            plt.imshow(z[i,j+5,:,:,0], cmap='gray')

        plt.show()

def test_non_linear_layer(stop_after_n_examples):
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

    train_params = test_non_linear_model.make_model(num_classes,
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

    d, h, w = input_shape

    inputs = []
    outputs = []


    x = np.zeros((batch_size, d, h, w))
    for l in range(batch_size):
        bubblePeriod = l % 4 + 3
        for k in range(d):
            for j in range(h):
                for i in range(w):
                    x[l,k,j,i] = np.cos(np.pi*float(i - w//2)/bubblePeriod) \
                               * np.cos(np.pi*float(j - h//2)/bubblePeriod) \
                               * np.sin(np.pi*float(k - d//2)/bubblePeriod) \
                               + i*0.05 - w*0.1//2.

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

    x = x[:,5:-5,5:-5,5:-5,:]

    for i in range(batch_size):

        if i >= stop_after_n_examples:
            break

        for j in range(15):

            plt.subplot(6,5,j+1)
            plt.title('original '+str(j))
            plt.imshow(x[i,j,:,:,0], vmin=0, vmax=1, cmap='gray')

            plt.subplot(6,5,3*5+j+1)
            plt.title('resampled '+str(j))
            plt.imshow(z[i,j,:,:,0], vmin=0, vmax=1, cmap='gray')

        plt.show()

def test_all_layers(stop_after_n_examples):
    dataset_path = '../data/lausanne'
    #dataset_path = '/scratch/xkv467/lausanne'
    #dataset_path = '../data/test_dataset'
    results_path = '../results'
    batch_size = 32
    bubbles = False
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    #img_class_map = [[0], [1], [2]]
    num_classes = len(img_class_map)
    norm_params = (142.1053396892233, 30.96410819657719)

    model_manager = ModelsManager(results_path)

    conv_dropout_p = 0.5
    dense_dropout_p = 0.5

    train_params = test_all.make_model(num_classes,
                                       norm_params=norm_params,
                                       )

    model, model_name, input_shape = train_params

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           norm_params=norm_params,
                           )

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

    d, h, w = input_shape

    inputs = []
    outputs = []

    if bubbles:
        x = np.zeros((batch_size, d, h, w))
        for l in range(batch_size):
            bubblePeriod = l % 4 + 3
            for k in range(d):
                for j in range(h):
                    for i in range(w):
                        x[l,k,j,i] = np.cos(np.pi*float(i - w//2)/bubblePeriod) \
                                   * np.cos(np.pi*float(j - h//2)/bubblePeriod) \
                                   * np.cos(np.pi*float(k - d//2)/bubblePeriod)
    else:
        x, _ = dataset.next_batch()

    x = x.reshape(batch_size, d, h, w, 1)
    output = model_class.model.predict(x)
    inputs.append(x)
    outputs.append(output)

    mu_before = np.mean(np.concatenate(inputs))
    mu_after = np.mean(np.concatenate(outputs))
    stddev_before = np.std(np.concatenate(inputs), ddof=1)
    stddev_after = np.std(np.concatenate(outputs), ddof=1)

    print("Before mean: {:04.2f}, stddev: {:04.2f}".format(mu_before, stddev_before))
    print("After mean: {:04.2f}, stddev: {:04.2f}".format(mu_after, stddev_after))

    x = inputs[0]
    z = outputs[0]

    x = x[:,5:-5,:,:,:]

    vmin_input, vmax_input = np.min(x), np.max(x)
    vmin_output, vmax_output = np.min(z), np.max(z)

    for i in range(batch_size):

        if i >= stop_after_n_examples:
            break

        for j in range(15):

            plt.subplot(6,5,j+1)
            plt.title('original '+str(j))
            plt.imshow(x[i,j,:,:,0], vmin=vmin_input, vmax=vmax_input, cmap='gray')

            plt.subplot(6,5,3*5+j+1)
            plt.title('resampled '+str(j))
            plt.imshow(z[i,j,:,:,0], vmin=vmin_output, vmax=vmax_output, cmap='gray')

        plt.show()

def test_noise_layer(stop_after_n_examples):
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

        if i >= stop_after_n_examples:
            break

        for j in range(14):

            plt.subplot(4,7,j+1)
            plt.title('original '+str(j+5))
            plt.imshow(x[i,j+5,:,:,0], cmap='gray')


            plt.subplot(4,7,2*7+j+1)
            plt.title('normed '+str(j+5))
            plt.imshow(z[i,j+5,:,:,0], cmap='gray')

        plt.show()

def test_softmax_layer(stop_after_n_examples):

    results_path = '../results'

    model_manager = ModelsManager(results_path)

    train_params = test_softmax.make_model()

    model, model_name, input_shape = train_params
    num_classes = 2

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

    x = np.linspace(-10, 10, 30)
    batch_size = x.shape[0]
    x = x.reshape(1, batch_size)

    y = model_class.model.predict(x)

    for i in range(batch_size):
        print("softmax({:04.2f}) = {:04.2f}".format(x[0,i], y[0,i]) )