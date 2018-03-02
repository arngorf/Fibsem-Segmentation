from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from CNN.FoveationLayer import FoveationLayer
from CNN.RotationLayer import RotationLayer
import keras
import keras.backend as K

def train_model(dataset,
                stored_model,
                batch_size,
                **kwargs,
                ):


    # --------------------------------------------------------------------------- #
    #   SET CONSTANTS  ---------------------------------------------------------- #
    # --------------------------------------------------------------------------- #

    number_of_examples_per_epoch = 482000
    number_of_epochs = (30 * 3000000) // number_of_examples_per_epoch

    test_path = None

    num_classes = len(img_class_map)

    model_name = stored_model.name
    model = stored_model.model

    # --------------------------------------------------------------------------- #
    #   ITERATION VARIABLES  ---------------------------------------------------- #
    # --------------------------------------------------------------------------- #

    steps = []
    losses = []
    accuracies = []

    # --------------------------------------------------------------------------- #
    #   DEFINE MODEL  ----------------------------------------------------------- #
    # --------------------------------------------------------------------------- #


    custom_objects={'RotationLayer':RotationLayer, 'FoveationLayer':FoveationLayer}

    if load_model:
        model_path = os.path.join(save_dir, load_model_name)
        model = keras.models.load_model(model_path, custom_objects=custom_objects)

    # --------------------------------------------------------------------------- #
    #   LOAD DATASET  ----------------------------------------------------------- #
    # --------------------------------------------------------------------------- #

    data = input_data.FibsemDataset(image_path,
                                    image_bounds,
                                    segmentation_path,
                                    number_of_slices,
                                    num_classes,
                                    output_feature_shape,
                                    batch_size,
                                    test_path,
                                    data_augmentation,
                                    img_class_map)

    for epoch in range(start_epoch, number_of_epochs):

        # Train the network for this epoch

        epoch_accuracies = []
        epoch_losses = []

        fetch_time = 0
        train_time = 0

        for step in range(int(number_of_examples_per_epoch / batch_size)):

            t = time.time()
            x_batch, y_batch = data.next_batch(batch_size)
            t = time.time() - t
            fetch_time += t

            x_batch = x_batch.reshape((batch_size, 45, 45, 45, 1))
            y = np.argmax(y_batch, 1)

            t = time.time()
            scores = model.train_on_batch(x_batch, y_batch)
            t = time.time() - t
            train_time += t

            epoch_losses.append(scores[0])
            epoch_accuracies.append(scores[1])

        # Summarize epoch results

        standard_error = np.std(epoch_accuracies)/np.sqrt(len(epoch_accuracies))

        epoch_mean_loss = np.mean(epoch_losses)
        epoch_mean_acc = np.mean(epoch_accuracies)

        print('Epoch', epoch,
              'Ep loss:', epoch_mean_loss,
              'Mean Ep acc:', epoch_mean_acc,
              '+/- std_err:', standard_error,
              'time:', time.strftime("%H:%M:%S"),
              'fetch time:', fetch_time,
              'train time:', train_time)

        steps.append(epoch)
        losses.append(epoch_mean_loss)
        accuracies.append(epoch_mean_acc)

        epoch_model_name = 'reg_fovea_two_layer_model_2_'+str(epoch)+'.h5'
        model_path = os.path.join(save_dir, epoch_model_name)
        model.save(model_path)

    # Save model and weights
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    print('Steps:', steps)
    print('Losses:', losses)
    print('Accuracies:', accuracies)

    del data

