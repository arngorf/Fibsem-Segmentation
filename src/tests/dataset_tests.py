from PatchDataset import PatchDataset
import numpy as np
from tqdm import tqdm

dataset_dir = '../data/lausanne/'
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

class_encoding = ['Cytosol', 'Membrane', 'Synapse', 'Vesicle', 'Endosome',
                  'EnRe', 'Mitochondria','Filament', 'DCV']

class_map_encoding = ['/'.join([class_encoding[label][:4] for label in m]) for m in img_class_map]

def validate_output_manually():
    x, y = dataset.next_batch()
    print(y)

    x = x.reshape(32, 45, 45, 45)
    x[:,45//2,:,:] = dataset_mean
    x[:,:,45//2,:] = dataset_mean

    import matplotlib.pyplot as plt

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.title(class_map_encoding[np.argmax(y[i])])
        plt.imshow(x[i,:,:,45//2], vmin=0, vmax=255, cmap='gray')

    plt.show()

def validate_test_output_manually():
    for x, y, progress in dataset.test_data_stream(batch_size):
        x = x.reshape(32, 45, 45, 45)
        x[:,45//2,:,:] = dataset_mean
        x[:,:,45//2,:] = dataset_mean

        import matplotlib.pyplot as plt

        for i in range(12):
            plt.subplot(3,4,i+1)
            plt.title(class_map_encoding[y[i]])
            plt.imshow(x[i,:,:,45//2], vmin=0, vmax=255, cmap='gray')

        plt.show()

def benchmark_dataset(iterations):

    for i in tqdm(range(iterations)):
        dataset.next_batch()

def benchmark_test_dataset(iterations):

    for x, y, progress in tqdm(dataset.test_data_stream(batch_size)): #, iterations
        pass#print(progress)