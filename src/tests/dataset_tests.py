from PatchDataset import PatchDataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def dataset_real():
    dataset_dir = '../data/lausanne'
    batch_size = 32
    feature_shape = (45, 45, 45)
    #img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1, 2]]
    img_class_map = [[0, 7], [1], [2], [3, 4, 8], [6], [5]]

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
    dataset_mean = 142.1053396892233
    dataset_std = 30.96410819657719

    dataset = PatchDataset(dataset_dir,
                            batch_size,
                            feature_shape,
                            img_class_map,
                            norm_params=(dataset_mean, dataset_std))

    return dataset

def get_image_stack(cidx, depth):
    assert depth % 2 == 1

    s = depth // 2

    left = []
    right = []

    for idx in range(1,s+1):

        left_path = '/home/dith/Dropbox/Fibsem-Segmentation/data/lausanne/image_dir/lausanne_' + str(cidx - idx) + '.png'
        #left_path = '/scratch1/xkv467/lausanne/image_dir/lausanne_' + str(cidx - idx) + '.png'
        left.append(np.array(Image.open(left_path), dtype=np.float32))

        right_path = '/home/dith/Dropbox/Fibsem-Segmentation/data/lausanne/image_dir/lausanne_' + str(cidx + idx) + '.png'
        #right_path = '/scratch1/xkv467/lausanne/image_dir/lausanne_' + str(cidx + idx) + '.png'
        right.append(np.array(Image.open(right_path), dtype=np.float32))

    center_path = '/home/dith/Dropbox/Fibsem-Segmentation/data/lausanne/image_dir/lausanne_' + str(cidx) + '.png'
    #center_path = '/scratch1/xkv467/lausanne/image_dir/lausanne_' + str(cidx) + '.png'

    center_img = np.array(Image.open(center_path), dtype=np.float32)
    result = left
    result.reverse()
    result.append(center_img)
    result.extend(right)

    return np.stack(result)

#img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1, 2]]
img_class_map = [[0, 7], [1], [2], [3, 4, 8], [6], [5]]
class_encoding = ['Cytosol', 'Membrane', 'Synapse', 'Vesicle', 'Endosome',
                  'EnRe', 'Mitochondria','Filament', 'DCV']

class_map_encoding = ['/'.join([class_encoding[label][:4] for label in m]) for m in img_class_map]

def validate_output_manually():
    dataset = dataset_real()

    x, y = dataset.next_batch()

    x = x.reshape(32, 45, 45, 45)
    x[:,45//2,[0,1,2,3,4,5,44,43,42,41,40,39],:] = np.max(x)
    x[:,[0,1,2,3,4,5,44,43,42,41,40,39],45//2,:] = np.max(x)

    for i in range(12):
        for j in range(45):
            plt.subplot(5,9,j+1)
            if j == 4:
                plt.title(class_map_encoding[np.argmax(y[i])])
            plt.imshow(x[i,:,:,j], vmin=0, vmax=255, cmap='gray')

        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        plt.show()

def validate_test_output_manually():
    dataset = dataset_real()

    for x, y, progress in dataset.test_data_stream(batch_size):
        x = x.reshape(32, 45, 45, 45)
        x[:,45//2,:,:] = dataset_mean
        x[:,:,45//2,:] = dataset_mean

        for i in range(12):
            plt.subplot(3,4,i+1)
            plt.title(class_map_encoding[y[i]])
            plt.imshow(x[i,:,:,45//2], vmin=0, vmax=255, cmap='gray')

        plt.show()

def validate_process_unlabeled_image():
    dataset = dataset_real()

    image = get_image_stack(210, 45)

    for x, jj, ii in dataset.process_unlabeled_image(image):

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))

        if jj == 20 and ii == 20:
            for i in range(45):
                if i == 4:
                    plt.title('First Image:')
                plt.subplot(5,9,i+1)
                plt.imshow(x[:,:,i], cmap='gray')

            plt.show()

        if jj == 20 and ii == 500:
            for i in range(45):
                if i == 4:
                    plt.title('Top boundary:')
                plt.subplot(5,9,i+1)
                plt.imshow(x[:,:,i], cmap='gray')

            plt.show()

        if jj == 500 and ii == 20:
            for i in range(45):
                if i == 4:
                    plt.title('Left boundary:')
                plt.subplot(5,9,i+1)
                plt.imshow(x[:,:,i], cmap='gray')

            plt.show()

        if jj == 500 and ii > 490 and ii < 700 and ii % 5 == 0:
            for i in range(45):
                if i == 4:
                    plt.title('Image: ii:' + str(ii) + 'jj:' + str(jj))
                plt.subplot(5,9,i+1)
                plt.imshow(x[:,:,i], cmap='gray')

            plt.show()

def benchmark_dataset(iterations):
    dataset = dataset_real()

    for i in tqdm(range(iterations)):
        dataset.next_batch()

def benchmark_test_dataset(iterations):
    dataset = dataset_real()

    for x, y, progress in tqdm(dataset.test_data_stream(batch_size)): #, iterations
        pass#print(progress)

def validate_train_test():

    dataset = dataset_synth()
    num_batches = 4

    train_examples = {}

    for i in range(num_batches):
        for k in range(32):
            x, y = dataset.next_batch()
            label = np.argmax(y[k])
            if label in train_examples:
                if not np.all(x[k,:] == train_examples[label]):
                    print('Inconsitent labeling in next_batch call')
                    for j in range(4):
                        plt.subplot(3,4,j+1)
                        xx = x[k,:].reshape((25,25,25))[25//2,:,:]
                        plt.imshow(xx, cmap='gray')
                        zz = train_examples[label].reshape((25,25,25))[25//2,:,:]
                        plt.subplot(3,4,j+5)
                        plt.imshow(zz, cmap='gray')
                        diff = zz == xx
                        plt.subplot(3,4,j+9)
                        plt.imshow(diff, cmap='gray')
                    plt.show()
            else:
                train_examples[label] = x[k,:]

    test_examples = {}

    for x, y, p in dataset.test_data_stream(32, num_batches):

        for k in range(32):
            label = y[k]
            if label in test_examples:
                if not np.all(x[k,:] == test_examples[label]):
                    print('Inconsitent labeling in test_data_stream call')
                    for j in range(4):
                        plt.subplot(3,4,j+1)
                        xx = x[k,:].reshape((25,25,25))[25//2-4+4*j,:,:]
                        plt.imshow(xx, cmap='gray')
                        zz = train_examples[label].reshape((25,25,25))[25//2-4+4*j,:,:]
                        plt.subplot(3,4,j+5)
                        plt.imshow(zz, cmap='gray')
                        diff = zz == xx
                        plt.subplot(3,4,j+9)
                        plt.imshow(diff, cmap='gray')
                    plt.show()
            else:
                test_examples[label] = x[k,:]

    print(train_examples.keys(), test_examples.keys())

    for key in test_examples.keys():
        if not key in train_examples:
            print('bad test example, not all keys represented in both training and testing batches')
        if np.any(train_examples[key] != test_examples[key]):
            print('Inconsitent labeling between training and testing patches or patches mismatch')
            print(np.where(train_examples[key] != test_examples[key]))
            for j in range(25):
                plt.subplot(3,25,j+1)
                plt.title('train'+str(j))
                xx = train_examples[key].reshape((25,25,25))[j,:,:]
                plt.imshow(xx, cmap='gray')
                zz = test_examples[key].reshape((25,25,25))[j,:,:]
                plt.subplot(3,25,j+26)
                plt.title('test'+str(j))
                plt.imshow(zz, cmap='gray')
                diff = zz == xx
                plt.subplot(3,25,j+51)
                plt.imshow(diff, cmap='gray')
            plt.show()

def validate_test_consistency():

    examples = {}

    dataset = dataset_real()

    batch_size = 32
    for x, y, cur_batch_size in tqdm(dataset.test_data_stream(batch_size)): #, iterations

        for i in range(cur_batch_size):
            val = y[i]
            key = hash(tuple(x[i,...].flatten()))
            if key in examples:
                stored_val = examples[key]
                if stored_val != val:
                    print('duplicate within dataset')
            else:
                examples[key] = val

    for _ in tqdm(range(10)):
        dataset = dataset_real()

        batch_size = 32
        for x, y, cur_batch_size in tqdm(dataset.test_data_stream(batch_size)): #, iterations

            for i in range(cur_batch_size):
                val = y[i]
                key = hash(tuple(x[i,...].flatten()))
                if key in examples:
                    stored_val = examples[key]
                    if stored_val != val:
                        print('value did not match')
                else:
                    print('new value')
                    examples[key] = val


def validate_test_consistency_2():


    for _ in range(8):
        dataset = dataset_real()

        batch_size = 32
        total = 0
        for x, y, cur_batch_size in tqdm(dataset.test_data_stream(batch_size)):
            total += cur_batch_size

        print('total', total)

