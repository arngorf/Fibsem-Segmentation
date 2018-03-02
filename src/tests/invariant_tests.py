import keras
from keras.models import Sequential
from layers.InvariantDifferentialOperatorLayer import InvariantDifferentialOperatorLayer
import matplotlib.pyplot as plt

batch_size = 32

number_of_examples_per_epoch = 482000
number_of_epochs = (30 * 3000000) / number_of_examples_per_epoch

initial_learning_rate = 0.001
learning_rate_decay = 1e-6

image_path = 'images/lausanne'
segmentation_path = 'train_data/segmented'
test_path = None
number_of_slices = [90,91,92,93,94,95,96,97,98,99,440,450,460,470,480,490,500]
image_bounds = (0, 1065)
data_augmentation = True
output_feature_shape = (45, 45, 45)
img_class_map = [[0, 7], [1], [2], [3, 4, 8], [6], [5]]
num_classes = len(img_class_map)

model = Sequential()

model.add(InvariantDifferentialOperatorLayer([2.5, 5, 10], [1, 2], input_shape=(45,45,45,1)))
#model.add(InvariantDifferentialOperatorLayer([1.7, 1.7, 1.7], [1, 2], input_shape=(45,45,45,1)))

opt = keras.optimizers.Adam(lr=initial_learning_rate,
                            decay=learning_rate_decay,
                            clipnorm=0.5)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())
    print(layer.weights)

# --------------------------------------------------------------------------- #
#   LOAD DATASET  ----------------------------------------------------------- #
# --------------------------------------------------------------------------- #

from PatchDataset import PatchDataset
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

x_batch, y_batch = dataset.next_batch()

x_batch = x_batch.reshape((batch_size, 45, 45, 45, 1))

res = model.predict(x_batch)

print(res.shape)

for i in range(32):
    print(res.shape)
    r0 = res[i,:,8,:,0]
    r1 = res[i,:,8,:,1]
    r2 = res[i,:,8,:,2]
    plt.subplot(1,2,1)
    plt.imshow(r1,cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(r2,cmap='gray')
    plt.show()

print(res.shape)