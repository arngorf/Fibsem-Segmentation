import FibsemDataset

dataset_dir = '../../data/lausanne/'
batch_size = 32
feature_shape = (45, 45, 45)
img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1, 2]]

dataset = FibsemDataset(dataset_dir,
                        batch_size,
                        feature_shape,
                        img_class_map)



def validate_output_manually():
    pass