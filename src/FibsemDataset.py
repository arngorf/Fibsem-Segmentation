from functools import reduce
from operator import add, mul
from PIL import Image
from random import shuffle, randint, uniform
from tqdm import tqdm
import DataUtil.Online_Parameter_Calculation as OPC
import CNN.Foveator
import numpy as np
import os

import sys
import threading

filetype = 'png'

def get_padded_image(img, img_k, img_j, img_i, img_feature_shape):

    k_d = img_feature_shape[0] // 2
    k_h = img_feature_shape[1] // 2
    k_w = img_feature_shape[2] // 2

    img_d, img_h, img_w = img.shape

    k_left = max(0, img_k - k_d)
    k_right = min(img_k + k_d + 1, img_d)

    k_left_miss = k_left - (img_k - k_d)
    k_right_miss = (img_k + k_d + 1) - k_right

    j_left = max(0, img_j - k_h)
    j_right = min(img_j + k_h + 1, img_h)

    j_left_miss = j_left - (img_j - k_h)
    j_right_miss = (img_j + k_h + 1) - j_right

    i_left = max(0, img_i - k_w)
    i_right = min(img_i + k_w + 1 ,img_w)

    i_left_miss = i_left - (img_i - k_w)
    i_right_miss = (img_i + k_w + 1) - i_right

    img_slice = img[k_left:k_right, j_left:j_right, i_left:i_right]

    padded_img_slice = np.pad(img_slice,
                              [ (k_left_miss, k_right_miss),
                                (j_left_miss, j_right_miss),
                                (i_left_miss, i_right_miss) ],
                              mode='reflect')

    return padded_img_slice

class ThreadStarterThread(threading.Thread):
    def __init__(self, dataset_handle, batch_size):

        threading.Thread.__init__(self)
        self.setDaemon(True)

        self.dataset_handle = dataset_handle
        self.batch_size = batch_size
        self.data_type = 'train'
        self.num_wanted_avail = batch_size*dataset_handle.number_of_available_batches

    def run(self):

        while True:

            #print("TST0")
            self.dataset_handle.is_non_full.wait()
            #print("TST1")
            with self.dataset_handle.num_running_threads_lock:
                #print("TST2")
                num_spawns = self.num_wanted_avail - self.dataset_handle.num_running_threads
                #print("TST3")
                self.dataset_handle.is_non_full.clear()
                #print("TST4")
                self.dataset_handle.num_running_threads = self.num_wanted_avail
                #print("TST5")

            #print("TST6")

            for i in range(num_spawns):
                self.dataset_handle._spawn_new_data_fetch_thread(self.data_type)

            while len(self.dataset_handle.joinable_threads) > 0:
                thread = self.dataset_handle.joinable_threads.pop()
                thread.join()



class SingleImageFetcherThread(threading.Thread):
    '''Thread class for asynchronously fetching data from the disk
    '''
    def __init__(self, handle, img_idx, dict_key, image_entry_index, bounds):

        threading.Thread.__init__(self)

        self.handle = handle
        self.img_idx = img_idx
        self.dict_key = dict_key
        self.image_entry_index = image_entry_index
        self.bounds = bounds

    def run(self):
        '''Called on thread.start(), calling the passed data fetching function
        '''

        img_path = self.handle.image_path + '_' + str(self.img_idx) + '.' + filetype

        img = np.array(Image.open(img_path))

        img = img[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3]]

        height, width = img.shape
        img = img.reshape((1,height,width))

        with self.handle.image_list_lock:
            self.handle.image_query_result_dict[self.dict_key][0][self.image_entry_index] = img
            self.handle.image_query_result_dict[self.dict_key][2] -= 1
            self.handle.joinable_threads.append(self)

class SingleFeatureClassFetcherThread(threading.Thread):
    '''Thread class for asynchronously fetching data from the data class classes
    '''
    def __init__(self, dataset_handle, class_handle):

        threading.Thread.__init__(self)

        self.dataset_handle = dataset_handle
        self.class_handle = class_handle

    def run(self):
        '''Called on thread.start(), calling the pair acquisition from the
        passed class handle
        '''
        result = self.class_handle.get_next_data_pair()

        self.dataset_handle.fetch_results_cond.acquire()
        thread = self.dataset_handle.fetch_results.append(result)
        self.dataset_handle.fetch_results_cond.notify()
        self.dataset_handle.fetch_results_cond.release()

        self.dataset_handle.joinable_threads.append(self)

class FeatureClassImageQuery():
    '''Class controls the reading of features for one feature class
    The class attempts to keep multiple images available using a dictionary of
    the images
    '''

    def __init__(self,
                 image_path,
                 segmentation_path,
                 image_indices,
                 bounds,
                 train_img_indices,
                 one_hot_idx,
                 n_classes,
                 class_map,
                 img_z_bounds,
                 img_feature_shape,
                 normalization_parameters,
                 number_of_live_images,
                 post_processing,
                 new_image_probability):

        ##  Set static class variables  ########################################

        self.image_path = image_path
        self.segmentation_path = segmentation_path
        self.image_indices = image_indices
        self.bounds = bounds
        self.train_img_indices = train_img_indices
        self.one_hot_idx = one_hot_idx
        self.n_classes = n_classes
        self.class_map = class_map
        self.img_z_bounds = img_z_bounds
        self.img_feature_shape = img_feature_shape
        self.normalization_parameters = normalization_parameters
        self.number_of_live_images = number_of_live_images
        self.post_processing = post_processing
        self.new_image_probability = new_image_probability

        assert img_feature_shape[0] % 2 == 1
        assert img_feature_shape[1] % 2 == 1
        assert img_feature_shape[2] % 2 == 1

        ##  Variables for asynchronous data fetching  ##########################
        # Warning: Use the lock for these!

        self.image_list_lock = threading.Lock()
        self.get_data_lock = threading.Lock()
        self.image_point_indices_pair_list = []
        self.image_query_result_dict = dict()
        self.cur_dict_key = 0
        self.number_images_begin_fetched = 0

        self.next_img_idx = randint(0, len(self.image_indices))
        self.joinable_threads = []

        ##  Grab first image  ##################################################
        tmp_path = image_path + '_' + str(self.train_img_indices[0]) + '.' + filetype
        tmp_img = np.array(Image.open(tmp_path))

        tmp_img = tmp_img[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3]]

        self.img_d = img_z_bounds[1]
        self.img_h = tmp_img.shape[0]
        self.img_w = tmp_img.shape[1]

        self._grab_new_image()

    def get_number_of_remaining_available_batches(self):
        with self.image_list_lock:
            number_of_remaining_avilable_batches = reduce(add, [len(pair[1]) for pair in self.image_point_indices_pair_list], 0)
        return number_of_remaining_avilable_batches

    def _get_next_img_idx(self):
        self.next_img_idx = (self.next_img_idx + 1) % len(self.image_indices)
        return self.image_indices[self.next_img_idx]

    def _grab_new_image(self):
        '''Spawn new thread for grabbing the next image
        '''
        img_k = self._get_next_img_idx()
        k_d = self.img_feature_shape[0]

        new_image_list = [None for i in range(k_d)]

        # find the bounds of the image we want to fetch
        k_back = img_k - k_d // 2
        k_front = img_k + k_d // 2

        threads = []

        cur_dict_key = self.cur_dict_key
        new_thread_result_entry = [[None for i in range(k_d)], img_k, k_d]
        self.image_query_result_dict[cur_dict_key] = new_thread_result_entry
        self.cur_dict_key += 1
        self.number_images_begin_fetched += 1

        image_entry_index = 0

        for img_idx in range(k_back, k_front + 1):
            while img_idx < 0 or img_idx >= self.img_d:
                if img_idx < 0:
                    img_idx = - img_idx
                if img_idx >= self.img_d:
                    img_idx = 2*self.img_d - img_idx - 1

            threads.append(SingleImageFetcherThread(self, img_idx, cur_dict_key, image_entry_index, self.bounds))
            threads[-1].start()
            image_entry_index += 1

    def _get_completed_queries(self):

        keys_to_delete = []
        query_results = []

        with self.image_list_lock:

            # For every image query group
            for key in self.image_query_result_dict:

                # Check if it is done
                if self.image_query_result_dict[key][2] == 0:

                    query_results.append(self.image_query_result_dict[key])
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del self.image_query_result_dict[key]
                self.number_images_begin_fetched -= 1

            for thread in self.joinable_threads:
                thread.join()

            self.joinable_threads = []

        for result in query_results:

            img = np.vstack(result[0])
            img_idx = result[1]

            segmented_path = self.segmentation_path + '_' + str(img_idx) + '.' + filetype

            segmented_img = np.array(Image.open(segmented_path))

            segmented_img = segmented_img[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3]]

            i_list = []
            j_list = []

            for k, label in enumerate(self.class_map):

                j, i = np.where(segmented_img == label)

                if j.shape[0] > 0:
                    i_list.append(i)
                    j_list.append(j)

            i = np.concatenate(i_list)
            j = np.concatenate(j_list)

            indices = list(zip(j, i))

            shuffle(indices)

            with self.image_list_lock:
                self.image_point_indices_pair_list.append((img, indices))

    def get_next_data_pair(self):

        # While running until data is available

        x = np.empty(self.img_feature_shape, dtype = np.float32)
        y = np.zeros((1, self.n_classes), dtype = np.uint8)

        y[0, self.one_hot_idx] = 1

        data_fetch_success = False

        with self.get_data_lock:
            while not data_fetch_success:

                # Get any completed queries
                self._get_completed_queries()

                # Check if image is available
                with self.image_list_lock:
                    if len(self.image_point_indices_pair_list) > 0:

                        # get next randomized index in the image
                        x_j, x_i = self.image_point_indices_pair_list[0][1].pop()

                        # grab the image data from that image
                        img = self.image_point_indices_pair_list[0][0]

                        x[:] = get_padded_image(img,
                                                self.img_feature_shape[0] // 2,
                                                x_j,
                                                x_i,
                                                self.img_feature_shape)

                        data_fetch_success = True

                        # if that image is out of points, delete the stored image
                        # also delete by given probability, but only if more than 1
                        # image is stored

                        if len(self.image_point_indices_pair_list[0][1]) == 0:
                            del self.image_point_indices_pair_list[0]

                        elif self.new_image_probability > 0:

                            random_roll = uniform(0, 1)

                            if random_roll < self.new_image_probability and len(self.image_point_indices_pair_list) > 1:

                                del self.image_point_indices_pair_list[0]
                    # Grab image if not enough is fetched or currently being fetched

                    if len(self.image_point_indices_pair_list) + \
                          self.number_images_begin_fetched < \
                          self.number_of_live_images:
                        self._grab_new_image()

        x = x.reshape((1, x.shape[0] * x.shape[1] * x.shape[2]))

        return (x, y)


class FibsemDataset():
    def __init__(self,
                 dataset_dir,
                 batch_size,
                 feature_shape,
                 img_class_map):

        '''Fibsem Dataset class

        # Arguments
            dataset_dir: Path to the parent directory of the dataset
                Expects existence folders, data, train, test and pp_train
            batch_size: Batch size returned for training upon calling method
                next_batch
            feature_shape: the dimensions wanted for the 3D patch. Must be odd
                symmetric across the center pixel.
            img_class_map: A mapping of the class labels into one-hot encoding.
                [[0],[1,2]] means labels segmentation label 0 is encoded as
                zero, and label 1 and 2 in the segmentation images is encoded
                as 1.
        # Keyword arguments
            image_path: Overwrites the standard image path wrt. dataset_dir
            test_path: Overwrites the standard test path wrt. dataset_dir
            train_path: Overwrites the standard train path wrt. dataset_dir
            post_processing_path: Overwrites the standard pp_train path wrt.
                dataset_dir
            new_image_probability: Probability that the training image is
                left before completion. Default: 0.0
            number_of_available_batches: How many batches should be kept
                pre-fetched at all time.
        '''

        self.dataset_dir = dataset_dir

        self.train_img_indices = self._image_nums_in_dir(self.train_path)
        self.number_of_classes = len(img_class_map)

        self.batch_size = batch_size

        self.feature_shape = feature_shape

        if isinstance(self.feature_shape, tuple):
            raise TypeError('feature shape must be of type tuple')

        if len(self.feature_shape) != 3:
            raise ValueError('feature shape must be of length 3')

        for shape_dim in self.feature_shape:
            if shape_dim % 2 != 1:
                raise ValueError('feature shape entries must be odd')

        self.img_class_map = img_class_map

        allowed_kwargs = {'image_dir',
                          'test_dir',
                          'train_dir',
                          'post_processing_dir',
                          'new_image_probability',
                          'number_of_available_batches',
                          }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        data_names = ['image',
                      'train',
                      'test',
                      'post_processing']

        '''
        The following sets class variables

        self._image_dir
        self._image_paths
        self._image_indices
        self._train_dir
        self._train_paths
        self._train_indices
        self._test_dir
        self._test_paths
        self._test_indices
        self._post_processing_dir
        self._post_processing_paths
        self._post_processing_indices
        '''

        for data_name in data_names:

            data_dir_path = data_name + '_dir'

            if data_dir_name in kwargs:
                self.__dict__['_' + data_dir_path] = kwargs[data_dir_path]
            else:
                self.__dict__['_' + data_dir_path] = dataset_dir + os.sep + data_name

            paths, indices = self._image_nums_in_dir(self, image_dir)

            self.__dict__['_' + data_name + '_paths'] = image_paths
            self.__dict__['_' + data_name + '_indices'] = image_indices

        if 'train_dir' in kwargs:
            self.train_dir = kwargs['train_dir']
        else:
            self.train_dir           = dataset_dir + os.sep + 'train'

        train_paths, train_indices = self._image_nums_in_dir(self, train_dir)

        self.train_paths = train_paths
        self.train_indices = train_indices

        if 'test_dir' in kwargs:
            self.test_dir = kwargs['test_dir']
        else:
            self.test_dir            = dataset_dir + os.sep + 'test'

        train_paths, train_indices = self._image_nums_in_dir(self, train_dir)

        self.train_paths = train_paths
        self.train_indices = train_indices

        if 'post_processing_dir' in kwargs:
            self.post_processing_dir = kwargs['post_processing_dir']
        else:
            self.post_processing_dir = dataset_dir + os.sep + 'pp_train'





        if 'new_image_probability' in kwargs:
            self.new_image_probability = kwargs['new_image_probability']
            if self.new_image_probability < 0 or self.new_image_probability > 1:
                raise ValueError('new_image_probability must be in the range [0.0,1.0]')
        else:
            self.new_image_probability = 0.0

        if 'number_of_available_batches' in kwargs:
            self.number_of_available_batches = kwargs['number_of_available_batches']
        else:
            self.number_of_available_batches = 4




        self.mean = None
        self.std  = None

        self.feature_classes = []

        image_indices, bounds = self._get_image_indices_with_class_list()

        for class_idx in range(number_of_classes):

            # Sort out feature mappings
            class_map = None

            if img_class_map == None:
                class_map = [class_idx]
            if img_class_map != None:
                class_map = img_class_map[class_idx]

            newFeatureClass = FeatureClassImageQuery(self.image_path,
                                                     self.train_path,
                                                     image_indices[class_idx],
                                                     bounds,
                                                     train_img_indices,
                                                     class_idx,
                                                     number_of_classes,
                                                     class_map,
                                                     img_z_bounds,
                                                     feature_shape,
                                                     (self.mean, 1./self.std),
                                                     1,
                                                     self.foveator.process,
                                                     self.new_image_probability)

            self.feature_classes.append(newFeatureClass)

        # Multi-threading data fetching variables

        self.fetch_results_cond = threading.Condition()
        self.fetch_results = []
        self.next_class_idx = 0

        # Mechanics for getting data in multiple threads

        self.num_running_threads_lock = threading.Lock()
        self.num_running_threads = 0
        self.is_non_full = threading.Event()
        self.is_non_full.set()

        self.joinable_threads = []

        self.threadStarterThread = ThreadStarterThread(self, self.batch_size)
        self.threadStarterThread.start()

    def get_number_of_remaining_available_batches(self):
        return [feature_class.get_number_of_remaining_available_batches() for feature_class in self.feature_classes]

    def _get_data_configuration(self, path):

        directory = os.fsencode(directory_in_str)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".asm") or filename.endswith(".py"):
                # print(os.path.join(directory, filename))
                continue
            else:
                continue

        return name, numbers


    def _get_image_indices_with_class_list(self):
        '''Helper function for generating list of image indices where this class
        has been segmented
        '''

        indices = [[] for i in range(self.number_of_classes)]

        i_max = 0
        i_min = None
        j_max = 0
        j_min = None

        opc = OPC.Online_Parameter_Calculation()

        img_shape = None

        img_h = None
        img_w = None

        for img_idx in tqdm(self.train_img_indices,
                            ascii = True,
                            desc = 'Indexing images'):

            path_seg = self.train_path + '_' + str(img_idx) + '.' + filetype
            path = self.image_path + '_' + str(img_idx) + '.' + filetype

            #try:
            img = np.array(Image.open(path))
            img_h, img_w = img.shape
            #except FileNotFoundError:
            #    pass

            #try:
            img_seg = np.array(Image.open(path_seg))
            #except FileNotFoundError:
            #    continue

            opc.add_next_set(img, lambda x: x != 0)

            if img_idx == self.train_img_indices[0]:
                img_shape = img.shape
                i_min = img_shape[1]
                j_min = img_shape[0]

            j, i = np.where(img_seg != 255)

            i_max = max(i_max, np.max(i))
            i_min = min(i_min, np.min(i))
            j_max = max(j_max, np.max(j))
            j_min = min(j_min, np.min(j))

            for class_idx in range(self.number_of_classes):
                for class_map_value in self.img_class_map[class_idx]:
                    if class_map_value in img_seg:
                        indices[class_idx].append(img_idx)
                        break

        for i in range(len(indices)):
            shuffle(indices[i])

        self.mean = opc.get_mean()
        self.std = opc.get_std()

        bounds = (max(j_min - self.feature_shape[1] // 2, 0),
                  min(j_max + self.feature_shape[1] // 2 + 1, img_h),
                  max(i_min - self.feature_shape[2] // 2, 0),
                  min(i_max + self.feature_shape[2] // 2 + 1, img_w))

        return indices, bounds

    def _spawn_new_data_fetch_thread(self, data_type):

        # To do: make data_type have the intended effect

        # Advance the class index such that we get even spread of class types
        self.next_class_idx = (self.next_class_idx + 1) % self.number_of_classes

        class_handle = self.feature_classes[self.next_class_idx]

        new_thread = SingleFeatureClassFetcherThread(self, class_handle)

        new_thread.start()

    def next_batch(self, data_type = 'train'):

        if data_type == 'train':

            queried_data = []

            for _ in range(self.batch_size):

                self.fetch_results_cond.acquire()

                if len(self.fetch_results) == 0:
                    self.fetch_results_cond.wait()

                result = self.fetch_results.pop()
                self.fetch_results_cond.release()

                with self.num_running_threads_lock:
                    self.num_running_threads -= 1
                    self.is_non_full.set()

                queried_data.append(result)

            x_list, y_list = zip(*queried_data)

            x = np.vstack(x_list)
            y = np.vstack(y_list)

            return x, y

        if data_type == 'test':

            pass

    def process_unlabeled_image(self, img):

        if len(img.shape) != 3:
            return

        d, h, w = img.shape

        pad_k = self.feature_shape[0] // 2
        pad_j = self.feature_shape[1] // 2
        pad_i = self.feature_shape[2] // 2

        img = np.pad(img,
                     [ (pad_k, pad_k),
                       (pad_j, pad_j),
                       (pad_i, pad_i) ],
                     mode='reflect')

        #for k in range(d):
        for k in [pad_k+img.shape[0]//2]:
            for j in range(pad_j, h+pad_j):
                for i in range(pad_i, w+pad_i):
                    #x = get_padded_image(img, k, j, i, self.feature_shape)
                    #x = self.foveator.process(x)

                    x = img[k-pad_k:k+pad_k+1, j-pad_j:j+pad_j+1, i-pad_i:i+pad_i+1]

                    yield x, k-pad_k, j-pad_j, i-pad_i