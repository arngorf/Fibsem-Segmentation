from functools import reduce
from operator import add, mul
from PIL import Image
from random import shuffle, randint, uniform
from tqdm import tqdm
import math
import numpy as np
import os
import re
import sys
import threading
import util.Online_Parameter_Calculation as OPC

NO_SEGMENTATION = 255
NO_LABEL = 255

def load_image(path, bounds):
    img = np.array(Image.open(path))
    return img[bounds[0]:bounds[1],
               bounds[2]:bounds[3]]

def get_padded_image(img, img_k, img_j, img_i, feature_shape):

    k_d = feature_shape[0] // 2
    k_h = feature_shape[1] // 2
    k_w = feature_shape[2] // 2

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

    __slots__ = ('dataset_handle',
                 'batch_size',
                 'data_type',
                 'num_wanted_avail'
                 )

    def __init__(self, dataset_handle, batch_size):

        threading.Thread.__init__(self)
        self.setDaemon(True)

        self.dataset_handle = dataset_handle
        self.batch_size = batch_size
        self.data_type = 'train'
        self.num_wanted_avail = batch_size*dataset_handle.number_of_available_batches

    def run(self):

        while True:

            self.dataset_handle.is_non_full.wait()

            with self.dataset_handle.num_running_threads_lock:

                num_spawns = self.num_wanted_avail - self.dataset_handle.num_running_threads
                self.dataset_handle.is_non_full.clear()
                self.dataset_handle.num_running_threads = self.num_wanted_avail

            for i in range(num_spawns):
                self.dataset_handle._spawn_new_data_fetch_thread(self.data_type)

            while len(self.dataset_handle.joinable_threads) > 0:
                thread = self.dataset_handle.joinable_threads.pop()
                thread.join()

class SingleImageFetcherThread(threading.Thread):
    '''Thread class for asynchronously fetching data from the disk
    '''

    __slots__ = ('handle',
                 'image_path',
                 'dict_key',
                 'image_entry_index',
                 'bounds',
                 )

    def __init__(self, handle, image_path, dict_key, image_entry_index, bounds):

        threading.Thread.__init__(self)

        self.handle = handle
        self.image_path = image_path
        self.dict_key = dict_key
        self.image_entry_index = image_entry_index
        self.bounds = bounds

    def run(self):
        '''Called on thread.start(), calling the passed data fetching function
        '''

        img_path = self.image_path

        img = np.array(Image.open(img_path))

        img = img[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3]]

        height, width = img.shape
        img = img.reshape((1,height,width))

        with self.handle.image_list_lock:
            self.handle.image_query_result_dict[self.dict_key][0][self.image_entry_index] = img
            self.handle.image_query_result_dict[self.dict_key][1] -= 1
            self.handle.joinable_threads.append(self)

class SingleFeatureClassFetcherThread(threading.Thread):
    '''Thread class for asynchronously fetching data from the data class classes
    '''

    __slots__ = ('dataset_handle',
                 'class_handle',
                 )

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

class FeatureClassImageQuery(object):
    '''Class controls the reading of features for one feature class
    The class attempts to keep multiple images available using a dictionary of
    the images
    '''

    __slots__ = ('segmentations',
                 'one_hot_encoding',
                 'class_map',
                 'feature_shape',
                 'number_of_live_images',
                 'new_image_probability',
                 'image_list_lock',
                 'get_data_lock',
                 'image_point_indices_pair_list',
                 'image_query_result_dict',
                 'cur_dict_key',
                 'number_images_begin_fetched',
                 'next_seg_idx',
                 'joinable_threads',
                 )

    def __init__(self,
                 segmentations,
                 one_hot_encoding,
                 class_map,
                 feature_shape,
                 number_of_live_images,
                 new_image_probability):

        ##  Set static class variables  #######################################

        self.segmentations = segmentations
        self.one_hot_encoding = one_hot_encoding
        self.class_map = class_map
        self.feature_shape = feature_shape
        self.number_of_live_images = number_of_live_images
        self.new_image_probability = new_image_probability

        ##  Variables for asynchronous data fetching  #########################
        # Warning: Use the lock for these!

        self.image_list_lock = threading.Lock()
        self.get_data_lock = threading.Lock()
        self.image_point_indices_pair_list = []
        self.image_query_result_dict = dict()
        self.cur_dict_key = 0
        self.number_images_begin_fetched = 0

        self.next_seg_idx = randint(0, len(self.segmentations) - 1)
        self.joinable_threads = []

        ##  Grab first image  #################################################
        self._grab_new_image()

    def get_number_of_remaining_available_batches(self):
        with self.image_list_lock:
            avail_batches = [len(pair[1]) for pair in self.image_point_indices_pair_list]
            number_of_remaining_avilable_batches = reduce(add,
                                                          avail_batches,
                                                          0)
        return number_of_remaining_avilable_batches

    def _get_next_segmentation(self):
        self.next_seg_idx = (self.next_seg_idx + 1) % len(self.segmentations)
        return self.segmentations[self.next_seg_idx]

    def _grab_new_image(self):
        '''Spawn new thread for grabbing the next image
        '''
        sparseSegmentations = self._get_next_segmentation()

        image_paths = sparseSegmentations.image_paths
        img_bounds = sparseSegmentations.bounds

        images_to_load = len(image_paths)

        cur_dict_key = self.cur_dict_key
        self.cur_dict_key += 1

        new_thread_result_entry = [[None for i in range(images_to_load)],
                                   images_to_load,
                                   sparseSegmentations.segmentation_path,
                                   sparseSegmentations.bounds]

        self.image_query_result_dict[cur_dict_key] = new_thread_result_entry

        self.number_images_begin_fetched += 1

        for image_entry_index, image_path in enumerate(image_paths):

            thread = SingleImageFetcherThread(self,
                                              image_path,
                                              cur_dict_key,
                                              image_entry_index,
                                              img_bounds).start()

    def _get_completed_queries(self):

        keys_to_delete = []
        query_results = []

        with self.image_list_lock:

            # For every image query group
            for key in self.image_query_result_dict:

                # Check if it is done
                if self.image_query_result_dict[key][1] == 0:

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
            segmented_path = result[2]
            img_bounds = result[3]

            segmented_img = np.array(Image.open(segmented_path))

            segmented_img = segmented_img[img_bounds[0]:img_bounds[1],
                                          img_bounds[2]:img_bounds[3]]

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

        x = np.empty(self.feature_shape, dtype = np.float32)
        y = np.copy(self.one_hot_encoding)

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
                                                self.feature_shape[0] // 2,
                                                x_j,
                                                x_i,
                                                self.feature_shape)

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


class PatchDataset(object):

    __slots__ = ('_dataset_dir',
                 '_n_classes',
                 '_batch_size',
                 '_feature_shape',
                 '_img_class_map',
                 '_image_params',
                 '_train_params',
                 '_test_params',
                 '_post_processing_params',
                 '_test_segmentations',
                 '_new_image_probability',
                 'number_of_available_batches',
                 '_mean',
                 '_std',
                 '_feature_classes',
                 'fetch_results_cond',
                 'fetch_results',
                 '_next_class_idx',
                 'num_running_threads_lock',
                 'num_running_threads',
                 'is_non_full',
                 'joinable_threads',
                 '_extra_batches',
                 '_threadStarterThread',
                 )

    def __init__(self,
                 dataset_dir,
                 batch_size,
                 feature_shape,
                 img_class_map,
                 **kwargs):

        '''Patch based dataset class

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
            image_dir: Overwrites the standard image path wrt. dataset_dir
            test_dir: Overwrites the standard test path wrt. dataset_dir
            train_dir: Overwrites the standard train path wrt. dataset_dir
            post_processing_path: Overwrites the standard pp_train path wrt.
                dataset_dir
            new_image_probability: Probability that the training image is
                left before completion. Default: 0.0
            number_of_available_batches: How many batches should be kept
                pre-fetched at all time.
        '''

        self._dataset_dir = dataset_dir

        if self._dataset_dir[-1] != os.sep:
            self._dataset_dir += os.sep

        self._n_classes = len(img_class_map)

        self._batch_size = batch_size

        self._feature_shape = feature_shape

        if not isinstance(self._feature_shape, tuple):
            raise TypeError('feature shape must be of type tuple')

        if len(self._feature_shape) != 3:
            raise ValueError('feature shape must be of length 3')

        for shape_dim in self._feature_shape:
            if shape_dim % 2 != 1:
                raise ValueError('feature shape entries must be odd')

        self._img_class_map = img_class_map

        allowed_kwargs = {'image_dir',
                          'test_dir',
                          'train_dir',
                          'post_processing_dir',
                          'new_image_probability',
                          'number_of_available_batches',
                          'norm_params',
                          }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        dir_names = ['image',
                     'train',
                     'test',
                     'post_processing',
                     ]

        for data_name in dir_names:
            data_dir_path = data_name + '_dir'
            if data_dir_path in kwargs:
                data_dir = kwargs[data_dir_path]
            else:
                data_dir = dataset_dir + os.sep + data_name

            sub_dir_path = self._dataset_dir + data_dir_path + os.sep
            paths, indices = self._get_data_parameters(sub_dir_path)

            if data_dir_path == 'image_dir':
                self._image_params = (data_dir, paths, indices)
            elif data_dir_path == 'train_dir':
                self._train_params = (data_dir, paths, indices)
            elif data_dir_path == 'test_dir':
                self._test_params = (data_dir, paths, indices)
            elif data_dir_path == 'post_processing_dir':
                self._post_processing_params = (data_dir, paths, indices)

        if 'new_image_probability' in kwargs:

            self._new_image_probability = kwargs['new_image_probability']

            if self._new_image_probability < 0 or self._new_image_probability > 1:
                raise ValueError('new_image_probability must be in the range [0.0,1.0]')

        else:

            self._new_image_probability = 0.0

        if 'number_of_available_batches' in kwargs:

            self.number_of_available_batches = kwargs['number_of_available_batches']

            if not isinstance(self.number_of_available_batches, int):
                raise TypeError('number_of_available_batches must be of type integer')

        else:

            self.number_of_available_batches = 4

        if 'norm_params' in kwargs:

            self._mean, self._std = kwargs['norm_params']

        else:

            self._mean, self._std = self._calculate_normalization_params()

        self._test_segmentations = self._get_image_indices_with_class_list(self._test_params)

        self._feature_classes = []

        segmentations, bounds = self._get_image_indices_with_class_list(self._train_params)
        number_of_live_images = 1

        for class_idx in range(self._n_classes):

            # Sort out feature mappings
            class_map = None

            if img_class_map == None:
                class_map = [class_idx]
            if img_class_map != None:
                class_map = img_class_map[class_idx]

            one_hot_encoding = np.zeros((1, self._n_classes), dtype = np.uint8)
            one_hot_encoding[0, class_idx] = 1

            newFeatureClass = FeatureClassImageQuery(segmentations[class_idx],
                                                     one_hot_encoding,
                                                     class_map,
                                                     feature_shape,
                                                     number_of_live_images,
                                                     self._new_image_probability)

            self._feature_classes.append(newFeatureClass)

        # Multi-threading data fetching variables

        self.fetch_results_cond = threading.Condition()
        self.fetch_results = []
        self._next_class_idx = 0

        # Mechanics for getting data in multiple threads

        self.num_running_threads_lock = threading.Lock()
        self.num_running_threads = 0
        self.is_non_full = threading.Event()
        self.is_non_full.set()
        self.joinable_threads = []
        self._extra_batches = []

        self._threadStarterThread = ThreadStarterThread(self, self._batch_size)
        self._threadStarterThread.start()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def feature_class(self):
        return self._feature_shape

    def get_number_of_remaining_available_batches(self):
        return [feature_class.get_number_of_remaining_available_batches() for feature_class in self.feature_classes]

    def _calculate_normalization_params(self):

        opc = OPC.Online_Parameter_Calculation()

        image_paths = self._image_params[1]

        for image_path in tqdm(image_paths,
                               ascii = True,
                               desc = 'Calculating normalization parameters'):

            image = np.array(Image.open(image_path))
            opc.add_next_set(image, lambda x: x != 0)

        print('Calculated mean and std of dataset to:',
              '\nMean:',opc.mean(),
              '\nStd:' ,opc.std())

        return opc.mean(), opc.std()

    def _get_data_parameters(self, path):

        directory = os.fsencode(path)

        found_filetypes = {'.png':0,
                           '.tif':0,
                           '.bmp':0,
                           '.jpg':0,
                           }

        filenames = []
        file_indices = []

        for file in os.listdir(directory):

            fname = os.fsdecode(file)
            current_ftype = None

            for ftype in found_filetypes.keys():

                if fname.endswith(ftype):

                    current_ftype = ftype
                    found_filetypes[ftype] += 1
                    break

            if not current_ftype:
                continue

            fname_no_ext = fname[:-4]

            match = re.search(r'\d+$', fname_no_ext)

            if match:
                file_index = int(match.group())

                if file_index in file_indices:
                    raise ValueError('Index ' + \
                                     str(file_index) + \
                                     'found multiple times in: ' + \
                                     directory)

                file_indices.append(file_index)
                filenames.append(path + fname)

        file_indices, filenames = zip(*sorted(zip(file_indices, filenames)))

        return filenames, file_indices


    def _get_image_indices_with_class_list(self, image_params):
        '''Helper function for generating list of image indices where this class
        has been segmented
        '''

        sparseSegmentations = [[] for i in range(self._n_classes)]

        i_max = j_max = 0
        i_min = j_min = 2**64

        for img_idx, path in tqdm(zip(image_params[2], image_params[1]),
                                   ascii = True,
                                   desc = 'Indexing images'):

            if not os.path.isfile(path):
                raise FileNotFoundError('Could not find segmentation file: ' + path)

            img = np.array(Image.open(path))
            img_h, img_w = img.shape

            j, i = np.where(img != NO_SEGMENTATION)

            bounds = (max(np.min(j) - self._feature_shape[1] // 2, 0),
                      min(np.max(j) + self._feature_shape[1] // 2 + 1, img_h),
                      max(np.min(i) - self._feature_shape[2] // 2, 0),
                      min(np.max(i) + self._feature_shape[2] // 2 + 1, img_w))

            segmentation = SparseImageSegmentation(img_idx,
                                                   path,
                                                   self._image_params[2],
                                                   self._image_params[1],
                                                   self._feature_shape,
                                                   bounds)

            for class_idx in range(self._n_classes):

                for class_map_value in self._img_class_map[class_idx]:

                    if class_map_value in img:

                        sparseSegmentations[class_idx].append(segmentation)
                        break

        for i in range(len(sparseSegmentations)):
            shuffle(sparseSegmentations[i])

        return sparseSegmentations, bounds

    def _spawn_new_data_fetch_thread(self, data_type):

        # To do: make data_type have the intended effect

        # Advance the class index such that we get even spread of class types
        self._next_class_idx = (self._next_class_idx + 1) % self._n_classes

        class_handle = self._feature_classes[self._next_class_idx]

        new_thread = SingleFeatureClassFetcherThread(self, class_handle)

        new_thread.start()

    def next_batch(self):

        queried_data = []
        counts = [0 for i in range(self._n_classes)]

        for i in range(len(self._extra_batches)-1,-1,-1):

            y = self._extra_batches[i][1]
            label = np.argmin(y)

            if counts[label] < np.ceil(self._batch_size / self._n_classes):
                queried_data.append(self._extra_batches[i])
                del self._extra_batches[i]
                counts[label] += 1

        while len(queried_data) < self._batch_size:

            self.fetch_results_cond.acquire()

            if len(self.fetch_results) == 0:
                self.fetch_results_cond.wait()

            result = self.fetch_results.pop()
            self.fetch_results_cond.release()

            with self.num_running_threads_lock:
                self.num_running_threads -= 1
                self.is_non_full.set()

            y = result[1]
            label = np.argmin(y)
            if counts[label] < np.ceil(self._batch_size / self._n_classes):
                queried_data.append(result)
                counts[label] += 1
            else:
                self._extra_batches.append(result)

        shuffle(queried_data)

        x_list, y_list = zip(*queried_data)

        x = np.vstack(x_list)
        y = np.vstack(y_list)

        return x, y

    def test_data_stream(self, batch_size, max_batches=None):

        segmentations, bounds = self._test_segmentations
        reverse_class_map = {}
        reverse_mapper = np.vectorize(lambda label: reverse_class_map[label])

        for used_label, image_labels in enumerate(self._img_class_map):
            for image_label in image_labels:
                reverse_class_map[image_label] = used_label

        for sparseSegmentations in segmentations:

            sparseSegmentation = sparseSegmentations[0]
            segmentation_path = sparseSegmentation.segmentation_path
            image_paths = sparseSegmentation.image_paths
            bounds = sparseSegmentation.bounds

            images_to_load = len(image_paths)

            image_block = []
            for path in image_paths:
                image_block.append(load_image(path, bounds))

            image = np.stack(image_block, axis=0)

            pad_k = self._feature_shape[0] // 2
            pad_j = self._feature_shape[1] // 2
            pad_i = self._feature_shape[2] // 2

            image = np.pad(image,
                           [(pad_k, pad_k),
                            (pad_j, pad_j),
                            (pad_i, pad_i)
                           ],
                           mode='reflect',
                           )

            image = image.astype(np.float32)

            segmented_img = np.array(Image.open(segmentation_path))

            segmented_img = segmented_img[bounds[0]:bounds[1],
                                          bounds[2]:bounds[3]]

            segmented_img = np.pad(segmented_img,
                                   [(pad_j, pad_j),
                                    (pad_i, pad_i),
                                   ],
                                   mode='constant',
                                   constant_values=NO_LABEL,
                                   )

            j, i = np.where(segmented_img != NO_LABEL)

            cur_idx = 0

            total_batches = i.shape[0]//32
            kkk = self._feature_shape[0]//2

            if max_batches != None:
                total_batches = min(total_batches, max_batches)

            for batch_idx in range(total_batches):

                progress = (batch_idx+1) / (total_batches)

                ii = i[cur_idx:cur_idx + batch_size]
                jj = j[cur_idx:cur_idx + batch_size]

                x = []

                for idx in range(batch_size):

                    jjj = jj[idx]
                    iii = ii[idx]

                    x.append(image[kkk-pad_k:kkk+pad_k+1,
                                   jjj-pad_j:jjj+pad_j+1,
                                   iii-pad_i:iii+pad_i+1],
                                   )

                x = np.stack(x, axis=0)
                y = reverse_mapper(segmented_img[jj, ii])
                cur_idx += batch_size

                yield x, y, progress

    def process_unlabeled_image(self, img):

        if len(img.shape) != 3:
            return

        d, h, w = img.shape

        pad_k = self._feature_shape[0] // 2
        pad_j = self._feature_shape[1] // 2
        pad_i = self._feature_shape[2] // 2

        img = np.pad(img,
                     [ (pad_k, pad_k),
                       (pad_j, pad_j),
                       (pad_i, pad_i) ],
                     mode='reflect')

        #for k in range(d):
        for k in [pad_k+img.shape[0]//2]:
            for j in range(pad_j, h+pad_j):
                for i in range(pad_i, w+pad_i):

                    x = img[k-pad_k:k+pad_k+1,
                            j-pad_j:j+pad_j+1,
                            i-pad_i:i+pad_i+1]

                    yield x, k-pad_k, j-pad_j, i-pad_i

class SparseImageSegmentation(object):

    __slots__ = ('_segmentation_index',
                 '_segmentation_path',
                 '_image_indices',
                 '_image_paths',
                 '_bounds',
                 )

    def __init__(self,
                 segmentation_index,
                 segmentation_path,
                 image_indices,
                 image_paths,
                 feature_shape,
                 bounds):

        kw = feature_shape[0] // 2

        self._segmentation_index = segmentation_index
        self._segmentation_path = segmentation_path

        self._image_indices = []
        self._image_paths = []

        local_seg_idx = segmentation_index - min(image_indices)
        local_max_idx = len(image_indices)

        for img_idx in range(local_seg_idx - kw, local_seg_idx + kw + 1):

            if img_idx < 0 or img_idx >= local_max_idx:
                if img_idx < 0:
                    img_idx = - img_idx
                if img_idx >= local_max_idx:
                    img_idx = 2*local_max_idx - img_idx - 1

            self._image_indices.append(img_idx + min(image_indices))
            self._image_paths.append(image_paths[img_idx])

        self._bounds = bounds

    @property
    def image_indices(self):
        return self._image_indices

    @property
    def segmentation_index(self):
        return self._segmentation_index

    @property
    def image_paths(self):
        return self._image_paths

    @property
    def segmentation_path(self):
        return self._segmentation_path

    @property
    def bounds(self):
        return self._bounds