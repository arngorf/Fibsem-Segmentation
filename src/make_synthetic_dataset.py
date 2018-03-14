import numpy as np
import os
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from PIL import Image
import util.Online_Parameter_Calculation as OPC

input_shape = (25,25,25)
num_shape = (50,50,50)
total_shapes = num_shape[0] * num_shape[1] * num_shape[2]
img_dims = (input_shape[0]*50, input_shape[1]*50, input_shape[2]*50)
path = '../data/test_dataset/'
if not os.path.isdir(path):
    os.makedirs(path)
directories = ['image_dir', 'train_dir', 'test_dir', 'post_processing_dir']
dir_paths = []
for directory in directories:
    dir_path = os.path.join(path, directory)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    dir_paths.append(dir_path)

target_mu = 142.1053396892233
target_std = 30.96410819657719

circle_radius = 7
box_radius = 7
cross_radius = 9
cross_thickness = 1

cz = input_shape[0]//2
cy = input_shape[1]//2
cx = input_shape[2]//2

circle_image = np.full(input_shape, 0, dtype=np.float64)
cross_image  = np.full(input_shape, 0, dtype=np.float64)
box_image    = np.full(input_shape, 0, dtype=np.float64)

for k in range(input_shape[0]):
    for j in range(input_shape[1]):
        for i in range(input_shape[0]):
            ii = i-cx
            jj = j-cy
            kk = k-cz
            if ii**2 + jj**2 + kk**2 < circle_radius**2:
                circle_image[k,j,i] = 255

            if abs(ii) < box_radius and abs(jj) < box_radius and abs(kk) < box_radius:
                box_image[k,j,i] = 255

            if ii**2 + jj**2 + kk**2 < cross_radius**2:
                if abs(ii) <= cross_thickness or abs(jj) <= cross_thickness or abs(kk) <= cross_thickness:
                    cross_image[k,j,i] = 255

circle_label = np.full(input_shape, 255, dtype=np.uint8)
cross_label  = np.full(input_shape, 255, dtype=np.uint8)
box_label    = np.full(input_shape, 255, dtype=np.uint8)

circle_label[cz, cy, cx] = 0
cross_label[cz, cy, cx] = 1
box_label[cz, cy, cx] = 2

structures = [(circle_image, circle_label),
              (cross_image, cross_label),
              (box_image, box_label),
              ]


img = np.full(img_dims, 0, dtype=np.float32)
labels = np.full(img_dims, 255, dtype=np.uint8)

indices = [i % len(structures) for i in range(total_shapes)]
shuffle(indices)
cur_idxidx = 0

for k in tqdm(range(num_shape[0]), desc='generating images and labels'):
    for j in range(num_shape[1]):
        for i in range(num_shape[2]):

            idx = indices[cur_idxidx]
            cur_idxidx += 1
            struct_img, struct_label = structures[idx]

            k_start = k*input_shape[0]
            k_end = (k+1)*input_shape[0]
            j_start = j*input_shape[1]
            j_end = (j+1)*input_shape[1]
            i_start = i*input_shape[2]
            i_end = (i+1)*input_shape[2]

            img[k_start:k_end, j_start:j_end, i_start:i_end] = struct_img
            labels[k_start:k_end, j_start:j_end, i_start:i_end] = struct_label

'''opc = OPC.Online_Parameter_Calculation()

for k in tqdm(range(img_dims[0]), desc='calculating normalization parameters'):
    opc.add_next_set(img[0,:,:])

mu = opc.mean
std = opc.std'''
mu = np.mean(img)
std = np.std(img, ddof=1)
print(mu, std)
img -= mu
img /= std
img *= target_std
img += target_mu
mu = np.mean(img)
std = np.std(img, ddof=1)
print(mu, std)

for k in tqdm(range(img_dims[0]), desc='Saving images'):
    result = Image.fromarray(img[k,:,:].astype(np.uint8))
    path = os.path.join(dir_paths[0], 'test_dataset_' + str(k) + '.png')
    result.save(path)

for k in tqdm(range(num_shape[0]), desc='saving labels'):
    img_idx = cz+k*input_shape[0]
    dir_path = dir_paths[k % (len(dir_paths) - 1) + 1]
    result = Image.fromarray(labels[img_idx,:,:].astype(np.uint8))
    path = os.path.join(dir_path, 'segmented_' + str(img_idx) + '.png')
    result.save(path)
