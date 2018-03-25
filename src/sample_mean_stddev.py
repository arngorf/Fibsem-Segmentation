from PatchDataset import PatchDataset
import numpy as np
from tqdm import tqdm

def sample_mean_stddev(num_batches):
    dataset_path = '../data/lausanne'

    batch_size = 32
    img_class_map = [[0, 3, 4, 5, 6, 7, 8], [1,2]]
    num_classes = len(img_class_map)
    norm_params = (142.1053396892233, 30.96410819657719)

    input_shape = (25, 25, 25)

    dataset = PatchDataset(dataset_path,
                           batch_size,
                           input_shape,
                           img_class_map,
                           norm_params=norm_params,
                           )

    s  = 0
    ss = 0

    n = num_batches
    m = input_shape[0]*input_shape[1]*input_shape[2]
    N = n*m*batch_size

    for i in tqdm(range(num_batches)):
        x, _ = dataset.next_batch()
        s += np.sum(x) / N
        ss += np.sum(x**2) / N

    print('new norm_params =', (s, np.sqrt(ss - (s)**2)))

if __name__ == '__main__':

    #sample_mean_stddev(1000)
    sample_mean_stddev(1000000)
    '''n = 500
    m = 5000
    N = n*m

    x = np.random.random((n,m))

    print(np.mean(x), np.std(x))

    s  = 0
    ss = 0

    for i in range(n):
        s += np.sum(x[i,:])
        ss += np.sum(x[i,:]**2)

    #print(np.sqrt(np.mean(x[:,0]**2)-np.mean(x[:,0])**2))
    print(s/N, np.sqrt(ss/N - (s/N)**2))'''

'''
100000 (a)
(126.12972547016045, 29.09073523250412)
100000 (b)
(126.10186066424009, 29.099877576809433)
100000 (c)
(126.08273137527857, 29.084899638376303)
1000000 (a)
'''