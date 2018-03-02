import numpy as np

def test_model(dataset, model, norm_params=None):

    batch_size = dataset.batch_size

    true = 0
    total = 0

    if norm_params:
        mean, std = norm_params
        dataset_scaling = 1./std

    for x, y_true, progress in dataset.test_data_stream(batch_size):
        x = x.reshape((x.shape[0],x.shape[1],x.shape[2],x.shape[3], 1))
        print(x.dtype)
        #if norm_params:
        x = x.astype(np.float32)
        x = np.subtract(x, mean)
        x = x * dataset_scaling

        print(x.dtype)

        y_predict = np.argmax(model.predict(x), axis=1)
        print(y_true)
        print(y_predict)
        true += np.sum(y_true==y_predict)
        total += batch_size

    return true / total
