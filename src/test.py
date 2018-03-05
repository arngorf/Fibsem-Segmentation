import numpy as np

def test_model(dataset, model):

    batch_size = dataset.batch_size

    true = 0
    total = 0

    for x, y_true, progress in dataset.test_data_stream(batch_size):

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1))

        y_predict = np.argmax(model.predict(x), axis=1)

        true += np.sum(y_true==y_predict)
        total += batch_size

    return true / total
