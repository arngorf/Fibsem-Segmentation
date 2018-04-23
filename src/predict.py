import numpy as np

def predict_image(dataset, model, image):

    batch_size = dataset.batch_size
    d, h, w = image.shape
    n_classes = dataset.n_classes

    output_image = np.empty((h, w, n_classes), dtype=np.float32)

    batch = []
    i = []
    j = []

    for x, jj, ii in dataset.process_unlabeled_image(image):

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
        batch.append(x)
        i.append(ii)
        j.append(jj)

        if len(batch) == batch_size:

            stacked_batch = np.stack(batch)

            y_predict = model.predict(stacked_batch, batch_size=batch_size)

            output_image[j,i,:] = y_predict
            batch = []
            i = []
            j = []

    if len(batch) == 1:
        batch = batch[0].reshape((1, x.shape[0], x.shape[1], x.shape[2], 1))
        y_predict = model.predict(batch, batch_size=1)
        output_image[j,i,:] = y_predict

    if len(batch) > 1:
        y_predict = model.predict(np.stack(batch), batch_size=len(batch))
        output_image[j,i,:] = y_predict

    return output_image
