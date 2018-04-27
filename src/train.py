from test import test_model
from tqdm import tqdm
import numpy as np
import keras.backend as K

def train_model(dataset,
                model_class,
                batch_size,
                iterations_per_epoch=4096*32,
                max_epochs=30,
                **kwargs):

    allowed_kwargs = {'avg_grad_n',
                      'avg_grad_stop',
                      }

    for kwarg in kwargs:
        if kwarg not in allowed_kwargs:
            raise TypeError('Keyword argument not understood:', kwarg)

    avg_grad_n = 5
    if 'avg_grad_n' in kwargs:
        avg_grad_n = kwargs['avg_grad_n']

    avg_grad_stop = False
    if 'avg_grad_stop' in kwargs:
        avg_grad_stop = kwargs['avg_grad_stop']

    model = model_class.model
    start_epoch = model_class.next_epoch
    d, h, w = model_class.input_shape

    #model.summary()

    all_test_accs = []

    if start_epoch == 1:

        test_acc = test_model(dataset, model)

        model_class.save_model(model, None, test_acc, 0, change_epoch=False)

    pbar = tqdm(range(start_epoch, max_epochs+1),
                desc='Epoch: {:d}'.format(start_epoch),
                #initial=start_epoch,
                )

    for epoch in pbar:

        # Train the network for this epoch
        train_accs = []
        losses = []

        fetch_time = 0
        train_time = 0
        #K.set_learning_phase(1)

        for step in tqdm(range(int(iterations_per_epoch / batch_size))):

            x, y = dataset.next_batch()

            x = x.reshape((batch_size, d, h, w, 1))

            scores = model.train_on_batch(x, y)

            losses.append(scores[0])
            train_accs.append(scores[1])


        # Summarize epoch results

        train_acc = np.mean(train_accs)

        test_acc = test_model(dataset, model)
        all_test_accs.append(test_acc)

        last_n_test = all_test_accs[-avg_grad_n:]
        N = len(last_n_test)

        if len(last_n_test) == 1:
            acc_change = 0
        else:
            changes = [last_n_test[i] - last_n_test[i-1] for i in range(1,N)]
            acc_change = np.mean(changes)

        desc =  'Epoch {:d} '.format(epoch)
        desc += 'acc ({:04.2f}, {:04.2f}) '.format(train_acc, test_acc)
        desc += '(test change({:d}): {:04.4f})'.format(avg_grad_n, acc_change)

        model_class.save_model(model, train_acc, test_acc, epoch)

        pbar.set_description(desc)

        if avg_grad_stop and N >= avg_grad_n and acc_change < 0:
            break

    model_class.training_summary()



