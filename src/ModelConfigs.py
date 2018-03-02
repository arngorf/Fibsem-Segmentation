import pickle
import os

class ModelConfigs():

    __slots__ = ('models',
                 'saved_models_path',
                 )

    def __init__(self, saved_models_path=None):

        self.models = {}

        if saved_models_path == None:
            self.saved_models_path = os.path.join(os.getcwd(), 'saved_models')
        else:
            self.saved_models_path = saved_models_path

        directory = os.fsencode(saved_models_path)

        for file in os.listdir(directory):

            fname = os.fsdecode(file)

            print(name)

    new

class StoredModel():

    __slots__ = ('model',
                 'name',
                 'model_path',
                 'loss',
                 'opt',
                 'metrics',
                 )

    def __init__(self, model, name, saved_models_path, **kwargs):

        self.model = model
        self.name = name
        self.model_path = os.join(saved_models_path, name)

        if not os.path.isdir(model_path):
            os.makedirs(save_dir)
        else:
            err_msg = "Model already exists:'" + self.model_path + "'"
            raise FileExistsError(err_msg)

        allowed_kwargs = {'loss',
                          'opt',
                          'metrics',
                          }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        if 'loss' in kwargs:
            self.loss = kwargs['loss']
        else:
            self.loss = 'categorical_crossentropy'

        if 'opt' in kwargs:
            self.opt = kwargs['opt']
        else:
            #opt = keras.optimizers.Adam(lr=initial_learning_rate,
            #                            decay=learning_rate_decay,
            #                            clipnorm=1.0)
            self.opt = keras.optimizers.SGD(lr=initial_learning_rate,
                                       momentum=0.9,
                                       decay=learning_rate_decay,
                                       nesterov=True)

        if 'metrics' in kwargs:
            self.metrics = kwargs['metrics']
        else:
            self.metrics = ['accuracy']

        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=metrics,
                      )
