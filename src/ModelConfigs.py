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

    def _store_models(self):
        for key in models:
            model = models[key]
            if model.changed:
                model.clear_current_model()
                pickle.dump( model, open( model.model_path, "wb" ) )

    def _load_models(self):
        directory = os.fsencode(saved_models_path)

        for file in os.listdir(directory):

            fname = os.fsdecode(file)

            print(name)
            favorite_color = pickle.load( open( fname, "rb" ) )

class StoredModel():

    __slots__ = ('_model',
                 '_name',
                 '_model_path',
                 '_loss',
                 '_opt',
                 '_metrics',
                 '_init_lr',
                 '_lr_decay',
                 )

    def __init__(self, model, name, saved_models_path, **kwargs):

        self._model = model
        self._name = name
        self._model_path = os.join(saved_models_path, name)

        if not os.path.isdir(model_path):
            os.makedirs(save_dir)
        else:
            err_msg = "Model already exists:'" + self.model_path + "'"
            raise FileExistsError(err_msg)

        allowed_kwargs = {'loss',
                          'opt',
                          'metrics',
                          'init_lr',
                          'lr',
                          'decay',
                          }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        if 'loss' in kwargs:
            self._loss = kwargs['loss']
        else:
            self._loss = 'categorical_crossentropy'

        if 'opt' in kwargs:
            self._opt = kwargs['opt']
        else:
            #opt = keras.optimizers.Adam(lr=initial_learning_rate,
            #                            decay=learning_rate_decay,
            #                            clipnorm=1.0)
            self._opt = keras.optimizers.SGD(lr=initial_learning_rate,
                                       momentum=0.9,
                                       decay=learning_rate_decay,
                                       nesterov=True)

        if 'metrics' in kwargs:
            self._metrics = kwargs['metrics']
        else:
            self._metrics = ['accuracy']

        if 'lr' in kwargs or 'init_lr' in kwargs:
            if 'lr' in kwargs:
                self._init_lr = kwargs['lr']
            elif 'init_lr' in kwargs:
                self._init_lr = kwargs['init_lr']
            else:
                slef._init_lr = 0.001

        if 'decay' in kwargs:
            self._lr_decay = kwargs['decay']
        else:
            self._lr_decay = 1e-6

        self.model.compile(loss=self._loss,
                           optimizer=self._opt,
                           metrics=self._metrics,
                           )

    def load_model(self, run='latest', which='best'):
        pass

    def save_model(self, model, epoch, run='latest'):
        pass

    def clear_current_model(self):
        if self._model != None:
            self._model = None

    def summary(self):

        print('___Model Summary___')
        print('name:', self._name)
        print('loss:', self._loss)
        print('opt:', type(self._opt).__name__)
        print('metrics:', self._metrics)

        self._model.summary()

        for layer in self._model.layers:
            print(layer.get_output_at(0).get_shape().as_list())

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model
