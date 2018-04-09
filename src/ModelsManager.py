import keras
import os
import pickle
import warnings
from time import time

UNCOMPILED_MODEL_FILENAME = 'base_model.h5'
MODEL_CLASS_FILENAME = 'saved_model.p'

from layers.GeometricTransformationLayer import GeometricTransformationLayer
from layers.FoveationLayer import FoveationLayer

CUSTOM_OBJECTS = {'GeometricTransformationLayer':GeometricTransformationLayer,
                  'FoveationLayer':FoveationLayer,
                  }

def topology_is_equal(model_a, model_b):

    if model_a.layers == None and model_b.layers == None:
        return True

    if len(model_a.layers) != len(model_b.layers):
        return False

    for layer_a, layer_b in zip(model_a.layers, model_b.layers):

        if type(layer_a) != type(layer_b):
            return False
        if layer_a.input_shape != layer_b.input_shape:
            return False
        if layer_a.output_shape != layer_b.output_shape:
            return False

    return True

class ModelsManager():

    __slots__ = ('_models',
                 '_results_path',
                 '_model_params',
                 )

    def __init__(self, results_path=None):
        print('__init__::Begin')
        self._models = {}

        if results_path == None:
            self._results_path = os.path.join(os.getcwd(), '../results')
        else:
            self._results_path = results_path

        self._load_models()
        print('__init__::End')

    def _load_models(self):
        directory = os.fsencode(self._results_path)

        for file in os.listdir(directory):

            model_dir = os.fsdecode(file)
            model_class_path = os.path.join(self._results_path,
                                            model_dir,
                                            MODEL_CLASS_FILENAME,
                                            )

            model_class = pickle.load(open(model_class_path, "rb"))

            name = model_class.name

            def callback():
                self.new_checkpoint_callback(name)

            model_class.set_model_callback(callback)

            model_class.set_model_path(os.path.join(self._results_path,
                                                    model_dir))
            #model_class.load_model('latest', 'base')

            self._models[name] = model_class


    def new_model(self, model, name, input_shape, output_shape, **kwargs):

        if name in self._models:
            existing_model_class = self._models[name]
            existing_model = existing_model_class.model
            if not topology_is_equal(model, existing_model):
                err_msg = "Model with name '" + name + "' already exists, " + \
                          "but the topology differs"
                raise ValueError(err_msg)

            return # model already exists

        model_class = ModelClass(model,
                                 name,
                                 self._results_path,
                                 input_shape,
                                 output_shape,
                                 **kwargs)

        def callback():
            self.new_checkpoint_callback(name)

        model_class.set_model_callback(callback)

        self._models[name] = model_class

        callback()

    def get_model(self, name):
        return self._models[name]

    def new_checkpoint_callback(self, model_id):

        model_class = self._models[model_id]

        model = model_class.pop_current_model()
        callback = model_class.pop_model_callback()

        model_dir = model_class.model_dir

        model_class_path = os.path.join(model_dir,
                                        MODEL_CLASS_FILENAME,
                                        )

        pickle.dump(model_class, open(model_class_path, "wb"))

        model_class.set_current_model(model)
        model_class.set_model_callback(callback)

    @property
    def models(self):
        return self._models.keys()

    def has_model(self, model_name):
        return model_name in self._models



class ModelClass():

    __slots__ = ('_model',
                 '_name',
                 '_model_dir',
                 '_base_model_path',
                 '_sessions',
                 '_session',
                 '_save_model_callback',
                 '_loss',
                 '_opt',
                 '_metrics',
                 '_init_lr',
                 '_lr_decay',
                 '_epoch',
                 '_input_shape',
                 '_output_shape',
                 )

    def __init__(self,
                 model,
                 name,
                 results_path,
                 input_shape,
                 output_shape,
                 **kwargs):

        self._model = model
        self._name = name
        self._model_dir = os.path.join(results_path, name)
        self._base_model_path = os.path.join(self._model_dir, 'base_model.h5')
        self._epoch = 0
        self._input_shape = input_shape
        self._output_shape = output_shape

        self._sessions = {}
        self._session = 'default'
        self._sessions[self._session] = []

        if os.path.isdir(self._model_dir):
            err_msg = "Model already exists:'" + self._model_dir + "'"
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

        os.makedirs(self._model_dir)
        self._model.save(self._base_model_path)

        if 'loss' in kwargs:
            self._loss = kwargs['loss']
        else:
            self._loss = 'categorical_crossentropy'

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
            self._init_lr = 0.001

        if 'decay' in kwargs:
            self._lr_decay = kwargs['decay']
        else:
            self._lr_decay = 1e-6

        if 'opt' in kwargs:
            if not isinstance(kwargs['opt'], str):
                raise TypeError("keyword argument 'opt' must be of type str")
            self._opt = kwargs['opt'].lower()
        else:

            self._opt = 'sgd'#

    def _get_opt(self, **kwargs):

        if self._opt == 'sgd':
            return keras.optimizers.SGD(lr=self._init_lr,
                                        momentum=0.9,
                                        decay=self._lr_decay,
                                        nesterov=True)
        elif self._opt == 'adam':
            return keras.optimizers.Adam(lr=self._init_lr,
                                         decay=self._lr_decay) #, clipnorm=1.0

    def load_model(self, session_name='latest', which='latest'):

        if session_name == 'latest':
            session_name = self._session #self._latest_session
        else:
            session_name = session

        saved_models_list = self._sessions[session_name]

        if which == 'base':
            load_path = self._base_model_path
        elif which == 'latest':
            saved_model = saved_models_list[-1]
            load_path = os.path.join(self._model_dir,
                                     session_name,
                                     saved_model.name)
        elif which == 'best':
            saved_model = max(saved_models_list, key=lambda sm: sm.test_acc)
            load_path = os.path.join(self._model_dir,
                                     session_name,
                                     saved_model.name)

        if which == 'base':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = keras.models.load_model(load_path,
                                                      custom_objects=CUSTOM_OBJECTS)

            opt = self._get_opt()
            self._model.compile(loss=self._loss,
                                optimizer=opt,
                                metrics=self._metrics,
                                )

        else:
            self._model = keras.models.load_model(load_path,
                                                  custom_objects=CUSTOM_OBJECTS)

    def save_model(self, model, train_acc, test_acc, epoch, session_name):

        session_dir = os.path.join(self._model_dir, session_name)

        if not os.path.isdir(session_dir):
            os.makedirs(session_dir)

        model_name = 'epoch_' + str(epoch) + '.h5'

        model_path = os.path.join(session_dir, model_name)

        model.save(model_path)

        saved_model = SavedModel(model_name, train_acc, test_acc, epoch)

        if not session_name in self._sessions:
            self._sessions[session_name] = [saved_model]
        else:
            self._sessions[session_name].append(saved_model)

        self._epoch = epoch

        self._save_model_callback()

    def pop_current_model(self):

        model = self._model
        self._model = None

        return model

    def set_current_model(self, model):
        self._model = model

    def pop_model_callback(self):

        callback = self._save_model_callback
        self._save_model_callback = None

        return callback

    def set_model_callback(self, callback):
        self._save_model_callback = callback

    def set_model_path(self, model_path):
        self._model_dir = model_path
        self._base_model_path = os.path.join(self._model_dir, 'base_model.h5')

    def set_session(self, session_name):
        self._session = session_name
        self._epoch = len(self._sessions[self._session])

    def summary(self):

        print('___Model Summary___')
        print('name:', self._name)
        print('loss:', self._loss)
        print('opt:', type(self._opt).__name__)
        print('metrics:', self._metrics)

        self._model.summary()

        for layer in self._model.layers:
            print(layer.get_output_at(0).get_shape().as_list())

    def session_summary(self, session_name='default'):

        saved_model_list = self._sessions[session_name]

        print('Model:', self._name, 'session:', self.session, 'summary:')
        for saved_model in saved_model_list:
            msg =  'Epoch {:d}, '.format(saved_model.epoch)
            msg += 'train acc: {:04.2f}, '.format(saved_model.train_acc)
            msg += 'test acc: {:04.2f}, '.format(saved_model.test_acc)
            print(msg)

    def session_stats(self, session_name='default'):
        saved_model_list = self._sessions[session_name]

        stats = [(m.epoch,
                  m.train_acc,
                  m.test_acc,
                  ) for m in saved_model_list]
        epoch, train_acc, test_acc = zip(*stats)
        return epoch, train_acc, test_acc

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        if self._model == None:
            self._model = self.load_model('latest', 'base')
        return self._model

    @property
    def next_epoch(self):
        return self._epoch + 1

    @property
    def session(self):
        return self._session

    @property
    def sessions(self):
        return self._sessions.keys()

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

class SavedModel():

    __slots__ = ('_name',
                 '_test_acc',
                 '_train_acc',
                 '_epoch',
                 )

    def __init__(self, name, train_acc, test_acc, epoch):
        self._name = name
        self._test_acc = test_acc
        self._train_acc = train_acc
        self._epoch = epoch

    @property
    def name(self):
        return self._name

    @property
    def test_acc(self):
        return self._test_acc

    @test_acc.setter
    def test_acc(self, value):
        self._test_acc = value

    @property
    def train_acc(self):
        return self._train_acc

    @train_acc.setter
    def train_acc(self, value):
        self._train_acc = value

    @property
    def epoch(self):
        return self._epoch
