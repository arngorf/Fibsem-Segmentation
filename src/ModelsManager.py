import keras
import os
import pickle
import warnings
from time import time
from models import micro, mini, midi, conv_2_layer

UNCOMPILED_MODEL_FILENAME = 'base_model.h5'
MODEL_CLASS_FILENAME = 'saved_model.p'

from layers.GeometricTransformationLayer import GeometricTransformationLayer
from layers.FoveationLayer import FoveationLayer

CUSTOM_OBJECTS = {'GeometricTransformationLayer':GeometricTransformationLayer,
                  'FoveationLayer':FoveationLayer,
                  }

model_dict = {'micro':micro,
              'mini': mini,
              'midi': midi,
              'conv_2_layer': conv_2_layer}

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

        self._models = {}

        if results_path == None:
            self._results_path = os.path.join(os.getcwd(), '../results')
        else:
            self._results_path = results_path

        self._load_models()

    def _load_models(self):
        directory = os.fsencode(self._results_path)

        for file in os.listdir(directory):

            model_dir = os.fsdecode(file)
            model_class_path = os.path.join(self._results_path,
                                            model_dir,
                                            MODEL_CLASS_FILENAME,
                                            )

            model_class = pickle.load(open(model_class_path, "rb"))

            model_id = model_class.model_id

            def callback():
                self.new_checkpoint_callback(model_id)

            model_class.set_model_callback(callback)

            model_class.set_model_path(os.path.join(self._results_path,
                                                    model_dir))

            self._models[model_id] = model_class

    def new_model(self, model_type, model_id, **model_params):

        if model_id in self._models:
            return

        model_class = ModelClass(model_type,
                                 model_id,
                                 self._results_path,
                                 **model_params)

        def callback():
            self.new_checkpoint_callback(model_id)

        model_class.set_model_callback(callback)

        self._models[model_id] = model_class

        callback()

    def get_model(self, model_id):
        return self._models[model_id]

    def new_checkpoint_callback(self, model_id):

        model_class = self._models[model_id]

        callback = model_class.pop_model_callback()

        model_dir = model_class.model_dir

        model_class_path = os.path.join(model_dir,
                                        MODEL_CLASS_FILENAME,
                                        )

        pickle.dump(model_class, open(model_class_path, "wb"))

        model_class.set_model_callback(callback)

    @property
    def models(self):
        return self._models.keys()

    def has_model(self, model_id):
        return model_id in self._models



class ModelClass():

    __slots__ = ('_model_type',
                 '_model_id',
                 '_model_dir',
                 '_base_model_path',
                 '_saved_models_list',
                 '_save_model_callback',
                 '_loss',
                 '_opt',
                 '_metrics',
                 '_init_lr',
                 '_lr_decay',
                 '_epoch',
                 '_input_shape',
                 '_output_size',
                 '_norm_params',
                 '_model_params',
                 )

    def __init__(self,
                 model_type,
                 model_id,
                 results_path,
                 **kwargs):

        self._model_type = model_type
        self._model_id = model_id
        self._model_dir = os.path.join(results_path, model_id)
        self._base_model_path = os.path.join(self._model_dir, 'base_model.h5')
        self._epoch = 0
        self._input_shape = None

        self._saved_models_list = []

        if os.path.isdir(self._model_dir):
            err_msg = "Model already exists:'" + self._model_dir + "'"
            raise FileExistsError(err_msg)

        allowed_kwargs = {'loss',
                          'opt',
                          'metrics',
                          'init_lr',
                          'lr',
                          'decay',
                          'normalize',
                          'foveation',
                          'linear_deformation',
                          'rotation',
                          'noise',
                          'conv_dropout_p',
                          'dense_dropout_p',
                          'norm_params',
                          'output_size',
                          }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        self._model_params = kwargs
        self._output_size = kwargs['output_size']
        self._norm_params = kwargs['norm_params']

        os.makedirs(self._model_dir)

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

    def load_model(self, which='latest'):

        if len(self._saved_models_list) == 0:
            which = 'base'

        if isinstance(which, int):
            for saved_model in self._saved_models_list:
                if saved_model.epoch == which:
                    load_path = os.path.join(self._model_dir,
                                             saved_model.filename)
                    break
        else:
            if which == 'base':
                load_path = self._base_model_path
            elif which == 'latest':
                saved_model = self._saved_models_list[-1]
                load_path = os.path.join(self._model_dir,
                                         saved_model.filename)
            elif which == 'best':
                saved_model = max(self._saved_models_list, key=lambda sm: sm.test_acc)
                load_path = os.path.join(self._model_dir,
                                         saved_model.filename)

        model_generator = model_dict[self._model_type]

        params = model_generator.make_model(self._output_size,
                                            **self._model_params)

        model, self._input_shape = params

        opt = self._get_opt()

        model.compile(loss=self._loss,
                      optimizer=opt,
                      metrics=self._metrics,
                      )

        if not which == 'base':
            model.load_weights(load_path)

        return model

    def save_model(self, model, train_acc, test_acc, epoch, **kwargs):

        if 'change_epoch' in kwargs:
            change_epoch = kwargs['change_epoch']
        else:
            change_epoch = True

        if not os.path.isdir(self._model_dir):
            os.makedirs(self._model_dir)

        model_filename = 'epoch_' + str(epoch) + '.h5'

        model_path = os.path.join(self._model_dir, model_filename)

        model.save_weights(model_path)

        saved_model = SavedModel(model_filename, train_acc, test_acc, epoch)

        self._saved_models_list.append(saved_model)

        if change_epoch:
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

    def summary(self):

        print('___Model Summary___')
        print('model id:', self._model_id)
        print('loss:', self._loss)
        print('opt:', type(self._opt).__name__)
        print('metrics:', self._metrics)

        self._model.summary()

        for layer in self._model.layers:
            print(layer.get_output_at(0).get_shape().as_list())

    def session_summary(self):

        print('Model:', self._model_id, 'session:', self.session, 'summary:')
        for saved_model in self._saved_models_list:
            msg =  'Epoch {:d}, '.format(saved_model.epoch)
            msg += 'train acc: {:04.2f}, '.format(saved_model.train_acc)
            msg += 'test acc: {:04.2f}, '.format(saved_model.test_acc)
            print(msg)

    def session_stats(self):

        stats = [(m.epoch,
                  m.train_acc,
                  m.test_acc,
                  ) for m in self._saved_models_list]
        print(stats)
        epoch, train_acc, test_acc = zip(*stats)
        return epoch, train_acc, test_acc

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def model_id(self):
        return self._model_id

    @property
    def model(self):
        model = self.load_model()
        return model

    @property
    def next_epoch(self):
        return self._epoch + 1

    @property
    def input_shape(self):
        if self._input_shape == None:
            self.load_model()
        return self._input_shape

    @property
    def output_size(self):
        return self._output_size

class SavedModel():

    __slots__ = ('_filename',
                 '_test_acc',
                 '_train_acc',
                 '_epoch',
                 )

    def __init__(self, filename, train_acc, test_acc, epoch):
        self._filename = filename
        self._test_acc = test_acc
        self._train_acc = train_acc
        self._epoch = epoch

    @property
    def filename(self):
        return self._filename

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
