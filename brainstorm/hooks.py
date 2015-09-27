#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.network import Network
import numpy as np
from six import string_types
import h5py

from collections import OrderedDict
from brainstorm.describable import Describable
from brainstorm.utils import get_by_path
from tools import evaluate


class Hook(Describable):
    __undescribed__ = {
        '__name__',  # the name is saved in the trainer
        'run_verbosity'
    }
    __default_values__ = {
        'timescale': 'epoch',
        'interval': 1,
        'verbose': None
    }

    def __init__(self, name=None, timescale='epoch', interval=1, verbose=None):
        self.timescale = timescale
        self.interval = interval
        self.__name__ = name or self.__class__.__name__
        self.priority = 0
        self.verbose = verbose
        self.run_verbosity = None

    def start(self, net, stepper, verbose, named_data_iters):
        if self.verbose is None:
            self.run_verbosity = verbose
        else:
            self.run_verbosity = self.verbose

    def message(self, msg):
        """Print an output message if :attr:`run_verbosity` is True."""
        if self.run_verbosity:
            print("{} >> {}".format(self.__name__, msg))

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        pass


class SaveNetwork(Hook):
    """
    Save the weights of the network to the given file on every call.
    Default is to save them once per epoch, but this can be configured using
    the timescale and interval parameters.
    """

    def __init__(self, filename, name=None, timescale='epoch', interval=1):
        super(SaveNetwork, self).__init__(name, timescale, interval)
        self.filename = filename

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        net.save_as_hdf5(self.filename)

    def load_network(self):
        return Network.from_hdf5(self.filename)


class SaveBestNetwork(Hook):
    """
    Check every epoch to see if the given objective is at it's best value and
    if so, save the network to the specified file.
    """
    __undescribed__ = {'parameters': None}
    __default_values__ = {'filename': None}

    def __init__(self, log_name, filename=None, name=None,
                 criterion='max', verbose=None):
        super(SaveBestNetwork, self).__init__(name, 'epoch', 1, verbose)
        self.log_name = log_name
        self.filename = filename
        self.parameters = None
        assert criterion == 'min' or criterion == 'max'
        self.criterion = criterion

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        e = get_by_path(logs, self.log_name)
        best_idx = np.argmin(e) if self.criterion == 'min' else np.argmax(e)
        if best_idx == len(e) - 1:
            params = net.handler.get_numpy_copy(net.buffer.parameters)
            if self.filename is not None:
                self.message("{} improved. Saving network to {} ...".
                             format(self.log_name, self.filename))
                net.save_as_hdf5(self.filename)
            else:
                self.message("{} improved. Caching parameters ...".
                             format(self.log_name))
                self.parameters = params
        elif self.run_verbosity:
            self.message("Last saved parameters after epoch {} when {} was {}".
                         format(best_idx, self.log_name, e[best_idx]))

    def load_parameters(self):
        return np.load(self.filename) if self.filename is not None \
            else self.parameters


class SaveLogs(Hook):
    def __init__(self, filename, name=None):
        super(SaveLogs, self).__init__(name, 'epoch', 1, False)
        self.filename = filename

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        with h5py.File(self.filename, 'w') as f:
            SaveLogs._save_recursively(f, logs)

    @staticmethod
    def _save_recursively(group, logs):
        for name, log in logs.items():
            if isinstance(log, dict):
                subgroup = group.create_group(name)
                SaveLogs._save_recursively(subgroup, log)
            else:
                group.create_dataset(name, data=np.array(log))


class MonitorLayerParameters(Hook):
    """
    Monitor some properties of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        if name is None:
            name = "MonitorParameters_{}".format(layer_name)
        super(MonitorLayerParameters, self).__init__(name, timescale,
                                                     interval, verbose)
        self.layer_name = layer_name

    def start(self, net, stepper, verbose, named_data_iters):
        assert self.layer_name in net.layers.keys(), \
            "{} >> No layer named {} present in network. Available layers " \
            "are {}.".format(self.__name__, self.layer_name, net.layers.keys())

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        log = OrderedDict()
        for key, v in net.buffer[self.layer_name].parameters.items():
            v = net.handler.get_numpy_copy(v)
            log[key] = OrderedDict()
            log[key]['min'] = v.min()
            log[key]['avg'] = v.mean()
            log[key]['max'] = v.max()
            if len(v.shape) > 1:
                log[key]['min_L2_norm'] = np.sqrt(np.sum(v ** 2, axis=1)).min()
                log[key]['avg_L2_norm'] = np.sqrt(np.sum(v ** 2,
                                                         axis=1)).mean()
                log[key]['max_L2_norm'] = np.sqrt(np.sum(v ** 2, axis=1)).max()

        return log


class MonitorLayerGradients(Hook):
    """
    Monitor some statistics about all the gradients of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        if name is None:
            name = "MonitorGradients_{}".format(layer_name)
        super(MonitorLayerGradients, self).__init__(name, timescale,
                                                    interval, verbose)
        self.layer_name = layer_name

    def start(self, net, stepper, verbose, named_data_iters):
        assert self.layer_name in net.layers.keys(), \
            "{} >> No layer named {} present in network. Available layers " \
            "are {}.".format(self.__name__, self.layer_name, net.layers.keys())

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        log = OrderedDict()
        for key, v in net.buffer[self.layer_name].gradients.items():
            v = net.handler.get_numpy_copy(v)
            log[key] = OrderedDict()
            log[key]['min'] = v.min()
            log[key]['avg'] = v.mean()
            log[key]['max'] = v.max()
        return log


class MonitorLayerDeltas(Hook):
    """
    Monitor some statistics about all the deltas of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        if name is None:
            name = "MonitorDeltas_{}".format(layer_name)
        super(MonitorLayerDeltas, self).__init__(name, timescale,
                                                 interval, verbose)
        self.layer_name = layer_name

    def start(self, net, stepper, verbose, named_data_iters):
        assert self.layer_name in net.layers.keys(), \
            "{} >> No layer named {} present in network. Available layers " \
            "are {}.".format(self.__name__, self.layer_name, net.layers.keys())

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        log = OrderedDict()
        for key, v in net.buffer[self.layer_name].internals.items():
            v = net.handler.get_numpy_copy(v)
            log[key] = OrderedDict()
            log[key]['min'] = v.min()
            log[key]['avg'] = v.mean()
            log[key]['max'] = v.max()

        out_deltas_log = log['output_deltas'] = OrderedDict()
        for key, v in net.buffer[self.layer_name].output_deltas.items():
            v = net.handler.get_numpy_copy(v)
            key_log = out_deltas_log[key] = OrderedDict()
            key_log['min'] = v.min()
            key_log['avg'] = v.mean()
            key_log['max'] = v.max()

        in_deltas_log = log['input_deltas'] = OrderedDict()
        for key, v in net.buffer[self.layer_name].input_deltas.items():
            key_log = in_deltas_log[key] = OrderedDict()
            v = net.handler.get_numpy_copy(v)
            key_log[key]['min'] = v.min()
            key_log[key]['avg'] = v.mean()
            key_log[key]['max'] = v.max()

        return log


class MonitorLayerInOuts(Hook):
    """
    Monitor some statistics about all the inputs and outputs of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        if name is None:
            name = "MonitorInOuts_{}".format(layer_name)
        super(MonitorLayerInOuts, self).__init__(name, timescale,
                                                 interval, verbose)
        self.layer_name = layer_name

    def start(self, net, stepper, verbose, named_data_iters):
        assert self.layer_name in net.layers.keys(), \
            "{} >> No layer named {} present in network. Available layers " \
            "are {}.".format(self.__name__, self.layer_name, net.layers.keys())

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        log = OrderedDict()
        input_log = log['inputs'] = OrderedDict()
        for key, v in net.buffer[self.layer_name].inputs.items():
            v = net.handler.get_numpy_copy(v)
            key_log = input_log[key] = OrderedDict()
            key_log['min'] = v.min()
            key_log['avg'] = v.mean()
            key_log['max'] = v.max()

        output_log = log['outputs'] = OrderedDict()
        for key, v in net.buffer[self.layer_name].outputs.items():
            key_log = output_log[key] = OrderedDict()
            v = net.handler.get_numpy_copy(v)
            key_log['min'] = v.min()
            key_log['avg'] = v.mean()
            key_log['max'] = v.max()

        return log


class StopAfterEpoch(Hook):
    def __init__(self, max_epochs, timescale='epoch', interval=1, name=None,
                 verbose=None):
        super(StopAfterEpoch, self).__init__(name, timescale,
                                             interval, verbose)
        self.max_epochs = max_epochs

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        if epoch_nr >= self.max_epochs:
            self.message("Stopping because the maximum number of epochs ({}) "
                         "was reached.".format(self.max_epochs))
            raise StopIteration()


class EarlyStopper(Hook):
    __default_values__ = {'patience': 1}

    def __init__(self, log_name, patience=1, name=None):
        super(EarlyStopper, self).__init__(name, 'epoch', 1)
        self.log_name = log_name
        self.patience = patience

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        e = get_by_path(logs, self.log_name)
        best_error_idx = np.argmin(e)
        if len(e) > best_error_idx + self.patience:
            self.message("Stopping because {} did not decrease for {} epochs.".
                         format(self.log_name, self.patience))
            raise StopIteration()


class StopOnNan(Hook):
    """ Stop the training if infinite or NaN values are found in parameters.

    Can also check logs for invalid values.
    """
    def __init__(self, logs_to_check=(), check_parameters=True,
                 check_training_loss=True, name=None, timescale='epoch',
                 interval=1):
        super(StopOnNan, self).__init__(name, timescale, interval)
        self.logs_to_check = ([logs_to_check] if isinstance(logs_to_check,
                                                            string_types)
                              else logs_to_check)
        self.check_parameters = check_parameters
        self.check_training_loss = check_training_loss

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        for log_name in self.logs_to_check:
            log = get_by_path(logs, log_name)
            if not np.all(np.isfinite(log)):
                self.message("NaN or inf detected in {}!".format(log_name))
                raise StopIteration()
        if self.check_parameters:
            params = net.handler.get_numpy_copy(net.buffer.parameters)
            if not np.all(np.isfinite(params)):
                self.message("NaN or inf detected in parameters!")
                raise StopIteration()

        if self.check_training_loss and logs['training_loss']:
            if not np.all(np.isfinite(logs['training_loss'][1:])):
                self.message("NaN or inf detected in training_loss!")
                raise StopIteration()


class InfoUpdater(Hook):
    """ Save the information from logs to the Sacred custom info dict"""
    def __init__(self, run, name=None):
        super(InfoUpdater, self).__init__(name, 'epoch', 1)
        self.run = run

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        info = self.run.info
        info['epoch'] = epoch_nr
        info['monitor'] = logs
        if 'nr_parameters' not in info:
            info['nr_parameters'] = net.buffer.parameters.size


class MonitorLoss(Hook):
    def __init__(self, iter_name, timescale='epoch', interval=1, name=None,
                 verbose=None):
        super(MonitorLoss, self).__init__(name, timescale, interval, verbose)
        self.iter_name = iter_name
        self.iter = None

    def start(self, net, stepper, verbose, named_data_iters):
        super(MonitorLoss, self).start(net, stepper, verbose, named_data_iters)
        assert self.iter_name in named_data_iters
        self.iter = named_data_iters[self.iter_name]

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        return evaluate(net, self.iter, scorers=[], verbose=self.verbose)


class MonitorScores(Hook):
    """
    Monitor the losses and optionally several scores using a given data
    iterator.

    Args:
        iter_name (str):
            name of the data iterator to use (as specified in the train() call)
        scorers (List[brainstorm.scorers.Scorer]):
            List of Scorers to evaluate.
        timescale (Optional[str]):
            Specifies whether the Monitor should be called after each epoch or
            after each update. Default is 'epoch'
        interval (Optional[int]):
            This monitor should be called every ``interval`` epochs/updates.
            Default is 1
        name (Optional[str]):
            Name of this monitor. This name is used as a key in the trainer
            logs. Default is 'MonitorScores'
        verbose: bool, optional
            Specifies whether the logs of this monitor should be printed, and
            acts as a fallback verbosity for the used data iterator.
            If not set it defaults to the verbosity setting of the trainer.

    See Also:
        MonitorLoss: monitor the overall loss of the network.

    """
    def __init__(self, iter_name, scorers, timescale='epoch', interval=1,
                 name=None, verbose=None):

        super(MonitorScores, self).__init__(name, timescale, interval,
                                            verbose)
        self.iter_name = iter_name
        self.iter = None
        self.scorers = scorers

    def start(self, net, stepper, verbose, named_data_iters):
        super(MonitorScores, self).start(net, stepper, verbose,
                                         named_data_iters)
        if self.iter_name not in named_data_iters:
            raise KeyError("{} >> {} is not present in named_data_iters. "
                           "Remember to pass it  as a kwarg to Trainer.train()"
                           .format(self.__name__, self.iter_name))
        self.iter = named_data_iters[self.iter_name]

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        return evaluate(net, self.iter, self.scorers, verbose=self.verbose)


class VisualiseAccuracy(Hook):
    """
    Visualises the accuracy using the bokeh.plotting library.

    By default the output saved as a .html file, however a display can be enabled

    Parameters
    ----------
    log_names : list, array, or dict
        Contains the name of the accuracies recorded by the accuracy monitors.
        Input should be of the form <monitorname>.accuracy
    filename : str
        The location to which the .html file containing the accuracy plot should be saved
    """
    def __init__(self, log_names, filename, timescale='epoch', interval=1, name=None, verbose=None):
        super(VisualiseAccuracy, self).__init__(name, timescale, interval, verbose)

        self.log_names = log_names
        self.filename = filename

        try:
            import bokeh.plotting as bk

            self.bk = bk
            self.TOOLS = "resize,crosshair,pan,wheel_zoom,box_zoom,reset"
            self.colors = ['blue', 'green', 'red', 'olive', 'cyan', 'aqua', 'gray']

            self.bk.output_server("Accuracy Monitor")
            self.fig = self.bk.figure(title="Accuracy Monitor", x_axis_label=self.timescale, y_axis_label='accuracy',
                            tools=self.TOOLS, x_range=(0, 10), y_range=(0, 1))

        except ImportError:
            print("bokeh is required for drawing networks but was not found.")

    def start(self, net, stepper, verbose, named_data_iters):
        count = 0

        # create empty line objects
        for log_name in self.log_names:
            self.fig.line([], [], legend=log_name, line_width=2, color=self.colors[count], name=log_name)
            count += 1

        self.bk.show(self.fig)
        self.bk.output_file(self.filename + ".html", title="Accuracy Monitor", mode="cdn")

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        count = 0
        for log_name in self.log_names:
            renderer = self.fig.select(dict(name=log_name))

            datasource = renderer[0].data_source
            datasource.data["y"] = get_by_path(logs, log_name)

            if self.timescale == 'epoch':
                datasource.data["x"] = range(epoch_nr)
            elif self.timescale == 'update':
                datasource.data["x"] = range(update_nr)

            self.bk.cursession().store_objects(datasource)
            count += 1

        self.bk.save(self.fig, filename=self.filename + ".html")
