#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import math
import signal
import sys
from collections import OrderedDict

import h5py
import numpy as np
from six import string_types

from brainstorm.describable import Describable
from brainstorm import optional
from brainstorm.structure.network import Network
from brainstorm.tools import evaluate
from brainstorm.utils import get_by_path, progress_bar, get_brainstorm_info


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


# -------------------------------- Saviors ---------------------------------- #

class SaveBestNetwork(Hook):
    """
    Check to see if the specified log entry is at it's best value and if so,
    save the network to a specified file.

    Can save the network when the log entry is at its minimum (such as an
    error) or maximum (such as accuracy) according to the ``criterion``
    argument.

    The ``timescale`` and ``interval`` should be the same as those for the
    monitoring hook which logs the quantity of interest.

    Args:
        log_name:
            Name of the log entry to be checked for improvement.
            It should be in the form <monitorname>.<log_name> where log_name
            itself may be a nested dictionary key in dotted notation.
        filename:
            Name of the HDF5 file to which the network should be saved.
        criterion:
            Indicates whether training should be stopped when the log entry is
            at its minimum or maximum value. Must be either 'min' or 'max'.
            Defaults to 'min'.
        name (Optional[str]):
            Name of this monitor. This name is used as a key in the trainer
            logs. Default is 'SaveBestNetwork'.
        timescale (Optional[str]):
            Specifies whether the Monitor should be called after each epoch or
            after each update. Default is 'epoch'.
        interval (Optional[int]):
            This monitor should be called every ``interval`` epochs/updates.
            Default is 1.
        verbose: bool, optional
            Specifies whether the logs of this monitor should be printed, and
            acts as a fallback verbosity for the used data iterator.
            If not set it defaults to the verbosity setting of the trainer.
    Examples:
        Add a hook to monitor a quantity of interest:

        >>> scorer = bs.scorers.Accuracy()
        >>> trainer.add_hook(bs.hooks.MonitorScores('valid_getter', [scorer],
        ...                                         name='validation'))

        Check every epoch and save the network if validation accuracy rises:

        >>> trainer.add_hook(bs.hooks.SaveBestNetwork('validation.Accuracy',
        ...                                           filename='best_acc.h5',
        ...                                           criterion='max'))

        Check every epoch and save the network if validation loss drops:

        >>> trainer.add_hook(bs.hooks.SaveBestNetwork('validation.total_loss',
            ...                                       filename='best_loss.h5',
        ...                                           criterion='min'))
    """
    __undescribed__ = {'parameters': None}
    __default_values__ = {'filename': None}

    def __init__(self, log_name, filename=None, criterion='max', name=None,
                 timescale='epoch', interval=1, verbose=None):
        super(SaveBestNetwork, self).__init__(name, timescale,
                                              interval, verbose)
        self.log_name = log_name
        self.filename = filename
        self.parameters = None
        assert criterion == 'min' or criterion == 'max'
        self.best_so_far = np.inf if criterion == 'min' else -np.inf
        self.best_t = None
        self.criterion = criterion

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        if epoch_nr == 0:
            try:
                e = get_by_path(logs, self.log_name)
            except KeyError:
                return

        e = get_by_path(logs, self.log_name)
        last = e[-1]
        if self.criterion == 'min':
            imp = last < self.best_so_far
        else:
            imp = last > self.best_so_far

        if imp:
            self.best_so_far = last
            self.best_t = epoch_nr if self.timescale == 'epoch' else update_nr
            params = net.get('parameters')
            if self.filename is not None:
                self.message("{} improved (criterion: {}). Saving network to "
                             "{}".format(self.log_name, self.criterion,
                                         self.filename))
                net.save_as_hdf5(self.filename)
            else:
                self.message("{} improved (criterion: {}). Caching parameters".
                             format(self.log_name, self.criterion))
                self.parameters = params
        else:
            self.message("Last saved parameters at {} {} when {} was {}".
                         format(self.timescale, self.best_t, self.log_name,
                                self.best_so_far))

    def load_parameters(self):
        return np.load(self.filename) if self.filename is not None \
            else self.parameters


class SaveLogs(Hook):
    """
    Periodically Save the trainer logs dictionary to an HDF5 file.
    Default behavior is to save every epoch.
    """
    def __init__(self, filename, name=None, timescale='epoch', interval=1):
        super(SaveLogs, self).__init__(name, timescale, interval)
        self.filename = filename

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        with h5py.File(self.filename, 'w') as f:
            f.attrs.create('info', get_brainstorm_info())
            f.attrs.create('format', b'Logs file v1.0')
            SaveLogs._save_recursively(f, logs)

    @staticmethod
    def _save_recursively(group, logs):
        for name, log in logs.items():
            if isinstance(log, dict):
                subgroup = group.create_group(name)
                SaveLogs._save_recursively(subgroup, log)
            else:
                group.create_dataset(name, data=np.array(log))


class SaveNetwork(Hook):
    """
    Periodically save the weights of the network to the given file.
    Default behavior is to save the network after every training epoch.
    """

    def __init__(self, filename, name=None, timescale='epoch', interval=1):
        super(SaveNetwork, self).__init__(name, timescale, interval)
        self.filename = filename

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        net.save_as_hdf5(self.filename)

    def load_network(self):
        return Network.from_hdf5(self.filename)


# -------------------------------- Monitors --------------------------------- #

class MonitorLayerDeltas(Hook):
    """
    Monitor some statistics about all the deltas of a layer.
    """
    def __init__(self, layer_name, name=None, timescale='epoch', interval=1,
                 verbose=None):
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


class MonitorLayerGradients(Hook):
    """
    Monitor some statistics about all the gradients of a layer.
    """
    def __init__(self, layer_name, name=None, timescale='epoch', interval=1,
                 verbose=None):
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


class MonitorLayerInOuts(Hook):
    """
    Monitor some statistics about all the inputs and outputs of a layer.
    """
    def __init__(self, layer_name, name=None, timescale='epoch', interval=1,
                 verbose=None):
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


class MonitorLayerParameters(Hook):
    """
    Monitor some statistics about all the parameters of a layer.
    """
    def __init__(self, layer_name, name=None, timescale='epoch', interval=1,
                 verbose=None):
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


class MonitorLoss(Hook):
    """
    Monitor the losses computed by the network on a dataset using a given data
    iterator.
    """
    def __init__(self, iter_name, name=None, timescale='epoch', interval=1,
                 verbose=None):
        super(MonitorLoss, self).__init__(name, timescale, interval, verbose)
        self.iter_name = iter_name
        self.iter = None

    def start(self, net, stepper, verbose, named_data_iters):
        super(MonitorLoss, self).start(net, stepper, verbose, named_data_iters)
        if self.iter_name not in named_data_iters:
            raise KeyError("{} >> {} is not present in named_data_iters. "
                           "Remember to pass it  as a kwarg to Trainer.train()"
                           .format(self.__name__, self.iter_name))
        self.iter = named_data_iters[self.iter_name]

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        return evaluate(net, self.iter, scorers=())


class MonitorScores(Hook):
    """
    Monitor the losses and optionally several scores using a given data
    iterator.

    Args:
        iter_name (str):
            name of the data iterator to use (as specified in the train() call)
        scorers (List[brainstorm.scorers.Scorer]):
            List of Scorers to evaluate.
        name (Optional[str]):
            Name of this monitor. This name is used as a key in the trainer
            logs. Default is 'MonitorScores'
        timescale (Optional[str]):
            Specifies whether the Monitor should be called after each epoch or
            after each update. Default is 'epoch'.
        interval (Optional[int]):
            This monitor should be called every ``interval`` epochs/updates.
            Default is 1.
        verbose: bool, optional
            Specifies whether the logs of this monitor should be printed, and
            acts as a fallback verbosity for the used data iterator.
            If not set it defaults to the verbosity setting of the trainer.

    See Also:
        MonitorLoss: monitor the overall loss of the network.

    """
    def __init__(self, iter_name, scorers, name=None, timescale='epoch',
                 interval=1, verbose=None):

        super(MonitorScores, self).__init__(name, timescale, interval, verbose)
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
        return evaluate(net, self.iter, self.scorers)


# -------------------------------- Stoppers --------------------------------- #

class EarlyStopper(Hook):
    """
    Stop the training if a log entry does not improve for some time.

    Can stop training when the log entry is at its minimum (such as an error)
    or maximum (such as accuracy) according to the ``criterion`` argument.

    The ``timescale`` and ``interval`` should be the same as those for the
    monitoring hook which logs the quantity of interest.

    Args:
        log_name:
            Name of the log entry to be checked for improvement.
            It should be in the form <monitorname>.<log_name> where log_name
            itself may be a nested dictionary key in dotted notation.
        patience:
            Number of log updates to wait before stopping training.
            Default is 1.
        criterion:
            Indicates whether training should be stopped when the log entry is
            at its minimum or maximum value. Must be either 'min' or 'max'.
            Defaults to 'min'.
        name (Optional[str]):
            Name of this monitor. This name is used as a key in the trainer
            logs. Default is 'EarlyStopper'.
        timescale (Optional[str]):
            Specifies whether the Monitor should be called after each epoch or
            after each update. Default is 'epoch'.
        interval (Optional[int]):
            This monitor should be called every ``interval`` epochs/updates.
            Default is 1.
        verbose: bool, optional
            Specifies whether the logs of this monitor should be printed, and
            acts as a fallback verbosity for the used data iterator.
            If not set it defaults to the verbosity setting of the trainer.
    Examples:
        Add a hook to monitor a quantity of interest:

        >>> scorer = bs.scorers.Accuracy()
        >>> trainer.add_hook(bs.hooks.MonitorScores('valid_getter', [scorer],
        ...                                         name='validation'))

        Stop training if validation set accuracy does not rise for 10 epochs:

        >>> trainer.add_hook(bs.hooks.EarlyStopper('validation.Accuracy',
        ...                                        patience=10,
        ...                                        criterion='max'))

        Stop training if loss on validation set does not drop for 5 epochs:

        >>> trainer.add_hook(bs.hooks.EarlyStopper('validation.total_loss',
        ...                                        patience=5,
        ...                                        criterion='min'))

    """
    __default_values__ = {'patience': 1}

    def __init__(self, log_name, patience=1, criterion='min',
                 name=None, timescale='epoch', interval=1, verbose=None):
        super(EarlyStopper, self).__init__(name, timescale, interval, verbose)
        self.log_name = log_name
        self.patience = patience
        if criterion not in ['min', 'max']:
            raise ValueError("Unknown criterion: '{}'"
                             "(Should be 'min' or 'max')".format(criterion))
        self.criterion = criterion

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        if epoch_nr == 0:
            try:
                e = get_by_path(logs, self.log_name)
            except KeyError:
                return
        e = get_by_path(logs, self.log_name)
        best_idx = np.argmin(e) if self.criterion == 'min' else np.argmax(e)
        if len(e) > best_idx + self.patience:
            self.message("Stopping because {} did not improve for {} checks "
                         "(criterion used : {}).".format(self.log_name,
                                                         self.patience,
                                                         self.criterion))
            raise StopIteration()


class StopAfterEpoch(Hook):
    """
    Stop the training after a specified number of epochs.

    Args:
        max_epochs (int):
            The number of epochs to train.
        name (Optional[str]):
            Name of this monitor. This name is used as a key in the trainer
            logs. Default is 'StopAfterEpoch'.
        timescale (Optional[str]):
            Specifies whether the Monitor should be called after each epoch or
            after each update. Default is 'epoch'.
        interval (Optional[int]):
            This monitor should be called every ``interval`` epochs/updates.
            Default is 1.
        verbose: bool, optional
            Specifies whether the logs of this monitor should be printed, and
            acts as a fallback verbosity for the used data iterator.
            If not set it defaults to the verbosity setting of the trainer.
    """
    def __init__(self, max_epochs, name=None, timescale='epoch', interval=1,
                 verbose=None):
        super(StopAfterEpoch, self).__init__(name, timescale,
                                             interval, verbose)
        self.max_epochs = max_epochs

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        if epoch_nr >= self.max_epochs:
            self.message("Stopping because the maximum number of epochs ({}) "
                         "was reached.".format(self.max_epochs))
            raise StopIteration()


class StopAfterThresholdReached(Hook):
    """
    Stop the training if a log entry reaches the given threshold

    Can stop training when the log entry becomes sufficiently small (such as an
    error) or sufficiently large (such as accuracy) according to the threshold.

    Args:
        log_name:
            Name of the log entry to be checked for improvement.
            It should be in the form <monitorname>.<log_name> where log_name
            itself may be a nested dictionary key in dotted notation.
        threshold:
            The threshold value to reach
        criterion:
            Indicates whether training should be stopped when the log entry is
            at its minimum or maximum value. Must be either 'min' or 'max'.
            Defaults to 'min'.
        name (Optional[str]):
            Name of this monitor. This name is used as a key in the trainer
            logs. Default is 'StopAfterThresholdReached'.
        timescale (Optional[str]):
            Specifies whether the Monitor should be called after each epoch or
            after each update. Default is 'epoch'.
        interval (Optional[int]):
            This monitor should be called every ``interval`` epochs/updates.
            Default is 1.
        verbose: bool, optional
            Specifies whether the logs of this monitor should be printed, and
            acts as a fallback verbosity for the used data iterator.
            If not set it defaults to the verbosity setting of the trainer.
    Examples:
        Stop training if validation set accuracy is at least 97 %:

        >>> trainer.add_hook(StopAfterThresholdReached('validation.Accuracy',
        ...                                            threshold=0.97,
        ...                                            criterion='max'))

        Stop training if loss on validation set goes below 0.2:

        >>> trainer.add_hook(StopAfterThresholdReached('validation.total_loss',
        ...                                            threshold=0.2,
        ...                                            criterion='min'))

    """

    def __init__(self, log_name, threshold, criterion='min',
                 name=None, timescale='epoch', interval=1, verbose=None):
        super(StopAfterThresholdReached, self).__init__(name, timescale,
                                                        interval, verbose)
        self.log_name = log_name
        self.threshold = threshold
        if criterion not in ['min', 'max']:
            raise ValueError("Unknown criterion: '{}'"
                             "(Must be 'min' or 'max')".format(criterion))
        self.criterion = criterion

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        e = get_by_path(logs, self.log_name)
        is_threshold_reached = False
        if self.criterion == 'max' and max(e) >= self.threshold:
            is_threshold_reached = True
        elif self.criterion == 'min' and min(e) <= self.threshold:
            is_threshold_reached = True
        if is_threshold_reached:
            self.message("Stopping because {} has reached the threshold {} "
                         "(criterion used : {})".format(
                             self.log_name, self.threshold, self.criterion))
            raise StopIteration()


class StopOnNan(Hook):
    """
    Stop the training if infinite or NaN values are found in parameters.

    This hook can also check a list of logs for invalid values.

    Args:
        logs_to_check (Optional[list, tuple]):
            A list of trainer logs to check in dotted notation. Defaults to ().
        check_parameters (Optional[bool]):
            Indicates whether the parameters should be checked for NaN.
            Defaults to True.
        name (Optional[str]):
            Name of this monitor. This name is used as a key in the trainer
            logs. Default is 'StopOnNan'.
        timescale (Optional[str]):
            Specifies whether the Monitor should be called after each epoch or
            after each update. Default is 'epoch'.
        interval (Optional[int]):
            This monitor should be called every ``interval`` epochs/updates.
            Default is 1.
        verbose: bool, optional
            Specifies whether the logs of this monitor should be printed, and
            acts as a fallback verbosity for the used data iterator.
            If not set it defaults to the verbosity setting of the trainer.
    """
    def __init__(self, logs_to_check=(), check_parameters=True,
                 check_training_loss=True, name=None, timescale='epoch',
                 interval=1, verbose=None):
        super(StopOnNan, self).__init__(name, timescale, interval, verbose)
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
            if not net.handler.is_fully_finite(net.buffer.parameters):
                self.message("NaN or inf detected in parameters!")
                raise StopIteration()

        if self.check_training_loss and 'rolling_training' in logs:
            rtrain = logs['rolling_training']
            if 'total_loss' in rtrain:
                loss = rtrain['total_loss']
            else:
                loss = rtrain['Loss']
            if not np.all(np.isfinite(loss)):
                self.message("NaN or inf detected in rolling training loss!")
                raise StopIteration()


class StopOnSigQuit(Hook):
    """
    Stop training after the next call if it received a SIGQUIT (Ctrl + \).

    This hook makes it possible to exit the training loop and continue with
    the rest of the program execution.

    Args:
        name (Optional[str]):
            Name of this monitor. This name is used as a key in the trainer
            logs. Default is 'StopOnSigQuit'.
        timescale (Optional[str]):
            Specifies whether the Monitor should be called after each epoch or
            after each update. Default is 'epoch'.
        interval (Optional[int]):
            This monitor should be called every ``interval`` epochs/updates.
            Default is 1.
        verbose: bool, optional
            Specifies whether the logs of this monitor should be printed, and
            acts as a fallback verbosity for the used data iterator.
            If not set it defaults to the verbosity setting of the trainer.
    """
    __undescribed__ = {'quit': False}

    def __init__(self, name=None, timescale='epoch', interval=1, verbose=None):
        super(StopOnSigQuit, self).__init__(name, timescale, interval,
                                            verbose=verbose)
        self.quit = False

    def start(self, net, stepper, verbose, named_data_iters):
        super(StopOnSigQuit, self).start(net, stepper, verbose,
                                         named_data_iters)
        self.quit = False
        signal.signal(signal.SIGQUIT, self.receive_signal)

    def receive_signal(self, signum, stack):
        self.message('Interrupting')
        self.quit = True

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        if self.quit:
            raise StopIteration('Received SIGQUIT signal.')


# ------------------------------ Visualizers -------------------------------- #

if not optional.has_bokeh:
    BokehVisualizer = optional.bokeh_mock
else:
    import bokeh.plotting as bk
    import warnings

    class BokehVisualizer(Hook):
        """
        Visualizes log values in your browser during training time using the
        Bokeh plotting library.

        Before running the trainer the user is required to have the Bokeh
        Server running.

        By default the visualization is discarded upon closing the webbrowser.
        However if an output file is specified then the .html file will be
        saved after each iteration at the specified location.

        Args:
            log_names (list, array):
                Contains the name of the logs that are being recorded to be
                visualized. log_names should be of the form
                <monitorname>.<log_name> where log_name itself may be a nested
                dictionary key in dotted notation.
            filename (Optional, str):
                The location to which the .html file containing the accuracy
                plot should be saved.
            timescale (Optional[str]):
                Specifies whether the Monitor should be called after each
                epoch or after each update. Default is 'epoch'
            interval (Optional[int]):
                This monitor should be called every ``interval``
                number of epochs/updates. Default is 1.
            name (Optional[str]):
                Name of this monitor. This name is used as a key in the trainer
                logs. Default is 'MonitorScores'
            verbose: bool, optional
                Specifies whether the logs of this monitor should be printed,
                and acts as a fallback verbosity for the used data iterator.
                If not set it defaults to the verbosity setting of the trainer.
        """
        def __init__(self, log_names, filename=None, timescale='epoch',
                     interval=1, name=None, verbose=None):
            super(BokehVisualizer, self).__init__(name, timescale, interval,
                                                  verbose)

            if isinstance(log_names, string_types):
                self.log_names = [log_names]
            elif isinstance(log_names, (tuple, list)):
                self.log_names = log_names
            else:
                raise ValueError('log_names must be either str or list but'
                                 ' was {}'.format(type(log_names)))

            self.filename = filename

            self.bk = bk
            self.TOOLS = "resize,crosshair,pan,wheel_zoom,box_zoom,reset,save"
            self.colors = ['blue', 'green', 'red', 'olive', 'cyan', 'aqua',
                           'gray']

            warnings.filterwarnings('error')
            try:
                self.bk.output_server(self.__name__)
                warnings.resetwarnings()
            except Warning:
                raise StopIteration('Bokeh server is not running')

            self.fig = self.bk.figure(
                title=self.__name__, x_axis_label=self.timescale,
                y_axis_label='value', tools=self.TOOLS,
                plot_width=1000, x_range=(0, 25), y_range=(0, 1))

        def start(self, net, stepper, verbose, named_data_iters):
            count = 0

            # create empty line objects
            for log_name in self.log_names:
                self.fig.line([], [], legend=log_name, line_width=2,
                              color=self.colors[count % len(self.colors)],
                              name=log_name)
                count += 1

            self.bk.show(self.fig)
            self.bk.output_file('bokeh_visualisation.html',
                                title=self.__name__, mode='cdn')

        def __call__(self, epoch_nr, update_nr, net, stepper, logs):
            if epoch_nr == 0:
                return
            for log_name in self.log_names:
                renderer = self.fig.select(dict(name=log_name))

                datasource = renderer[0].data_source
                datasource.data["y"] = get_by_path(logs, log_name)

                datasource.data["x"] = range(len(datasource.data["y"]))
                self.bk.cursession().store_objects(datasource)

            if self.filename is not None:
                self.bk.save(self.fig, filename=self.filename + ".html")


class ProgressBar(Hook):
    """ Adds a progress bar to show the training progress. """
    def __init__(self):
        super(ProgressBar, self).__init__(None, 'update', 1)
        self.length = None
        self.bar = None

    def start(self, net, stepper, verbose, named_data_iters):
        assert 'training_data_iter' in named_data_iters
        self.length = named_data_iters['training_data_iter'].length

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        assert epoch_nr == 0 or math.ceil(update_nr / self.length) == epoch_nr
        if update_nr % self.length == 1:
            self.bar = progress_bar(self.length)
            print(next(self.bar), end='')
            sys.stdout.flush()
        elif update_nr % self.length == 0:
            if self.bar:
                print(self.bar.send(self.length))
        else:
            print(self.bar.send(update_nr % self.length), end='')
            sys.stdout.flush()


# ----------------------------- Miscellaneous ------------------------------- #

class InfoUpdater(Hook):
    """ Save the information from logs to the Sacred custom info dict"""
    def __init__(self, run, name=None, timescale='epoch', interval=1):
        super(InfoUpdater, self).__init__(name, timescale, interval)
        self.run = run

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        info = self.run.info
        info['epoch_nr'] = epoch_nr
        info['update_nr'] = update_nr
        info['logs'] = logs
        if 'nr_parameters' not in info:
            info['nr_parameters'] = net.buffer.parameters.size


class ModifyStepperAttribute(Hook):
    """Modify an attribute of the training stepper."""
    def __init__(self, schedule, attr_name='learning_rate',
                 timescale='epoch', interval=1, name=None, verbose=None):
        super(ModifyStepperAttribute, self).__init__(name, timescale,
                                                     interval, verbose)
        self.schedule = schedule
        self.attr_name = attr_name

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(ModifyStepperAttribute, self).start(net, stepper, verbose,
                                                  monitor_kwargs)
        assert hasattr(stepper, self.attr_name), \
            "The stepper {} does not have the attribute {}".format(
                stepper.__class__.__name__, self.attr_name)

    def __call__(self, epoch_nr, update_nr, net, stepper, logs):
        setattr(stepper, self.attr_name,
                self.schedule(epoch_nr, update_nr, self.timescale,
                              self.interval, net, stepper, logs))
