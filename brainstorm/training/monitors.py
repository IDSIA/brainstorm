#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from six import string_types

from collections import OrderedDict
from brainstorm.describable import Describable
from brainstorm.training.trainer import run_network
from brainstorm.utils import get_by_path


class Monitor(Describable):
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

    def start(self, net, stepper, verbose, monitor_kwargs):
        if self.verbose is None:
            self.run_verbosity = verbose
        else:
            self.run_verbosity = self.verbose

    def __call__(self, epoch, net, stepper, logs):
        pass


class SaveWeights(Monitor):
    """
    Save the weights of the network to the given file on every call.
    Default is to save them once per epoch, but this can be configured using
    the timescale and interval parameters.
    """

    def __init__(self, filename, name=None, timescale='epoch', interval=1):
        super(SaveWeights, self).__init__(name, timescale, interval)
        self.filename = filename

    def __call__(self, epoch, net, stepper, logs):
        params = net.handler.get_numpy_copy(net.buffer.forward.parameters)
        np.save(self.filename, params)

    def load_weights(self):
        return np.load(self.filename)


class SaveBestWeights(Monitor):
    """
    Check every epoch to see if the validation error (or training error if
    there is no validation error) is at it's minimum and if so, save the
    weights to the specified file.
    """
    __undescribed__ = {'weights': None}
    __default_values__ = {'filename': None}

    def __init__(self, error_log_name, filename=None, name=None,
                 criterion='max', verbose=None):
        super(SaveBestWeights, self).__init__(name, 'epoch', 1, verbose)
        self.error_log_name = error_log_name.split('.')
        self.filename = filename
        self.weights = None
        assert criterion == 'min' or criterion == 'max'
        self.criterion = criterion

    def __call__(self, epoch, net, stepper, logs):
        e = logs
        for en in self.error_log_name:
            e = e[en]
        best_idx = np.argmin(e) if self.criterion == 'min' else np.argmax(e)
        if best_idx == len(e) - 1:
            params = net.handler.get_numpy_copy(net.buffer.forward.parameters)
            if self.filename is not None:
                if self.run_verbosity:
                    print(">> Saving weights to {0} ...".format(self.filename))
                np.save(self.filename, params)
            else:
                if self.run_verbosity:
                    print(">> Caching weights on improvement ...")
                self.weights = params
        elif self.run_verbosity:
            print(">> Last saved weights after epoch {}".format(best_idx))

    def load_weights(self):
        return np.load(self.filename) if self.filename is not None \
            else self.weights


class MonitorLayerProperties(Monitor):
    """
    Monitor some properties of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        if name is None:
            name = "Monitor{}Properties".format(layer_name)
        super(MonitorLayerProperties, self).__init__(name, timescale,
                                                     interval, verbose)
        self.layer_name = layer_name

    def __call__(self, epoch, net, stepper, logs):
        log = OrderedDict()
        for key, v in net.buffer.forward[self.layer_name].parameters.items():
            v = net.handler.get_numpy_copy(v)
            log[key] = OrderedDict()
            log[key]['min'] = v.min()
            log[key]['max'] = v.max()
            if len(v.shape) > 1 and v.shape[1] > 1:
                log[key]['min_l2'] = np.sqrt(np.sum(v ** 2, axis=1)).min()
                log[key]['max_l2'] = np.sqrt(np.sum(v ** 2, axis=1)).max()
        return log


class MaxEpochsSeen(Monitor):
    def __init__(self, max_epochs, timescale='epoch', interval=1, name=None,
                 verbose=None):
        super(MaxEpochsSeen, self).__init__(name, timescale, interval, verbose)
        self.max_epochs = max_epochs

    def __call__(self, epoch, net, stepper, logs):
        if epoch >= self.max_epochs:
            raise StopIteration("Maximum number of epochs ({}) reached."
                                .format(self.max_epochs))


class ErrorRises(Monitor):
    __default_values__ = {'delay': 1}

    def __init__(self, error, delay=1, name=None):
        super(ErrorRises, self).__init__(name, 'epoch', 1)
        self.error = error.split('.')
        self.delay = delay

    def __call__(self, epoch, net, stepper, logs):
        errors = logs
        for en in self.error:
            errors = errors[en]
        best_error_idx = np.argmin(errors)
        if len(errors) > best_error_idx + self.delay:
            raise StopIteration("Error did not fall for %d epochs! Stopping."
                                % self.delay)


class StopOnNan(Monitor):
    """ Stop the training if infinite or NaN values are found in parameters.

    Can also check logs for invalid values.
    """
    def __init__(self, logs_to_check=(), check_parameters=True, name=None):
        super(StopOnNan, self).__init__(name, 'epoch', 1)
        self.logs_to_check = ([logs_to_check] if isinstance(logs_to_check, string_types)
                              else logs_to_check)
        self.check_parameters = check_parameters

    def __call__(self, epoch, net, stepper, logs):
        for log_name in self.logs_to_check:
            log = get_by_path(logs, log_name)
            if not np.all(np.isfinite(log)):
                raise StopIteration("NaN or infinite value detected in {}"
                                    .format(log_name))
        if self.check_parameters:
            params = net.handler.get_numpy_copy(net.buffer.forward.parameters)
            if not np.all(np.isfinite(params)):
                raise StopIteration("Nan or infinite value detected in the "
                                    "parameters!")


class InfoUpdater(Monitor):
    """ Save the information from logs to the Sacred custom info dict"""
    def __init__(self, run, name=None):
        super(InfoUpdater, self).__init__(name, 'epoch', 1)
        self.run = run
        self.__name__ = self.__class__.__name__ if name is None else name

    def __call__(self, epoch, net, stepper, logs):
        info = self.run.info
        info['epoch'] = epoch
        info['monitor'] = logs
        if 'nr_parameters' not in info:
            info['nr_parameters'] = net.buffer.forward.parameters.size


class MonitorLoss(Monitor):
    def __init__(self, iter_name, timescale='epoch', interval=1, name=None,
                 verbose=None):
        super(MonitorLoss, self).__init__(name, timescale, interval, verbose)
        self.iter_name = iter_name
        self.iter = None

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorLoss, self).start(net, stepper, verbose, monitor_kwargs)
        assert self.iter_name in monitor_kwargs
        self.iter = monitor_kwargs[self.iter_name]

    def __call__(self, epoch, net, stepper, logs):
        iterator = self.iter(verbose=self.verbose, handler=net.handler)
        errors = []
        for _ in run_network(net, iterator):
            net.forward_pass()
            errors.append(net.get_loss_value())
        return np.mean(errors)


class MonitorAccuracy(Monitor):
    """
    Monitor the classification accuracy of a given layer wrt. to given targets
    using a given data iterator.

    Parameters
    ----------
    iter_name : str
        name of the data iterator to use (as specified in the train() call)
    output : str
        name of the output to use formatted like this:
        LAYER_NAME[.OUTPUT_NAME]
        Where OUTPUT_NAME defaults to 'default'
    targets_name : str, optional
        name of the targets (as specified in the InputLayer)
        defaults to 'targets'


    Other Parameters
    ----------------
    timescale : {'epoch', 'update'}, optional
        Specifies whether the Monitor should be called after each epoch or
        after each update. Default is 'epoch'
    interval : int, optional
        This monitor should be called every ``interval`` epochs/updates.
        Default is 1
    name: str, optional
        Name of this monitor. This name is used as a key in the trainer logs.
        Default is 'MonitorAccuracy'
    verbose: bool, optional
        Specifies whether the logs of this monitor should be printed, and
        acts as a fallback verbosity for the used data iterator.
        If not set it defaults to the verbosity setting of the trainer.

    See Also
    --------
    MonitorLoss : monitor the overall loss of the network.
    MonitorHammingScore : monitor the Hamming score which is a generalization
        of accuracy for multi-label classification tasks.

    Notes
    -----
    Can be used both with integer and one-hot targets.

    """
    def __init__(self, iter_name, output, targets_name='targets',
                 mask_name='mask', timescale='epoch', interval=1,
                 name=None, verbose=None):

        super(MonitorAccuracy, self).__init__(name, timescale, interval,
                                              verbose)
        self.iter_name = iter_name
        self.out_layer, _, self.out_name = output.partition('.')
        self.out_name = self.out_name or 'default'
        self.targets_name = targets_name
        self.mask_name = mask_name
        self.iter = None
        self.masked = False

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorAccuracy, self).start(net, stepper, verbose,
                                           monitor_kwargs)
        assert self.iter_name in monitor_kwargs
        assert self.out_layer in net.layers
        self.iter = monitor_kwargs[self.iter_name]
        self.masked = self.mask_name in self.iter.data.keys()

    def __call__(self, epoch, net, stepper, logs):
        iterator = self.iter(verbose=self.verbose, handler=net.handler)
        _h = net.handler
        errors = 0
        totals = 0
        for _ in run_network(net, iterator):
            net.forward_pass()
            out = _h.get_numpy_copy(net.buffer.forward[self.out_layer]
                                    .outputs[self.out_name])
            target = _h.get_numpy_copy(net.buffer.forward.InputLayer
                                       .outputs[self.targets_name])

            out = out.reshape(out.shape[0], out.shape[1], -1)
            target = target.reshape(target.shape[0], target.shape[1], -1)

            out_class = np.argmax(out, axis=2)
            if target.shape[2] > 1:
                target_class = np.argmax(target, axis=2)
            else:
                target_class = target[:, :, 0]

            assert out_class.shape == target_class.shape

            if self.masked:
                mask = _h.get_numpy_copy(net.buffer.forward.InputLayer
                                         .outputs[self.mask_name])[:, :, 0]
                errors += np.sum((out_class != target_class) * mask)
                totals += np.sum(mask)
            else:
                errors += np.sum(out_class != target_class)
                totals += np.prod(target_class.shape)

        return 1.0 - errors / totals


class MonitorHammingScore(Monitor):
    r"""
    Monitor the Hamming score of a given layer wrt. to given targets
    using a given data iterator.

    Hamming loss is defined as the fraction of the correct labels to the
    total number of labels.


    Parameters
    ----------
    iter_name : str
        name of the data iterator to use (as specified in the train() call)
    output : str
        name of the output to use formatted like this:
        LAYER_NAME[.OUTPUT_NAME]
        Where OUTPUT_NAME defaults to 'default'
    targets_name : str, optional
        name of the targets (as specified in the InputLayer)
        defaults to 'targets'


    Other Parameters
    ----------------
    timescale : {'epoch', 'update'}, optional
        Specifies whether the Monitor should be called after each epoch or
        after each update. Default is 'epoch'
    interval : int, optional
        This monitor should be called every ``interval`` epochs/updates.
        Default is 1
    name: str, optional
        Name of this monitor. This name is used as a key in the trainer logs.
        Default is 'MonitorAccuracy'
    verbose: bool, optional
        Specifies whether the logs of this monitor should be printed and
        acts as a fallback verbosity for the used data iterator.
        If not set it defaults to the verbosity setting of the trainer.

    See Also
    --------
    MonitorLoss : monitor the overall loss of the network.
    MonitorAccuracy : monitor the classification accuracy
    """
    def __init__(self, iter_name, output, targets_name, timescale='epoch',
                 interval=1, name=None, verbose=None):
        super(MonitorHammingScore, self).__init__(name, timescale, interval,
                                                  verbose)
        self.iter_name = iter_name
        self.out_layer, _, self.out_name = output.partition('.')
        self.out_name = self.out_name or 'default'
        self.targets_name = targets_name
        self.iter = None

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorHammingScore, self).start(net, stepper, verbose,
                                               monitor_kwargs)
        assert self.iter_name in monitor_kwargs
        assert self.out_layer in net.layers
        self.iter = monitor_kwargs[self.iter_name]

    def __call__(self, epoch, net, stepper, logs):
        iterator = self.iter(verbose=self.verbose, handler=net.handler)
        _h = net.handler
        errors = 0
        totals = 0
        for _ in run_network(net, iterator):
            net.forward_pass()
            out = _h.get_numpy_copy(net.buffer.forward[self.out_layer]
                                    .outputs[self.out_name])
            target = _h.get_numpy_copy(net.buffer.forward.InputLayer
                                       .outputs[self.targets_name])

            out = out.reshape(out.shape[0], out.shape[1], -1)
            target = target.reshape(target.shape[0], target.shape[1], -1)

            errors += np.sum(np.logical_xor(out >= 0.5, target))
            totals += np.prod(target.shape)

        return 1.0 - errors / totals
