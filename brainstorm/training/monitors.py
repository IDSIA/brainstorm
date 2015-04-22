#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np

from collections import OrderedDict
from brainstorm.describable import Describable
from brainstorm.training.trainer import run_network


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

    def __init__(self, error_log_name, filename=None, name=None, verbose=None):
        super(SaveBestWeights, self).__init__(name, 'epoch', 1, verbose)
        self.error_log_name = error_log_name.split('.')
        self.filename = filename
        self.weights = None

    def __call__(self, epoch, net, stepper, logs):
        e = logs
        for en in self.error_log_name:
            e = e[en]
        min_error_idx = np.argmin(e)
        if min_error_idx == len(e) - 1:
            params = net.handler.get_numpy_copy(net.buffer.forward.parameters)
            if self.filename is not None:
                if self.run_verbosity:
                    print(">> Saving weights to {0}...".format(self.filename))
                np.save(self.filename, params)
            else:
                if self.run_verbosity:
                    print(">> Caching weights")
                self.weights = params
        elif self.run_verbosity:
            print(">> Last saved weigths after epoch {}".format(min_error_idx))

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
            raise StopIteration


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
        for i in run_network(net, iterator):
            errors.append(net.get_loss_value())
        return np.mean(errors)