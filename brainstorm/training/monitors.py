#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import six

from collections import OrderedDict
from brainstorm.describable import Describable
from brainstorm.injectors.core import Injector
from brainstorm.injectors.error_functions import ClassificationError


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
        np.save(self.filename, net.buffer.parameters[:])

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
            if self.filename is not None:
                if self.run_verbosity:
                    print(">> Saving weights to {0}...".format(self.filename))
                np.save(self.filename, net.buffer.parameters[:])
            else:
                if self.run_verbosity:
                    print(">> Caching weights")
                self.weights = net.buffer.parameters[:].copy()
        elif self.run_verbosity:
            print(">> Last saved weigths after epoch {}".format(min_error_idx))

    def load_weights(self):
        return np.load(self.filename) if self.filename is not None \
            else self.weights


class MonitorInjector(Monitor):
    """
    Monitor the given injector (aggregated over all sequences).
    """
    __undescribed__ = {'data_iter'}

    def __init__(self, data_name, injector, name=None,
                 timescale='epoch', interval=1, verbose=None):
        if name is None:
            if isinstance(injector, six.string_types):
                name = 'Monitor' + injector
            else:
                name = 'Monitor' + injector.__class__.__name__

        super(MonitorInjector, self).__init__(name, timescale, interval,
                                              verbose)
        assert isinstance(data_name, six.string_types)
        self.data_name = data_name
        self.data_iter = None
        self.injector = injector

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorInjector, self).start(net, stepper, verbose,
                                           monitor_kwargs)
        self.data_iter = monitor_kwargs[self.data_name]

    def __call__(self, epoch, net, stepper, logs):
        if isinstance(self.injector, six.string_types):
            injector = net.injectors[self.injector]
        else:
            assert isinstance(self.injector, Injector)
            injector = self.injector
        errors = []
        # noinspection PyCallingNonCallable
        for x, t in self.data_iter(self.run_verbosity):
            net.forward_pass(x)
            error, _ = injector(net.buffer.outputs[injector.layer],
                                t.get(injector.target_from))
            errors.append(error)
        return injector.aggregate(errors)


class MonitorClassificationError(MonitorInjector):
    def __init__(self, data_name, name=None, timescale='epoch', interval=1,
                 verbose=None):
        super(MonitorClassificationError, self).__init__(
            data_name, injector=ClassificationError,
            name=name, timescale=timescale, interval=interval,
            verbose=verbose)


class MonitorMultipleInjectors(Monitor):
    """
    Monitor errors (aggregated over all sequences).
    """
    __undescribed__ = {'data_iter'}

    def __init__(self, data_name, injectors,
                 name=None, timescale='epoch', interval=1, verbose=None):
        super(MonitorMultipleInjectors, self).__init__(name, timescale,
                                                       interval, verbose)
        self.iter_name = data_name
        self.data_iter = None
        self.injectors = injectors

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorMultipleInjectors, self).start(net, stepper, verbose,
                                                    monitor_kwargs)
        self.data_iter = monitor_kwargs[self.iter_name]

    def __call__(self, epoch, net, stepper, logs):
        errors = {e: [] for e in self.injectors}
        # noinspection PyCallingNonCallable
        for x, t in self.data_iter(self.run_verbosity):
            net.forward_pass(x)
            for inj in self.injectors:
                if isinstance(inj, six.string_types):
                    injector = net.injectors[inj]
                else:
                    assert isinstance(inj, Injector)
                    injector = inj
                error, _ = injector(net.buffer.outputs[injector.layer],
                                    t.get(injector.target_from))
                errors[injector].append(error)

        return {err.__name__: err.aggregate(errors[err])
                for err in self.injectors}


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
        for key, value in net.buffer.parameters[self.layer_name].items():
            log['min_' + key] = value.min()
            log['max_' + key] = value.max()
            if value.shape[1] > 1:
                log['min_sq_norm_' + key] = np.sum(value ** 2, axis=1).min()
                log['max_sq_norm_' + key] = np.sum(value ** 2, axis=1).max()
        return log