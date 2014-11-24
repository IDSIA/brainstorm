#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from collections import OrderedDict
from brainstorm.describable import Describable
#from pylstm.error_functions import ClassificationError, LabelingError


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


class MonitorError(Monitor):
    """
    Monitor the given error (aggregated over all sequences).
    """
    __undescribed__ = {'data_iter'}

    def __init__(self, data_name, error_func=None,
                 name=None, timescale='epoch', interval=1):
        if name is None and error_func is not None:
            name = 'Monitor' + error_func.__name__
        super(MonitorError, self).__init__(name, timescale, interval)
        assert isinstance(data_name, basestring)
        self.data_name = data_name
        self.data_iter = None
        self.error_func = error_func

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorError, self).start(net, stepper, verbose, monitor_kwargs)
        self.data_iter = monitor_kwargs[self.data_name]

    def __call__(self, epoch, net, stepper, logs):
        error_func = self.error_func or net.error_func
        errors = []
        for x, t in self.data_iter(self.run_verbosity):
            y = net.forward_pass(x)
            error, _ = error_func(y, t)
            errors.append(error)
        return error_func.aggregate(errors)


class MonitorClassificationError(MonitorError):
    def __init__(self, data_name, name=None, timescale='epoch', interval=1):
        super(MonitorClassificationError, self).__init__(
            data_name,
            error_func=ClassificationError,
            name=name, timescale=timescale, interval=interval)


class MonitorLabelingError(MonitorError):
    def __init__(self, data_name, name=None, timescale='epoch', interval=1):
        super(MonitorLabelingError, self).__init__(
            data_name,
            error_func=LabelingError,
            name=name, timescale=timescale, interval=interval)


class MonitorMultipleErrors(Monitor):
    """
    Monitor errors (aggregated over all sequences).
    """
    __undescribed__ = {'data_iter'}

    def __init__(self, data_name, error_functions,
                 name=None, timescale='epoch', interval=1):
        super(MonitorMultipleErrors, self).__init__(name, timescale, interval)
        self.iter_name = data_name
        self.data_iter = None
        self.error_functions = error_functions

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorMultipleErrors, self).start(net, stepper, verbose,
                                                 monitor_kwargs)
        self.data_iter = monitor_kwargs[self.iter_name]

    def __call__(self, epoch, net, stepper, logs):
        errors = {e: [] for e in self.error_functions}
        for x, t in self.data_iter(self.run_verbosity):
            y = net.forward_pass(x)
            for error_func in self.error_functions:
                error, _ = error_func(y, t)
                errors[error_func].append(error)

        return {err.__name__: err.aggregate(errors[err])
                for err in self.error_functions}


class PlotMonitors(Monitor):
    """
    Open a window and plot the training and validation errors while training.
    """
    __undescribed__ = {'plt', 'fig', 'ax', 'lines', 'mins'}

    def __init__(self, name=None, show_min=True, timescale='epoch', interval=1):
        super(PlotMonitors, self).__init__(name, timescale, interval)
        self.show_min = show_min
        self.plt = None
        self.fig = None
        self.ax = None
        self.lines = None
        self.mins = None

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(PlotMonitors, self).start(net, stepper, verbose, monitor_kwargs)
        import matplotlib.pyplot as plt
        self.plt = plt
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots()
        self.ax.set_title('Training Progress')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Error')
        self.lines = dict()
        self.mins = dict()
        self.plt.show()

    def _plot(self, name, data):
        data = data[1:]  # ignore pre-training entry
        x = range(1, len(data)+1)
        if name not in self.lines:
            line, = self.ax.plot(x, data, '-', label=name)
            self.lines[name] = line
        else:
            self.lines[name].set_ydata(data)
            self.lines[name].set_xdata(x)

        if not self.show_min or len(data) < 2:
            return

        min_idx = np.argmin(data) + 1
        if name not in self.mins:
            color = self.lines[name].get_color()
            self.mins[name] = self.ax.axvline(min_idx, color=color)
        else:
            self.mins[name].set_xdata(min_idx)

    def __call__(self, epoch, net, stepper, logs):
        if epoch < 2:
            return

        for name, log in logs.items():
            if not isinstance(log, (list, dict)) or not log:
                continue
            if isinstance(log, dict):
                for k, v in log.items():
                    self._plot(name + '.' + k, v)
            else:
                self._plot(name, log)

        self.ax.legend()
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.draw()  # no idea why I need that twice, but I do


class MonitorLayerProperties(Monitor):
    """
    Monitor some properties of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None):
        if name is None:
            name = "Monitor{}Properties".format(layer_name)
        super(MonitorLayerProperties, self).__init__(name, timescale, interval)
        self.layer_name = layer_name

    def __call__(self, epoch, net, stepper, logs):
        log = OrderedDict()
        for key, value in net.get_param_view_for(self.layer_name).items():
            log['min_' + key] = value.min()
            log['max_' + key] = value.max()
            #if key.split('_')[-1] != 'bias':
            if value.shape[1] > 1:
                log['min_sq_norm_' + key] = np.sum(value ** 2, axis=1).min()
                log['max_sq_norm_' + key] = np.sum(value ** 2, axis=1).max()
        return log