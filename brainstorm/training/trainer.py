#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import sys
import numpy as np
from brainstorm.describable import Describable


class Trainer(Describable):
    """
    Trainer objects organize the process of training a network. They can employ
    different training methods (Steppers) and call Monitors.
    """
    __undescribed__ = {
        'current_epoch': 0,
        'logs': {},
        'failed_monitors': {}
    }
    __default_values__ = {'verbose': True}

    def __init__(self, stepper, verbose=True):
        self.stepper = stepper
        self.verbose = verbose
        self.monitors = OrderedDict()
        self.current_epoch = 0
        self.logs = dict()

    def add_monitor(self, monitor, name=None):
        if name is None:
            name = monitor.__name__
        if name in self.monitors:
            raise ValueError("Monitor '{}' already exists.".format(name))
        if self.monitors:
            last = next(reversed(self.monitors))
            priority = self.monitors[last].priority + 1
        else:
            priority = 0
        self.monitors[name] = monitor
        monitor.__name__ = name
        monitor.priority = priority

    def train(self, net, training_data_getter, **monitor_kwargs):
        if self.verbose:
            print('\n\n', 15 * '- ', "Before Training", 15 * ' -')
        self.stepper.start(net)
        self._start_monitors(net, monitor_kwargs)
        self._emit_monitoring(net, 'epoch')
        while True:
            self.current_epoch += 1
            sys.stdout.flush()
            train_errors = []
            if self.verbose:
                print('\n\n', 15 * '- ', "Epoch", self.current_epoch,
                      15 * ' -')
            for i, data in enumerate(
                    training_data_getter(verbose=self.verbose)):
                train_errors.append(self.stepper.run(data))
                if self._emit_monitoring(net, 'update', i + 1):
                    break

            self._add_log('training_errors', np.mean(train_errors))
            if self._emit_monitoring(net, 'epoch'):
                break

    def __init_from_description__(self, description):
        # recover the order of the monitors from their priorities
        # and set their names
        def get_priority(x):
            return getattr(x[1], 'priority', 0)
        ordered_mon = sorted(self.monitors.items(), key=get_priority)
        self.monitors = OrderedDict()
        for name, mon in ordered_mon:
            self.monitors[name] = mon
            mon.__name__ = name

    def _start_monitors(self, net, monitor_kwargs):
        self.logs = {'training_errors': [float('NaN')]}
        for name, monitor in self.monitors.items():
            self._start_monitor(net, name, monitor, monitor_kwargs)

    def _start_monitor(self, net, name, monitor, monitor_kwargs):
        try:
            if hasattr(monitor, 'start'):
                monitor.start(net, self.stepper, self.verbose, monitor_kwargs)
        except Exception:
            print('An error occurred while starting the "{}" monitor:'
                  ''.format(name), file=sys.stderr)
            raise

    def _add_log(self, name, val, verbose=None, logs=None, indent=0):
        if verbose is None:
            verbose = self.verbose
        if logs is None:
            logs = self.logs
        if isinstance(val, dict):
            if verbose:
                print(" " * indent + name)
            if name not in logs:
                logs[name] = dict()
            for k, v in val.items():
                self._add_log(k, v, verbose, logs[name], indent+2)
        elif val is not None:
            if verbose:
                print(" " * indent + ("{0:%d}: {1}" % (40-indent)).format(name,
                                                                          val))
            if name not in logs:
                logs[name] = []
            logs[name].append(val)

    def _emit_monitoring(self, net, timescale, update_nr=None):
        update_nr = self.current_epoch if timescale == 'epoch' else update_nr
        should_stop = False
        for name, monitor in self.monitors.items():
            if getattr(monitor, 'timescale', 'epoch') != timescale:
                continue
            if update_nr % getattr(monitor, 'interval', 1) == 0:
                monitor_log, stop = self._call_monitor(monitor, net)
                should_stop |= stop
                self._add_log(name, monitor_log,
                              verbose=getattr(monitor, 'verbose', None))

        return should_stop

    def _call_monitor(self, monitor, net):
        try:
            return monitor(epoch=self.current_epoch, net=net,
                           stepper=self.stepper, logs=self.logs), False
        except StopIteration as err:
            print(">> Stopping because:", err)
            if hasattr(err, 'value'):
                return err.value, True
            return None, True
        except Exception as err:
            if hasattr(err, 'args') and err.args:
                err.args = (str(err.args[0]) + " in " + str(monitor),)
            raise
