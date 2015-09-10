#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import sys
import threading
import numpy as np
from brainstorm.describable import Describable


class Trainer(Describable):
    """
    Trainer objects organize the process of training a network. They can employ
    different training methods (Steppers) and call Hooks.
    """
    __undescribed__ = {
        'current_epoch': 0,
        'logs': {},
        'failed_hooks': {}
    }
    __default_values__ = {'verbose': True}

    def __init__(self, stepper, verbose=True, double_buffering=True):
        self.stepper = stepper
        self.verbose = verbose
        self.double_buffering = double_buffering
        self.hooks = OrderedDict()
        self.current_epoch = 0
        self.logs = dict()

    def add_hook(self, hook, name=None):
        if name is None:
            name = hook.__name__
        if name in self.hooks:
            raise ValueError("Hook '{}' already exists.".format(name))
        if self.hooks:
            last = next(reversed(self.hooks))
            priority = self.hooks[last].priority + 1
        else:
            priority = 0
        self.hooks[name] = hook
        hook.__name__ = name
        hook.priority = priority

    def train(self, net, training_data_getter, **hook_kwargs):
        if self.verbose:
            print('\n\n', 15 * '- ', "Before Training", 15 * ' -')
        assert set(training_data_getter.data_names) == set(
            net.buffer.Input.outputs.keys()), \
            "The data names provided by the training data iterator {} do not " \
            "map to the network input names {}".format(
                training_data_getter.data_names,
                net.buffer.Input.outputs.keys())
        self.stepper.start(net)
        self._start_hooks(net, hook_kwargs)
        self._emit_hooks(net, 'epoch')

        run = (run_network_double_buffer if self.double_buffering else
               run_network)

        while True:
            self.current_epoch += 1
            sys.stdout.flush()
            train_loss = []
            if self.verbose:
                print('\n\n', 15 * '- ', "Epoch", self.current_epoch,
                      15 * ' -')
            iterator = training_data_getter(verbose=self.verbose,
                                            handler=net.handler)
            for i in run(net, iterator):
                train_loss.append(self.stepper.run())
                net.apply_weight_modifiers()
                if self._emit_hooks(net, 'update', i + 1):
                    break

            self._add_log('training_loss', np.mean(train_loss))
            if self._emit_hooks(net, 'epoch'):
                break

    def __init_from_description__(self, description):
        # recover the order of the Hooks from their priorities
        # and set their names
        def get_priority(x):
            return getattr(x[1], 'priority', 0)
        ordered_mon = sorted(self.hooks.items(), key=get_priority)
        self.hooks = OrderedDict()
        for name, mon in ordered_mon:
            self.hooks[name] = mon
            mon.__name__ = name

    def _start_hooks(self, net, hook_kwargs):
        self.logs = {'training_loss': [float('NaN')]}
        for name, hook in self.hooks.items():
            self._start_hook(net, name, hook, hook_kwargs)

    def _start_hook(self, net, name, hook, hook_kwargs):
        try:
            if hasattr(hook, 'start'):
                hook.start(net, self.stepper, self.verbose, hook_kwargs)
        except Exception:
            print('An error occurred while starting the "{}" hook:'
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

    def _emit_hooks(self, net, timescale, update_nr=None):
        update_nr = self.current_epoch if timescale == 'epoch' else update_nr
        should_stop = False
        for name, hook in self.hooks.items():
            if getattr(hook, 'timescale', 'epoch') != timescale:
                continue
            if update_nr % getattr(hook, 'interval', 1) == 0:
                hook_log, stop = self._call_hook(hook, net)
                should_stop |= stop
                self._add_log(name, hook_log,
                              verbose=getattr(hook, 'verbose', None))

        return should_stop

    def _call_hook(self, hook, net):
        try:
            return hook(epoch=self.current_epoch, net=net,
                        stepper=self.stepper, logs=self.logs), False
        except StopIteration as err:
            print(">> Stopping because:", err)
            if hasattr(err, 'value'):
                return err.value, True
            return None, True
        except Exception as err:
            if hasattr(err, 'args') and err.args:
                err.args = (str(err.args[0]) + " in " + str(hook),)
            raise


def run_network_double_buffer(net, iterator):
    def run_it(it):
        try:
            run_it.data = next(it)
        except StopIteration:
            run_it.data = StopIteration
    run_it.data = None

    run_it(iterator)
    i = 0
    while run_it.data != StopIteration:
        net.provide_external_data(run_it.data)
        t = threading.Thread(target=run_it, args=(iterator,))
        t.start()
        yield i
        t.join()
        i += 1


def run_network(net, iterator):
    for i, data in enumerate(iterator):
        net.provide_external_data(data)
        yield i
