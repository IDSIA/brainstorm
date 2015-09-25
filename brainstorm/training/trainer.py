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
        'current_epoch_nr': 0,
        'current_update_nr': 0,
        'logs': {},
        'failed_hooks': {}
    }
    __default_values__ = {'verbose': True}

    def __init__(self, stepper, verbose=True, double_buffering=True):
        self.stepper = stepper
        self.verbose = verbose
        self.double_buffering = double_buffering
        self.hooks = OrderedDict()
        self.current_epoch_nr = 0
        self.current_update_nr = 0
        self.logs = dict()

    def add_hook(self, hook):
        """Add a hook to this trainer.

        Note that hooks will be called in the order that they are added.

        :param hook: Any Hook object that should be called by this trainer.
        :type hook: brainstorm.hooks.Hook
        :raises ValueError: If a hook with the same name has already been added
        """
        if hook.__name__ in self.hooks:
            raise ValueError("Hook '{}' already exists.".format(hook.__name__))
        self.hooks[hook.__name__] = hook
        hook.priority = max([h.priority for h in self.hooks.values()]) + 1

    def train(self, net, training_data_getter, **hook_kwargs):
        if self.verbose:
            print('\n\n', 15 * '- ', "Before Training", 15 * ' -')
        assert set(training_data_getter.data.keys()) == set(
            net.buffer.Input.outputs.keys()), \
            "The data names provided by the training data iterator {} do not "\
            "map to the network input names {}".format(
                training_data_getter.data.keys(),
                net.buffer.Input.outputs.keys())
        self.stepper.start(net)
        self._start_hooks(net, hook_kwargs)
        if self._emit_hooks(net, 'epoch'):
            return

        run = (run_network_double_buffer if self.double_buffering else
               run_network)

        while True:
            self.current_epoch_nr += 1
            sys.stdout.flush()
            train_loss = []
            if self.verbose:
                print('\n\n', 15 * '- ', "Epoch", self.current_epoch_nr,
                      15 * ' -')
            iterator = training_data_getter(verbose=self.verbose,
                                            handler=net.handler)
            for _ in run(net, iterator):
                self.current_update_nr += 1
                train_loss.append(self.stepper.run())
                net.apply_weight_modifiers()
                if self._emit_hooks(net, 'update'):
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
            try:
                if hasattr(hook, 'start'):
                    hook.start(net, self.stepper, self.verbose, hook_kwargs)
            except Exception:
                print('An error occurred while starting the "{}" hook:'
                      .format(name), file=sys.stderr)
                raise

    def _emit_hooks(self, net, timescale):
        should_stop = False
        count = self.current_epoch_nr if timescale == 'epoch' else \
            self.current_update_nr

        for name, hook in self.hooks.items():
            if hook.timescale != timescale or count % hook.interval != 0:
                continue

            hook_log, stop = self._call_hook(hook, net)
            should_stop |= stop
            self._add_log(name, hook_log, hook.verbose)

        return should_stop

    def _call_hook(self, hook, net):
        try:
            return hook(epoch_nr=self.current_epoch_nr,
                        update_nr=self.current_update_nr,
                        net=net,
                        stepper=self.stepper, logs=self.logs), False
        except StopIteration as err:
            print(">> Stopping because:", err)
            return getattr(err, 'value', None), True
        except Exception:
            print('An error occurred while calling the "{}" hook:'
                  .format(hook.__name__), file=sys.stderr)
            raise

    def _add_log(self, name, val, verbose=None, logs=None, indent=0):
        if val is None:
            return

        verbose = self.verbose if verbose is None else verbose
        logs = self.logs if logs is None else logs

        if isinstance(val, dict):
            if verbose:
                print(" " * indent + name)
            logs[name] = dict() if name not in logs else logs[name]

            for k, v in val.items():
                self._add_log(k, v, verbose, logs[name], indent + 2)
        else:
            if verbose:
                print(" " * indent + ("{0:%d}: {1}" % (40 - indent))
                      .format(name, val))
            logs[name] = [] if name not in logs else logs[name]
            logs[name].append(val)


def run_network_double_buffer(net, iterator):
    def run_it(it):
        try:
            net.provide_external_data(next(it))
        except StopIteration:
            run_it.stop = True

    run_it.stop = False

    run_it(iterator)
    i = 0
    while not run_it.stop:
        t = threading.Thread(target=run_it, args=(iterator,))
        t.start()
        yield i
        t.join()
        i += 1


def run_network(net, iterator):
    for i, data in enumerate(iterator):
        net.provide_external_data(data)
        yield i
