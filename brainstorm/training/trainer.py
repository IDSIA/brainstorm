#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import sys
import traceback
from collections import OrderedDict

from brainstorm.describable import Describable
from brainstorm.scorers import (aggregate_losses_and_scores,
                                gather_losses_and_scores)
from brainstorm.training.utils import run_network


class Trainer(Describable):
    """
    Trainer objects organize the process of training a network. They can employ
    different training methods (``Steppers``) and call ``Hooks``.
    """
    __undescribed__ = {
        'current_epoch_nr': 0,
        'current_update_nr': 0,
        'logs': {},
        'results': {},
        'failed_hooks': {}
    }
    __default_values__ = {'verbose': True}

    def __init__(self, stepper, verbose=True):
        """Create a new Trainer.

        Args:
            stepper (brainstorm.training.steppers.TrainingStepper):
            verbose (bool):
        """
        self.stepper = stepper
        self.verbose = verbose
        self.hooks = OrderedDict()
        self.train_scorers = []
        self.current_epoch_nr = 0
        self.current_update_nr = 0
        self.logs = {}
        self.results = {}

    def add_hook(self, hook):
        """Add a hook to this trainer.

        Hooks add a variety of functionality to the trainer and can be
        called after every specified number of parameter updates or epochs.
        See documentation for ::class::`Hook` for more details.

        Note:
            During training, hooks will be called in the same order that they
            were added. This should be kept in mind when using a hook which
            relies on another hook having been called.
        Args:
            hook (brainstorm.hooks.Hook): Any ::class::`Hook` object that
                                          should be called by this trainer.
        Raises:
            ValueError: If a hook with the same name has already been added.
        """
        if hook.__name__ in self.hooks:
            raise ValueError("Hook '{}' already exists.".format(hook.__name__))
        self.hooks[hook.__name__] = hook
        hook.priority = max([h.priority for h in self.hooks.values()]) + 1

    def train(self, net, training_data_iter, **named_data_iters):
        """
        Train a network using a data iterator and further named data
        iterators.
        """
        if self.verbose:
            print('\n\n', 10 * '- ', "Before Training", 10 * ' -')
        assert set(training_data_iter.data_shapes.keys()) == set(
            net.buffer.Input.outputs.keys()), \
            "The data names provided by the training data iterator {} do not "\
            "map to the network input names {}".format(
                training_data_iter.data_shapes.keys(),
                net.buffer.Input.outputs.keys())
        self.stepper.start(net)
        named_data_iters['training_data_iter'] = training_data_iter
        self._start_hooks(net, named_data_iters)
        if self._emit_hooks(net, 'update') or self._emit_hooks(net, 'epoch'):
            return

        should_stop = False
        while not should_stop:
            self.current_epoch_nr += 1
            sys.stdout.flush()
            train_scores = {s.__name__: [] for s in self.train_scorers}
            train_scores.update({n: [] for n in net.get_loss_values()})

            if self.verbose:
                print('\n\n', 12 * '- ', "Epoch", self.current_epoch_nr,
                      12 * ' -')
            iterator = training_data_iter(handler=net.handler)
            for _ in run_network(net, iterator):
                self.current_update_nr += 1
                self.stepper.run()
                gather_losses_and_scores(net, self.train_scorers, train_scores)
                net.apply_weight_modifiers()
                if self._emit_hooks(net, 'update'):
                    should_stop = True
                    break

            self._add_log('rolling_training',
                          aggregate_losses_and_scores(train_scores, net,
                                                      self.train_scorers))

            should_stop |= self._emit_hooks(net, 'epoch')

    def evaluate(self, net, **named_data_iters):
        self._start_hooks(net, named_data_iters)
        self._emit_hooks(net, 'epoch', logs=self.results)
        self._emit_hooks(net, 'update', logs=self.results)
        return self.results

    def __init_from_description__(self, description):
        # Recover the hooks in order of priority and set their names.
        def get_priority(x):
            return getattr(x[1], 'priority', 0)
        ordered_mon = sorted(self.hooks.items(), key=get_priority)
        self.hooks = OrderedDict()
        for name, mon in ordered_mon:
            self.hooks[name] = mon
            mon.__name__ = name

    def _start_hooks(self, net, named_data_iters):
        """Call the ::attr::`start()` methods for all the hooks."""
        for name, hook in self.hooks.items():
            try:
                if hasattr(hook, 'start'):
                    hook.start(net, self.stepper, self.verbose,
                               named_data_iters)
            except Exception:
                print('An error occurred while starting the "{}" hook:'
                      .format(name), file=sys.stderr)
                raise

    def _emit_hooks(self, net, timescale, logs=None):
        """Call the hooks which should be called at this timescale."""
        should_stop = False
        count = self.current_epoch_nr if timescale == 'epoch' else \
            self.current_update_nr

        for name, hook in self.hooks.items():
            if hook.timescale != timescale or count % hook.interval != 0:
                continue

            hook_log, stop = self._call_hook(hook, net)
            should_stop |= stop
            self._add_log(name, hook_log, hook.verbose, logs=logs)

        return should_stop

    def _call_hook(self, hook, net):
        """Call a hook and check if raises a stopping signal."""
        try:
            return hook(epoch_nr=self.current_epoch_nr,
                        update_nr=self.current_update_nr,
                        net=net,
                        stepper=self.stepper, logs=self.logs), False
        except StopIteration as err:
            return getattr(err, 'value', None), True
        except Exception as e:
            print('An error occurred while calling the "{}" hook:'
                  .format(hook.__name__), file=sys.stderr)
            print(traceback.format_exc())
            raise e

    def _add_log(self, name, val, verbose=None, logs=None, indent=0):
        """Accumulate the logs (possibly a nested dictionary) recursively."""
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
