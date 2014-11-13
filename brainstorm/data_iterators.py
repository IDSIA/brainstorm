#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from datetime import datetime
import math
import sys
import numpy as np
from brainstorm.randomness import Seedable
from brainstorm.targets import Targets


class ProgressBar(object):
    def __init__(self, stream=sys.stdout):
        self.start_time = datetime.utcnow()
        self.progress = 0
        self.progress_string = \
            "====1====2====3====4====5====6====7====8====9====0"
        self.prefix = '['
        self.suffix = '] Took: {0}\n'
        self.stream = stream
        self.stream.write(str(self.prefix))
        self.stream.flush()

    def update_progress(self, fraction):
        assert 0.0 <= fraction <= 1.0
        new_progress = math.trunc(fraction * len(self.progress_string))
        if new_progress > self.progress:
            self.stream.write(
                str(self.progress_string[self.progress:new_progress]))
            self.progress = new_progress
        if new_progress == len(self.progress_string):
            elapsed = datetime.utcnow() - self.start_time
            elapsed_str = str(elapsed)[:-5]
            self.stream.write(str(self.suffix.format(elapsed_str)))
        self.stream.flush()


class DataIterator(object):
    def __call__(self, verbose=False):
        pass


class Undivided(DataIterator):
    """
    Processes the data in one block (only one iteration).
    """
    def __init__(self, input_data, default=None, **named_targets):
        """
        :param input_data: Batch of sequences. shape = (time, sample, feature)
        :type input_data: ndarray
        :param default: Targets object with name 'default'
        :type default: brainstorm.targets.Target
        :param named_targets: other named targets objects
        :type named_targets: dict[str, brainstorm.targets.Target]
        """
        self.input_data = _assert_correct_input_data(input_data)
        self.targets = _construct_and_validate_targets(input_data, default,
                                                       named_targets)

    def __call__(self, verbose=False):
        yield self.input_data, self.targets


class Online(DataIterator, Seedable):
    """
    Online (one sample at a time) iterator for inputs and targets.
    """
    def __init__(self, input_data, default, shuffle=True, verbose=None,
                 seed=None, **named_targets):
        Seedable.__init__(self, seed=seed)
        self.input_data = _assert_correct_input_data(input_data)
        self.targets = _construct_and_validate_targets(input_data, default,
                                                       named_targets)
        self.shuffle = shuffle
        self.verbose = verbose

    def __call__(self, verbose=False):
        if self.verbose is not None:
            verbose = self.verbose
        p = ProgressBar() if verbose else None
        nr_sequences = self.input_data.shape[1]
        indices = np.arange(nr_sequences)
        if self.shuffle:
            self.rnd.shuffle(indices)
        for i, idx in enumerate(indices):
            targets = {}
            max_len = 0
            for t_name, tar in self.targets.items():
                targets[t_name] = tar[:, idx]
                max_len = max(max_len, targets[t_name].sequence_lengths.max())

            for tar in targets.values():
                tar.trim(max_len)

            input_data = self.input_data[:max_len, idx:idx+1, :]
            yield input_data, targets
            if verbose:
                p.update_progress((i+1)/nr_sequences)


def _assert_correct_input_data(input_data):
    assert isinstance(input_data, np.ndarray), \
        "input_data has to be of type numpy.ndarray " \
        "but was {}".format(type(input_data))
    assert input_data.ndim == 3, "input_data has to be 3D " \
                                 "but was {}".format(input_data.shape)
    return input_data


def _construct_and_validate_targets(input_data, default, named_targets):
    targets = {}
    if default is not None:
        targets['default'] = default
    targets.update(named_targets)
    for t_name, t in targets.items():
        assert isinstance(t, Targets), '{} is not an targets object!' \
                                       ''.format(t_name)
        assert input_data.shape[1] == t.shape[1], \
            "number of targets in {} doesn't match number of input " \
            "sequences ({} != {})".format(t_name, t.shape[1],
                                          input_data.shape[1])
    return targets
