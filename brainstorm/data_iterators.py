#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from datetime import datetime
import math
import numpy as np
from brainstorm.randomness import Seedable
from brainstorm.targets import Targets


def progress_bar(maximum, prefix='[',
                 bar='====1====2====3====4====5====6====7====8====9====0',
                 suffix='] Took: {0}\n'):
    i = 0
    start_time = datetime.utcnow()
    out = prefix
    while i < len(bar):
        progress = yield out
        j = math.trunc(progress/maximum * len(bar))
        out = bar[i:j]
        i = j
    elapsed_str = str(datetime.utcnow() - start_time)[:-5]
    yield out + suffix.format(elapsed_str)


def silence():
    while True:
        _ = yield ''


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
    def __init__(self, input_data, default=None, shuffle=True, verbose=None,
                 seed=None, **named_targets):
        Seedable.__init__(self, seed=seed)
        self.input_data = _assert_correct_input_data(input_data)
        self.targets = _construct_and_validate_targets(input_data, default,
                                                       named_targets)
        self.shuffle = shuffle
        self.verbose = verbose

    def __call__(self, verbose=False):
        nr_sequences = self.input_data.shape[1]
        if (self.verbose is None and verbose) or self.verbose:
            p_bar = progress_bar(nr_sequences)
        else:
            p_bar = silence()

        print(next(p_bar), end='')
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
            print(p_bar.send(i+1), end='')


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
