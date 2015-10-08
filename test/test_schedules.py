#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import pytest
import six

from brainstorm.training.schedules import Exponential, Linear, MultiStep


def test_linear():
    sch = Linear(initial_value=1.0, final_value=0.5, num_changes=5)
    epochs = [0] * 2 + [1] * 2 + [2] * 2 + [3] * 2 + [4] * 2
    updates = range(10)

    values = [sch(epoch, update, 'epoch', 1, None, None, None)
              for epoch, update in six.moves.zip(epochs, updates)]
    assert values == [1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6]

    values = [sch(epoch, update, 'update', 1, None, None, None)
              for epoch, update in six.moves.zip(epochs, updates)]
    assert values == [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5]

    values = [sch(epoch, update, 'update', 3, None, None, None)
              for epoch, update in six.moves.zip(epochs, updates)]
    assert values == [1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.7]


def test_exponential():
    sch = Exponential(initial_value=1.0, factor=0.99, minimum=0.97)
    epochs = [0] * 4 + [1] * 4 + [2] * 4
    updates = range(12)

    values = [sch(epoch, update, 'epoch', 1, None, None, None)
              for epoch, update in six.moves.zip(epochs, updates)]
    assert values == [1.0] * 4 + [0.99] * 4 + [0.99 * 0.99] * 4

    values = [sch(epoch, update, 'update', 1, None, None, None)
              for epoch, update in six.moves.zip(epochs, updates)]
    assert values == [1.0 * (0.99 ** x) for x in range(4)] + [0.97] * 8

    values = [sch(epoch, update, 'update', 3, None, None, None)
              for epoch, update in six.moves.zip(epochs, updates)]
    assert values == [1.0] * 3 + [0.99] * 3 + [0.9801] * 3 + [0.99 ** 3] * 3


def test_multistep():
    sch = MultiStep(initial_value=1.0, steps=[3, 5, 8],
                    values=[0.1, 0.01, 0.001])
    epochs = [0] * 2 + [1] * 2 + [2] * 2 + [3] * 2 + [4] * 2
    updates = range(10)

    values = [sch(epoch, update, 'epoch', 1, None, None, None)
              for epoch, update in six.moves.zip(epochs, updates)]
    assert values == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]

    values = [sch(epoch, update, 'update', 1, None, None, None)
              for epoch, update in six.moves.zip(epochs, updates)]
    assert values == [1.0, 1.0, 1.0, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001]

    with pytest.raises(AssertionError):
        _ = sch(0, 0, 'update', 3, None, None, None)
