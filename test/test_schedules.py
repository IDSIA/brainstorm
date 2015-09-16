#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.training.schedules import Constant, Linear, Exponential, \
    MultiStep


def test_constant():
    sch = Constant(0.1)
    values = [sch() for _ in range(10)]
    assert values == [0.1] * 10


def test_linear():
    sch = Linear(initial_value=1.0, final_value=0.5, num_changes=5, interval=1)
    values = [sch() for _ in range(10)]
    assert values == [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5]


def test_exponential():
    sch = Exponential(initial_value=1.0, interval=1, factor=0.99, minimum=0.97)
    values = [sch() for _ in range(5)]
    assert values == [1.0, 0.99, 0.9801, 0.970299, 0.97]


def test_multistep():
    sch = MultiStep(initial_value=1.0, steps=[3, 5, 8],
                    values=[0.1, 0.01, 0.001])
    values = [sch() for _ in range(10)]
    assert values == [1.0, 1.0, 1.0, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001]

