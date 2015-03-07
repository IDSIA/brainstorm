#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest

from brainstorm.structure import build_net
from brainstorm.structure.construction import ConstructionLayer, \
    ConstructionInjector
from brainstorm.initializers import Gaussian
from brainstorm.targets import FramewiseTargets
from brainstorm.data_iterators import Undivided


def test_network_forward_pass_succeeds():
    i = ConstructionLayer("InputLayer", 2)
    l1 = ConstructionLayer("FeedForwardLayer", 4, act_func='tanh')
    l2 = ConstructionLayer("FeedForwardLayer", 3, act_func='tanh')
    net = build_net(i >> l1 >> l2)
    net.initialize(Gaussian())
    net.forward_pass(np.random.randn(4, 3, 2))


def test_network_forward_backward_pass_succeed():
    i = ConstructionLayer("InputLayer", 2)
    l1 = ConstructionLayer("FeedForwardLayer", 4, act_func='tanh')
    l2 = ConstructionLayer("FeedForwardLayer", 3, act_func='tanh')
    e = ConstructionInjector("MeanSquaredError")

    i >> l1 >> l2 << e
    net = build_net(i)
    net.initialize(Gaussian())
    net.forward_pass(np.random.randn(4, 3, 2))
    net.backward_pass(
        {"default_target": FramewiseTargets(np.random.randn(4, 3, 3))})