#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.handlers import PyCudaHandler
import pytest

from brainstorm.structure import build_net
from brainstorm.structure.construction import ConstructionWrapper, \
    ConstructionInjector
from brainstorm.initializers import Gaussian
from brainstorm.targets import FramewiseTargets



def test_network_forward_pass_succeeds():
    i = ConstructionWrapper("InputLayer", 2)
    l1 = ConstructionWrapper("FeedForwardLayer", 4, act_func='tanh')
    l2 = ConstructionWrapper("FeedForwardLayer", 3, act_func='rel')
    net = build_net(i >> l1 >> l2)
    net.set_memory_handler(PyCudaHandler())
    net.initialize(Gaussian())
    net.forward_pass(np.random.randn(4, 3, 2))


def test_network_forward_backward_pass_succeed():
    i = ConstructionWrapper("InputLayer", 2)
    l1 = ConstructionWrapper("FeedForwardLayer", 4, act_func='tanh')
    l2 = ConstructionWrapper("FeedForwardLayer", 3, act_func='rel')
    e = ConstructionInjector("MeanSquaredError")

    i >> l1 >> l2 << e
    net = build_net(i)
    net.set_memory_handler(PyCudaHandler())
    net.initialize(Gaussian())
    net.forward_pass(np.random.randn(4, 3, 2))
    targets = net.handler.zeros((4, 3, 3))
    net.handler.set_from_numpy(targets, np.random.randn(4, 3, 3))
    net.backward_pass({"default_target": FramewiseTargets(targets)})