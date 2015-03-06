#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest

from brainstorm.structure import build_net
from brainstorm.structure.construction import ConstructionLayer
from brainstorm.initializers import Gaussian


def test_network_forward_pass():
    i = ConstructionLayer("InputLayer", 2)
    l1 = ConstructionLayer("FeedForwardLayer", 4, act_func='tanh')
    l2 = ConstructionLayer("FeedForwardLayer", 3, act_func='tanh')
    net = build_net(i >> l1 >> l2)
    net.initialize(Gaussian())
    net.forward_pass(np.random.randn(4, 3, 2))