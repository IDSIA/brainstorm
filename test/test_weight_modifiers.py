#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.weight_modifiers import ConstrainL2Norm
from brainstorm.handlers import PyCudaHandler, default_handler


def test_limit_incoming_weights_squared():

    for orig in (np.random.rand(4, 5), np.random.randn(3, 5, 4, 6)):
        for limit in [0.00001, 1, 10, 10000]:
            x = orig.reshape(orig.shape[0], orig.size / orig.shape[0]).copy()
            divisor = (x * x).sum(axis=1, keepdims=True) ** 0.5 / limit
            divisor[divisor < 1] = 1
            out = (x / divisor).reshape(orig.shape)

            y = orig.copy()
            mod = ConstrainL2Norm(limit)
            mod(default_handler, y)
            assert np.allclose(y, out)

            handler = PyCudaHandler()
            y = handler.create_from_numpy(orig)
            mod(handler, y)
            assert np.allclose(handler.get_numpy_copy(y), out)
