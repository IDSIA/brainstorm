#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.layers import FullyConnectedLayerImpl
from brainstorm.handlers import NumpyHandler
from brainstorm.structure.shapes import ShapeTemplate
from test.helpers import setup_buffers


def test_fully_connected_layer_overwrites_internal():
    in_shapes = {'default': ShapeTemplate('T', 'B', 3)}

    layer = FullyConnectedLayerImpl('TestLayer', in_shapes, [], [])
    layer.set_handler(NumpyHandler(np.float64))

    forward_buffers, backward_buffers = setup_buffers(3, 2, layer)

    layer.forward_pass(forward_buffers)
    out1 = forward_buffers.outputs.default.copy()

    for v in forward_buffers.internals:
        v[:] = 1

    layer.forward_pass(forward_buffers)
    out2 = forward_buffers.outputs.default.copy()

    np.testing.assert_allclose(out1, out2)
