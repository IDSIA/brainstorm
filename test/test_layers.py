#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.structure.buffer_views import BufferView
import numpy as np
import pytest

from brainstorm.initializers import Gaussian
from brainstorm.layers.python_layers import FeedForwardLayer


def setup_buffers(time_steps, num, layer):
    forward_buffer_names = []
    forward_buffer_views = []
    backward_buffer_names = []
    backward_buffer_views = []

    # setup parameters
    param_structure = layer.get_parameter_structure()
    print()
    print("Parameter structure: ", param_structure)
    param_names = []
    forward_param_buffers = []
    backward_param_buffers = []
    for entry in param_structure:
        param_names.append(entry['name'])
        forward_param_buffers.append(np.zeros(entry['shape']))
        backward_param_buffers.append(np.zeros(entry['shape']))

    forward_buffer_names.append('parameters')
    forward_buffer_views.append(BufferView(param_names, forward_param_buffers))
    backward_buffer_names.append('parameters')
    backward_buffer_views.append(BufferView(param_names, backward_param_buffers))

    # setup inputs
    input_names = []
    assert set(layer.input_names) == (layer.in_shapes.keys())
    for key, value in layer.in_shapes.items():

    print("Input names: ", input_names)
    forward_input_buffers = []
    backward_input_buffers = []
    forward_buffer_names.append('inputs')
    forward_buffer_views.append(BufferView(input_names, forward_input_buffers))
    backward_buffer_names.append('inputs')
    backward_buffer_views.append(BufferView(input_names,
                                            backward_input_buffers))

    # setup outputs
    output_names = layer.output_names
    print("Output names: ", output_names)
    forward_output_buffers = []
    backward_output_buffers = []
    forward_buffer_names.append('outputs')
    forward_buffer_views.append(BufferView(output_names,
                                           forward_output_buffers))
    backward_buffer_names.append('outputs')
    backward_buffer_views.append(BufferView(output_names,
                                            backward_output_buffers))

    # setup internals
    internal_names = []
    forward_internal_buffers = []
    backward_internal_buffers = []
    internal_structure = layer.get_internal_structure()
    print(internal_structure)

    forward_buffer_names.append('internal')
    forward_buffer_views.append(BufferView(internal_names,
                                           forward_internal_buffers))
    backward_buffer_names.append('internal')
    backward_buffer_views.append(BufferView(internal_names,
                                            backward_internal_buffers))

    # Finally, setup forward and backward buffers
    forward_buffers = BufferView(forward_buffer_names, forward_buffer_views)
    backward_buffers = BufferView(backward_buffer_names, backward_buffer_views)
    return forward_buffers, backward_buffers


def test_fully_connected_layer():

    time_steps = 2
    num = 1
    in_shapes = {'default': (10,)}
    layer = FeedForwardLayer(in_shapes, [], [], shape=5)
    forward_buffers, backward_buffers = setup_buffers(time_steps, num, layer)
    # layer.forward_pass(forward_buffers)
    # layer.backward_pass(forward_buffers, backward_buffers)