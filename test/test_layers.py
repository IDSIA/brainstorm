#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.structure.buffer_views import BufferView
import pytest

from brainstorm.initializers import Gaussian
from brainstorm.layers.python_layers import FeedForwardLayer
from brainstorm.handlers import default_handler


def setup_buffers(time_steps, num, layer):
    H = layer.handler
    forward_buffer_names = []
    forward_buffer_views = []
    backward_buffer_names = []
    backward_buffer_views = []

    # setup inputs
    input_names = layer.input_names
    forward_input_buffers = []
    backward_input_buffers = []

    print("Input names: ", input_names)
    assert set(input_names) == set(layer.in_shapes.keys())
    for name in input_names:
        shape = layer.in_shapes[name]
        forward_input_buffers.append(H.zeros((time_steps, num) + shape))
        backward_input_buffers.append(H.zeros((time_steps, num) + shape))

    forward_buffer_names.append('inputs')
    forward_buffer_views.append(BufferView(input_names, forward_input_buffers))
    backward_buffer_names.append('inputs')
    backward_buffer_views.append(BufferView(input_names,
                                            backward_input_buffers))

    # setup outputs
    output_names = layer.output_names
    forward_output_buffers = []
    backward_output_buffers = []

    print("Output names: ", output_names)
    assert set(output_names) == set(layer.in_shapes.keys())
    for name in output_names:
        shape = layer.out_shapes[name]
        forward_output_buffers.append(H.zeros((time_steps, num) + shape))
        backward_output_buffers.append(H.zeros((time_steps, num) + shape))

    forward_buffer_names.append('outputs')
    forward_buffer_views.append(BufferView(output_names,
                                           forward_output_buffers))
    backward_buffer_names.append('outputs')
    backward_buffer_views.append(BufferView(output_names,
                                            backward_output_buffers))

    # setup parameters
    param_names = []
    forward_param_buffers = []
    backward_param_buffers = []

    param_structure = layer.get_parameter_structure()
    print()
    print("Parameter structure: ", param_structure)
    for name, attributes in sorted(param_structure.items(),
                                   key=lambda x: x[1]['index']):
        print(name, attributes)
        param_names.append(name)
        forward_param_buffers.append(H.zeros(attributes['shape']))
        backward_param_buffers.append(H.zeros(attributes['shape']))

    forward_buffer_names.append('parameters')
    forward_buffer_views.append(BufferView(param_names, forward_param_buffers))
    backward_buffer_names.append('parameters')
    backward_buffer_views.append(BufferView(param_names, backward_param_buffers))

    # setup internals
    internal_names = []
    forward_internal_buffers = []
    backward_internal_buffers = []
    internal_structure = layer.get_internal_structure()
    print("Internal structure: ", internal_structure)
    for name, attributes in sorted(internal_structure.items(),
                                   key=lambda x: x[1]['index']):
        print(name, attributes)
        internal_names.append(name)
        forward_internal_buffers.append(H.zeros((time_steps, num) +
                                                attributes['shape']))
        backward_internal_buffers.append(H.zeros((time_steps, num) +
                                                 attributes['shape']))

    forward_buffer_names.append('internals')
    forward_buffer_views.append(BufferView(internal_names,
                                           forward_internal_buffers))
    backward_buffer_names.append('internals')
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
    layer.set_handler(default_handler)
    forward_buffers, backward_buffers = setup_buffers(time_steps, num, layer)
    layer.forward_pass(forward_buffers)
    layer.backward_pass(forward_buffers, backward_buffers)