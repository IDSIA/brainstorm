#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
import numpy as np
from brainstorm.handlers import NumpyHandler
from brainstorm.randomness import global_rnd
from brainstorm.structure.network import build_network_from_architecture
from brainstorm.initializers import Gaussian
from ..helpers import approx_fprime



@pytest.fixture(scope='module')
def input_data():
    global_rnd.set_seed(1337)
    return global_rnd.randn(7, 5, 3)


@pytest.fixture(scope='module')
def targets():
    global_rnd.set_seed(7331)
    return global_rnd.randn(7, 5, 2)

architectures = [{
    'InputLayer': {
        '@type': 'InputLayer',
        'out_shapes': {'default': ('T', 'B', 3,)},
        '@outgoing_connections': {
            'default': ['OutputLayer'],
        }},
    'OutputLayer': {
        '@type': 'FullyConnectedLayer',
        'shape': 2,
        '@outgoing_connections': {
            'default': ['LossLayer']
        }},
    'LossLayer': {
        '@type': 'LossLayer',
        '@outgoing_connections': {}
    }}
]


@pytest.fixture(scope='module', params=architectures)
def net(request):
    n = build_network_from_architecture(request.param)
    n.set_memory_handler(NumpyHandler(dtype=np.float64))
    n.initialize(Gaussian(1), seed=235)
    return n


def test_deltas_finite_differences(net, input_data):
    # ######## calculate deltas ##########
    net.provide_external_data({'default': input_data})
    net.forward_pass(training_pass=True)
    net.backward_pass()
    delta_calc = net.buffer.backward.InputLayer.outputs.default.flatten()

    # ######## estimate deltas ##########
    def f(x):
        net.provide_external_data({'default': x.reshape(input_data.shape)})
        net.forward_pass()
        return net.get_loss_value()
    delta_approx = approx_fprime(input_data.copy().flatten(), f, 1e-5)

    # ######## compare them #############
    nr_sequences = input_data.shape[1]
    mse = np.sum((delta_approx - delta_calc) ** 2) / nr_sequences
    if mse > 1e-4:
        diff = (delta_approx - delta_calc).reshape(input_data.shape)
        for t in range(diff.shape[0]):
            print("======== t=%d =========" % t)
            print(diff[t])
    print("Checking Deltas = %0.4f" % mse)

    assert mse < 1e-4


def test_gradient_finite_differences(net, input_data):
    # ######## calculate deltas ##########
    net.provide_external_data({'default': input_data})
    net.forward_pass(training_pass=True)
    net.backward_pass()
    gradient_calc = net.buffer.backward.parameters

    # ######## estimate deltas ##########
    def f(x):
        net.buffer.forward.parameters[:] = x
        net.forward_pass()
        return net.get_loss_value()
    initial_weigths = net.buffer.forward.parameters.copy()
    gradient_approx = approx_fprime(initial_weigths, f, 1e-6)

    # ######## compare them #############
    nr_sequences = input_data.shape[1]
    diff = gradient_approx - gradient_calc
    mse = np.sum(diff ** 2) / nr_sequences
    if mse > 1e-4:
        # Hijack the network gradient buffer for the view
        net.buffer.gradients[:] = diff
        for layer_name in net.layers:
            if not net.buffer.backward[layer_name]:
                continue
            print("============= Layer: {} =============".format(layer_name))
            for view_name in net.buffer.backward[layer_name].parameters.keys():
                print("------------- {} -------------".format(view_name))
                print(net.buffer.backward[layer_name].parameters[view_name])

    print(">> Checking Gradient = %0.4f" % mse)
    assert mse < 1e-4
