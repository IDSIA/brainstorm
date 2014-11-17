#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
import numpy as np
from brainstorm.randomness import global_rnd
from brainstorm.targets import FramewiseTargets
from brainstorm.structure.network import build_network_from_architecture
from brainstorm.initializers import Gaussian


def approx_fprime(xk, f, epsilon, *args):
    f0 = f(*((xk,)+args))
    grad = np.zeros((len(xk),), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = epsilon
        grad[k] = (f(*((xk+ei,)+args)) - f0)/epsilon
        ei[k] = 0.0
    return grad


@pytest.fixture(scope='module')
def input_data():
    global_rnd.set_seed(1337)
    return global_rnd.randn(7, 5, 3)


@pytest.fixture(scope='module')
def targets():
    global_rnd.set_seed(7331)
    return {
        'test_target': FramewiseTargets(global_rnd.randn(7, 5, 2))
    }

architectures = [
    {'InputLayer': {
        '@type': 'InputLayer',
        'size': 3,
        'sink_layers': ['OutputLayer']},
     'OutputLayer': {
         '@type': 'FeedForwardLayer',
         'size': 2,
         'sink_layers': []}
     }
]


@pytest.fixture(scope='module', params=architectures)
def net(request):
    injectors = {
        'MSE': {
            '@type': 'MeanSquaredError',
            'layer': 'OutputLayer',
            'target_from': 'test_target'
        }
    }
    n = build_network_from_architecture(request.param, injectors)
    n.initialize(Gaussian(), seed=235)
    return n


def test_deltas_finite_differences(net, input_data, targets):
    # ######## calculate deltas ##########
    net.forward_pass(input_data)
    delta_calc = net.backward_pass(targets).flatten()

    # ######## estimate deltas ##########
    def f(x):
        net.forward_pass(x.reshape(input_data.shape))
        return net.calculate_errors(targets)['MSE']
    delta_approx = approx_fprime(input_data.copy().flatten(), f, 1e-7)

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


def test_gradient_finite_differences(net, input_data, targets):
    # ######## calculate deltas ##########
    net.forward_pass(input_data)
    net.backward_pass(targets)
    gradient_calc = net.buffer.gradient.get_raw()

    # ######## estimate deltas ##########
    def f(x):
        net.buffer.parameters.get_raw()[:] = x
        net.forward_pass(input_data)
        return net.calculate_errors(targets)['MSE']
    initial_weigths = net.buffer.parameters.get_raw().copy()
    gradient_approx = approx_fprime(initial_weigths, f, 1e-7)

    # ######## compare them #############
    nr_sequences = input_data.shape[1]
    diff = gradient_approx - gradient_calc
    mse = np.sum(diff ** 2) / nr_sequences
    if mse > 1e-4:
        # Hijack the network gradient buffer for the view
        net.buffer.gradient.get_raw()[:] = diff
        for layer_name in net.buffer.gradient:
            if net.buffer.gradient[layer_name] is None:
                continue
            print("============= Layer: {} =============".format(layer_name))
            for view_name in net.buffer.gradient[layer_name]:
                print("------------- {} -------------".format(view_name))
                print(net.buffer.gradient[layer_name][view_name])

    print(">> Checking Gradient = %0.4f" % mse)
    assert mse < 1e-4
