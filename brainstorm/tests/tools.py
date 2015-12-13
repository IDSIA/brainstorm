#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from brainstorm.tests.test_layers import (
    spec_list, test_deltas_calculation_of_layer, test_layer_add_to_deltas,
    test_layer_backward_pass_insensitive_to_internal_state_init,
    test_layer_forward_pass_insensitive_to_internal_state_init,
    test_gradients_for_layer)


def get_test_configurations():
    for spec in spec_list:
        time_steps, batch_size, activation = spec
        yield {
            'time_steps': time_steps,
            'batch_size': batch_size,
            'activation': activation
        }


def run_layer_tests(layer, spec):
    spec_str = "time_steps={time_steps}, batch_size={batch_size}," \
               " activation={activation}".format(**spec)
    print('======= Testing {} for {} ====='.format(layer.name, spec_str))
    print('Testing Delta Calculations ...')
    test_deltas_calculation_of_layer((layer, spec))
    print('Testing Gradient Calculations ...')
    test_gradients_for_layer((layer, spec))
    print('Verifying that layer ADDS to deltas ...')
    test_layer_add_to_deltas((layer, spec))
    print('Verifying that the forward pass is insensitive to initialization of'
          ' internals ...')
    test_layer_forward_pass_insensitive_to_internal_state_init((layer, spec))
    print('Verifying that the backward pass is insensitive to initialization'
          ' of internals ...')
    test_layer_backward_pass_insensitive_to_internal_state_init((layer, spec))
    print("")
