#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.handlers import NumpyHandler
import numpy as np
from brainstorm.structure.buffers import get_total_size_slices_and_shapes, \
    create_buffer_views_from_layout
from brainstorm.structure.layout import create_layout

np.random.seed(1234)
HANDLER = NumpyHandler(np.float64)


def approx_fprime(x0, f, epsilon, *args):
    """
    Calculates the 2-sided numerical gradient of $f$.

    :param x0: A 1-D array which specifies the point at which gradient is
    computed.
    :param f: A function whose gradient will be computed. Must return a
    scalar value.
    :param epsilon: Perturbation value for numerical gradient calculation.
    :param args: Any arguments which need to be passed on to $f$.
    :return: Numerically computed gradient of same shape as $x0$.
    """
    grad = np.zeros((len(x0),), float)
    ei = np.zeros((len(x0),), float)
    for k in range(len(x0)):
        ei[k] = epsilon
        f_right = f(*((x0 + ei,) + args))
        f_left = f(*((x0 - ei,) + args))
        grad[k] = (f_right - f_left)/(2 * epsilon)
        ei[k] = 0.0
    return grad


def set_up_layer(layer, specs):
    layer.set_handler(HANDLER)
    time_steps = specs.get('time_steps', 3)
    batch_size = specs.get('batch_size', 2)
    hubs, layout = create_layout({'test_layer': layer})
    total_size, slices, shapes = get_total_size_slices_and_shapes(
        hubs, time_steps, batch_size)
    full_buffer = HANDLER.allocate(total_size)
    buffers = [full_buffer[slice_].reshape(shape)
               for slice_, shape in zip(slices, shapes)]
    view = create_buffer_views_from_layout(layout, buffers, hubs)

    layer_buffers = view.test_layer

    # init parameters randomly
    HANDLER.set_from_numpy(view.parameters,
                           np.random.randn(len(view.parameters)) * 0.1)

    for key, value in view.test_layer.inputs.items():
        if key in specs:  # if a special input is given use that
            # print("Using special input:", key)
            HANDLER.set_from_numpy(layer_buffers.inputs[key], specs[key])
        else:  # otherwise randomize the input
            HANDLER.set_from_numpy(
                layer_buffers.inputs[key],
                np.random.randn(*layer_buffers.inputs[key].shape))

    return layer_buffers


def run_deltas_test(layer, specs, inputs_name, outputs_name):
    eps = specs.get('eps', 1e-5)
    print("Checking input '{}' ...".format(inputs_name))
    layer_buffers = set_up_layer(layer, specs)
    # First do a forward and backward pass to calculate gradients
    layer.forward_pass(layer_buffers)
    HANDLER.fill(layer_buffers.output_deltas[outputs_name], 1.0)
    layer.backward_pass(layer_buffers)
    delta_calc = layer_buffers.input_deltas[inputs_name]
    delta_approx = get_approx_deltas(layer, inputs_name,
                                     outputs_name, layer_buffers,
                                     eps).reshape(delta_calc.shape)
    if np.allclose(delta_approx, delta_calc, rtol=1e-4, atol=1e-4):
        return True

    print("Deltas check for '{}' WRT '{}' failed with a MSE of {}"
          .format(inputs_name, outputs_name,
                  np.sqrt(np.sum((delta_calc - delta_approx)**2))))
    if layer.in_shapes[inputs_name].scales_with_time:
        for t in range(delta_calc.shape[0]):
            print(".......... Timestep {} ..........".format(t))
            print("Calculated Deltas:\n", delta_calc[t])
            print("Approx Deltas:\n", delta_approx[t])
            print("Difference:\n", delta_calc[t] - delta_approx[t])
    else:
        print("Calculated Deltas:\n", delta_calc)
        print("Approx Deltas:\n", delta_approx)
        print("Difference:\n", delta_calc - delta_approx)

    return False


def get_approx_deltas(layer, inputs_name, outputs_name, layer_buffers, eps):
    """
    Approximates the derivative of one layer input with respect to some outputs

    :param layer: The layer whose derivative should be approximated
    :param inputs_name: The input for which to approximate the derivative
    :param outputs_name: The output wrt. to which to approximate the derivative
    :param layer_buffers: Buffers view for the layer
    :param eps: Size of perturbation for numerical gradient computation
    """
    _h = layer.handler

    view = layer_buffers.inputs[inputs_name]
    size = _h.size(view)
    x0 = _h.get_numpy_copy(view).reshape((size,))

    def f(x):
        flat_view = _h.reshape(view, (size,))
        _h.set_from_numpy(flat_view, x)  # set to new value
        layer.forward_pass(layer_buffers)
        return _h.get_numpy_copy(layer_buffers.outputs[outputs_name]).sum()

    return approx_fprime(x0, f, eps)


def get_approx_gradients(layer, parameter_name, outputs_name, layer_buffers,
                         eps):
    """
    Approximates the derivative of one layer parameter with respect to
    some outputs.

    :param layer: The layer whose derivative should be approximated
    :param parameter_name: The parameters for which to approximate the
                           derivative
    :param outputs_name: The output wrt. to which to approximate the derivative
    :param layer_buffers: Forward buffers view for the layer
    :param eps: Size of perturbation for numerical gradient computation
    """
    _h = layer.handler

    view = layer_buffers.parameters[parameter_name]
    size = _h.size(view)
    x0 = _h.get_numpy_copy(view).reshape((size,))

    def f(x):
        flat_view = _h.reshape(view, (size,))
        _h.set_from_numpy(flat_view, x)  # set to new value
        layer.forward_pass(layer_buffers)
        return _h.get_numpy_copy(layer_buffers.outputs[outputs_name]).sum()

    return approx_fprime(x0, f, eps)


def run_gradients_test(layer, specs, parameter_name, outputs_name):
    eps = specs.get('eps', 1e-5)
    print("Checking parameter '{}' ...".format(parameter_name))
    layer_buffers = set_up_layer(layer, specs)
    print("Shape of parameter is {}".
          format(layer_buffers.parameters[parameter_name].shape))
    # First do a forward and backward pass to calculate gradients
    layer.forward_pass(layer_buffers)
    HANDLER.fill(layer_buffers.output_deltas[outputs_name], 1.0)
    layer.backward_pass(layer_buffers)
    grad_calc = layer_buffers.gradients[parameter_name]
    grad_approx = get_approx_gradients(layer, parameter_name,
                                       outputs_name, layer_buffers,
                                       eps).reshape(grad_calc.shape)
    if np.allclose(grad_approx, grad_calc, rtol=1e-4, atol=1e-4):
        return True

    print("Gradient check for '{}' WRT '{}' failed with a MSE of {}"
          .format(parameter_name, outputs_name,
                  np.sqrt(np.sum((grad_calc - grad_approx)**2))))
    print("Calculated Deltas:\n", grad_calc)
    print("Approx Deltas:\n", grad_approx)
    print("Difference:\n", grad_calc - grad_approx)

    return False

