#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.structure.buffer_views import BufferView
from brainstorm.handlers import NumpyHandler
import numpy as np

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


def get_output_error(_h, forward_buffers, skip_outputs=()):
    error = 0.0
    for key in forward_buffers.outputs.keys():
        if key in skip_outputs:
            continue
        value = _h.get_numpy_copy(forward_buffers.outputs[key])
        error += value.sum()
    return error


def buffer_shape_from_in_out_shape(time_steps, batch_size, shape):
    """
    Computes the size of the buffers for inputs and outputs given the shape
    representation by the layer such as ('T', 'B', 3).

    :param shape: A shape tuple, whose first entry might be 'T' and second
    entry might be 'B', but rest must be positive integers.
    :return: A shape tuple of same length as inputs, but consisting of
    positive integers.
    """
    return (time_steps, batch_size) + shape[2:]


def setup_buffers(time_steps, batch_size, layer):
    """
    Sets up the required forward and backward buffers for gradient checking.

    This function will also randomly initialize the parameters and inputs.

    :param time_steps: Number of time-steps in each sequence.
    :param batch_size: Number of sequences.
    :param layer: A $Layer$ object.
    :return: BufferViews for forward and backward buffers
    """
    _h = layer.handler
    forward_buffer_names = []
    forward_buffer_views = []
    backward_buffer_names = []
    backward_buffer_views = []

    # setup inputs
    input_names = layer.inputs
    forward_input_buffers = []
    backward_input_buffers = []

    assert set(input_names) == set(layer.in_shapes.keys())
    for name in input_names:
        shape = layer.in_shapes[name].get_shape(time_steps, batch_size)
        data = _h.zeros(shape)
        _h.set_from_numpy(data, np.random.randn(*shape))
        forward_input_buffers.append(data)
        backward_input_buffers.append(_h.zeros(shape))

    forward_buffer_names.append('inputs')
    forward_buffer_views.append(BufferView(input_names, forward_input_buffers))
    backward_buffer_names.append('inputs')
    backward_buffer_views.append(BufferView(input_names,
                                            backward_input_buffers))

    # setup outputs
    output_names = layer.outputs
    forward_output_buffers = []
    backward_output_buffers = []

    assert set(output_names) == set(layer.out_shapes.keys())
    for name in output_names:
        shape = layer.out_shapes[name].get_shape(time_steps, batch_size)
        forward_output_buffers.append(_h.zeros(shape))
        backward_output_buffers.append(_h.zeros(shape))

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
    for name, shape_template in param_structure.items():
        param_names.append(name)
        shape = shape_template.get_shape(1, 1)
        data = _h.zeros(shape)
        _h.set_from_numpy(data, np.random.randn(*shape))
        forward_param_buffers.append(data)
        backward_param_buffers.append(_h.zeros(shape))

    forward_buffer_names.append('parameters')
    forward_buffer_views.append(BufferView(param_names, forward_param_buffers))
    backward_buffer_names.append('parameters')
    backward_buffer_views.append(BufferView(param_names,
                                            backward_param_buffers))

    # setup internals
    internal_names = []
    forward_internal_buffers = []
    backward_internal_buffers = []

    internal_structure = layer.get_internal_structure()
    for name, shape_template in internal_structure.items():
        internal_names.append(name)
        shape = shape_template.get_shape(time_steps, batch_size)
        forward_internal_buffers.append(_h.zeros(shape))
        backward_internal_buffers.append(_h.zeros(shape))

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


def set_up_layer(layer, specs):
    layer.set_handler(HANDLER)
    time_steps = specs.get('time_steps', 3)
    batch_size = specs.get('batch_size', 2)

    fwd_buffers, bwd_buffers = setup_buffers(time_steps, batch_size, layer)

    for key, value in fwd_buffers.inputs.items():
        if key in specs:
            # print("Using special input:", key)
            HANDLER.set_from_numpy(fwd_buffers.inputs[key], specs[key])

    return fwd_buffers, bwd_buffers


def run_deltas_test(layer, specs, inputs_name, outputs_name):
    eps = specs.get('eps', 1e-5)
    print("Checking input '{}' ...".format(inputs_name))
    fwd_buffers, bwd_buffers = set_up_layer(layer, specs)
    # First do a forward and backward pass to calculate gradients
    layer.forward_pass(fwd_buffers)
    HANDLER.fill(bwd_buffers.outputs[outputs_name], 1.0)
    layer.backward_pass(fwd_buffers, bwd_buffers)
    delta_calc = bwd_buffers.inputs[inputs_name]
    delta_approx = get_approx_deltas(layer, inputs_name,
                                     outputs_name, fwd_buffers,
                                     eps).reshape(delta_calc.shape)
    if np.allclose(delta_approx, delta_calc, rtol=1e-4, atol=1e-4):
        return True

    print("Deltas check for '{}' WRT '{}' failed with a MSE of {}"
          .format(inputs_name, outputs_name,
                  np.sqrt(np.sum((delta_calc - delta_approx)**2))))
    for t in range(delta_calc.shape[0]):
        print(".......... Timestep {} ..........".format(t))
        print("Calculated Deltas:\n", delta_calc[t])
        print("Approx Deltas:\n", delta_approx[t])
        print("Difference:\n", delta_calc[t] - delta_approx[t])

    return False


def get_approx_deltas(layer, inputs_name, outputs_name, forward_buffers, eps):
    """
    Approximates the derivative of one layer input with respect to some outputs

    :param layer: The layer whose derivative should be approximated
    :param inputs_name: The input for which to approximate the derivative
    :param outputs_name: The output wrt. to which to approximate the derivative
    :param forward_buffers: Forward buffers view for the layer
    :param eps: Size of perturbation for numerical gradient computation
    """
    _h = layer.handler

    view = forward_buffers.inputs[inputs_name]
    size = _h.size(view)
    x0 = _h.get_numpy_copy(view).reshape((size,))

    def f(x):
        flat_view = _h.reshape(view, (size,))
        _h.set_from_numpy(flat_view, x)  # set to new value
        layer.forward_pass(forward_buffers)
        return _h.get_numpy_copy(forward_buffers.outputs[outputs_name]).sum()

    return approx_fprime(x0, f, eps)


def get_approx_gradients(layer, parameter_name, outputs_name, forward_buffers,
                         eps):
    """
    Approximates the derivative of one layer parameter with respect to
    some outputs.

    :param layer: The layer whose derivative should be approximated
    :param parameter_name: The parameters for which to approximate the
                           derivative
    :param outputs_name: The output wrt. to which to approximate the derivative
    :param forward_buffers: Forward buffers view for the layer
    :param eps: Size of perturbation for numerical gradient computation
    """
    _h = layer.handler

    view = forward_buffers.parameters[parameter_name]
    size = _h.size(view)
    x0 = _h.get_numpy_copy(view).reshape((size,))

    def f(x):
        flat_view = _h.reshape(view, (size,))
        _h.set_from_numpy(flat_view, x)  # set to new value
        layer.forward_pass(forward_buffers)
        return _h.get_numpy_copy(forward_buffers.outputs[outputs_name]).sum()

    return approx_fprime(x0, f, eps)


def run_gradients_test(layer, specs, parameter_name, outputs_name):
    eps = specs.get('eps', 1e-5)
    print("Checking parameter '{}' ...".format(parameter_name))
    fwd_buffers, bwd_buffers = set_up_layer(layer, specs)
    # First do a forward and backward pass to calculate gradients
    layer.forward_pass(fwd_buffers)
    HANDLER.fill(bwd_buffers.outputs[outputs_name], 1.0)
    layer.backward_pass(fwd_buffers, bwd_buffers)
    grad_calc = bwd_buffers.parameters[parameter_name]
    grad_approx = get_approx_gradients(layer, parameter_name,
                                       outputs_name, fwd_buffers,
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

