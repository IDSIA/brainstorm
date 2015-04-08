#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.structure.buffer_views import BufferView
import numpy as np
np.random.seed(1234)


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
        ei[k] = -epsilon
        f_left = f(*((x0 + ei,) + args))
        grad[k] = (f_right - f_left)/(2 * epsilon)
        ei[k] = 0.0
    return grad


def get_output_error(_h, forward_buffers):
    error = 0.0
    for key in forward_buffers.outputs.keys():
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

    print("Setting up inputs")
    assert set(input_names) == set(layer.in_shapes.keys())
    for name in input_names:
        shape = buffer_shape_from_in_out_shape(time_steps,
                                               batch_size,
                                               layer.in_shapes[name])
        print(name, " : ", shape)
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

    print("Setting up outputs")
    assert set(output_names) == set(layer.out_shapes.keys())
    for name in output_names:
        shape = buffer_shape_from_in_out_shape(time_steps,
                                               batch_size,
                                               layer.out_shapes[name])
        print(name, " : ", shape)
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
    print("Setting up parameters")
    for name, attributes in sorted(param_structure.items(),
                                   key=lambda x: x[1]['@index']):
        param_names.append(name)
        print(name, " : ", attributes['@shape'])
        data = _h.zeros(attributes['@shape'])
        _h.set_from_numpy(data, np.random.randn(*attributes['@shape']))
        forward_param_buffers.append(data)
        backward_param_buffers.append(_h.zeros(attributes['@shape']))

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
    print("Setting up internals")
    for name, attributes in sorted(internal_structure.items(),
                                   key=lambda x: x[1]['@index']):
        print(name, attributes)
        internal_names.append(name)
        shape = buffer_shape_from_in_out_shape(time_steps,
                                               batch_size,
                                               attributes['@shape'])
        print(name, " : ", shape)
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


def run_layer_test(layer, time_steps, batch_size, eps,
                   skip_inputs=(), skip_parameters=(), **inputs):
    """
    Checks the gradients w.r.t. parameters and inputs for a given layer.
    Accepts a named list of initializations for inputs views only.

    :param layer: The $Layer$ object which should be tested.
    :param time_steps: Number of time-steps in each sequence.
    :param batch_size: Number of sequences.
    :param eps: Size of perturbation for analytical gradient computation.
    :param skip_inputs: A list of names of inputs to skip checking.
    :param skip_parameters: A list of names of parameters to skip checking.
    :return:
    """
    _h = layer.handler
    forward_buffers, backward_buffers = setup_buffers(time_steps, batch_size,
                                                      layer)
    for key, value in forward_buffers.inputs.items():
        if key in inputs:
            print("Found:", key)
            _h.set_from_numpy(forward_buffers.inputs[key], inputs[key])

    # First do a forward and backward pass to calculate gradients
    layer.forward_pass(forward_buffers)
    for key in forward_buffers.outputs.keys():
        _h.fill(backward_buffers.outputs[key], 1.0)
    layer.backward_pass(forward_buffers, backward_buffers)

    # Now calculate approximate gradients
    for key in forward_buffers.parameters.keys():
        if key not in skip_parameters:
            print("\nChecking parameter: ", key)
            view = forward_buffers.parameters[key]
            size = _h.size(forward_buffers.parameters[key])
            x0 = _h.get_numpy_copy(view).reshape((size,))
            grad_calc = _h.get_numpy_copy(backward_buffers.parameters[
                key]).reshape((size,))
            print("x0: ", x0)
            print("Expected grad: ", grad_calc)

            def f(x):
                flat_view = _h.reshape(view, (size,))
                _h.set_from_numpy(flat_view, x)  # set to new value
                layer.forward_pass(forward_buffers)
                _h.set_from_numpy(flat_view, x0)  # reset
                return get_output_error(_h, forward_buffers)

            grad_approx = approx_fprime(x0, f, eps)
            print("Approx grad:", grad_approx)
            assert np.allclose(grad_approx, grad_calc, rtol=1e-4, atol=1e-4)

        else:
            print("\nSkipping parameter: ", key)

    for key in forward_buffers.inputs.keys():
        if key not in skip_inputs:
            print("\nChecking input: ", key)
            view = forward_buffers.inputs[key]
            size = _h.size(forward_buffers.inputs[key])
            x0 = _h.get_numpy_copy(view).reshape((size,))
            grad_calc = _h.get_numpy_copy(backward_buffers.inputs[
                key]).reshape((size,))
            print("x0: ", x0)
            print("Expected grad: ", grad_calc)

            def f(x):
                flat_view = _h.reshape(view, (size,))
                _h.set_from_numpy(flat_view, x)  # set to new value
                layer.forward_pass(forward_buffers)
                _h.set_from_numpy(flat_view, x0)  # reset
                return get_output_error(_h, forward_buffers)

            grad_approx = approx_fprime(x0, f, eps)
            print("Approx grad:", grad_approx)
            assert np.allclose(grad_approx, grad_calc, rtol=1e-4, atol=1e-4)

        else:
            print("\nSkipping input: ", key)
