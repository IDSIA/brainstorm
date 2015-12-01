#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import itertools

import numpy as np
import pytest

from brainstorm.handlers import NumpyHandler
from brainstorm.optional import has_pycuda

non_default_handlers = []
handler_ids = []
if has_pycuda:
    from brainstorm.handlers import PyCudaHandler
    non_default_handlers.append(PyCudaHandler())
    handler_ids.append("PyCudaHandler")

# np.random.seed(1234)
ref_dtype = np.float32
ref = NumpyHandler(ref_dtype)
some_2d_shapes = ((1, 1), (4, 1), (1, 4), (5, 5), (3, 4), (4, 3))
some_nd_shapes = ((1, 1, 4), (1, 1, 3, 3), (3, 4, 2, 1))

np.set_printoptions(linewidth=150)


def operation_check(handler, op_name, ref_args, ignored_args=(), atol=1e-8):
    args = get_args_from_ref_args(handler, ref_args)
    getattr(ref, op_name)(*ref_args)
    getattr(handler, op_name)(*args)
    check_list = []
    for i, (ref_arg, arg) in enumerate(zip(ref_args, args)):
        if i in ignored_args:
            # print(i, "was ignored")
            continue
        if type(ref_arg) is ref.array_type:
            arg_ref = handler.get_numpy_copy(arg)
            check = np.allclose(ref_arg, arg_ref, atol=atol)
            check_list.append(check)
            if not check:
                print("-" * 40)
                print("\nCheck failed for argument number %d:" % i)
                print("Reference (expected) array {}:\n{}".format(
                    ref_arg.shape, ref_arg))
                print("\nObtained array {}:\n{}".format(arg_ref.shape,
                                                        arg_ref))
                d = ref_arg.ravel() - arg_ref.ravel()
                print("Frobenius Norm of differences: ", np.sum(d*d))
        else:
            check = (ref_arg == arg)
            check_list.append(check)
            if not check:
                print("-" * 40)
                print("Check failed for argument number %d:" % i)
                print("\nReference (expected) value:\n", ref_arg)
                print("\nObtained value:\n", arg)
                d = ref_arg.ravel() - arg_ref.ravel()
                print("Frobenius Norm of differences: ", np.sum(d*d))
        # print("Check was ", check)
    if False in check_list:
        return False
    else:
        return True


def get_args_from_ref_args(handler, ref_args):
    args = []
    for ref_arg in ref_args:
        if type(ref_arg) is ref.array_type:
            temp = handler.create_from_numpy(ref_arg)
            args.append(temp)
        else:
            args.append(ref_arg)
    return args


def get_random_arrays(shapes=some_2d_shapes, dtype=ref_dtype):
    arrays = []
    for shape in shapes:
        arrays.append(np.random.randn(*shape).astype(dtype))
    return arrays


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_sum_t(handler):
    list_a = get_random_arrays()
    list_axis = [0, 1, None]
    for a, axis in itertools.product(list_a, list_axis):
        if axis == 0:
            out = np.zeros((1, a.shape[1]), dtype=ref_dtype)
        elif axis == 1:
            out = np.zeros((a.shape[0]), dtype=ref_dtype)
        else:
            out = np.array([0.], dtype=ref_dtype).reshape(tuple())
        ref_args = (a, axis, out)

        assert operation_check(handler, 'sum_t', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_dot_mm(handler):
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b.T.copy() for b in list_b]

    for a, b in zip(list_a, list_b):
        out = np.zeros((a.shape[0], a.shape[0]), dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'dot_mm', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_dot_add_mm(handler):
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b.T.copy() for b in list_b]

    for a, b in zip(list_a, list_b):
        out = np.random.randn(a.shape[0], a.shape[0]).astype(ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'dot_add_mm', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_mult_tt(handler):
    list_a = get_random_arrays(some_2d_shapes + some_nd_shapes)
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'mult_tt', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_mult_add_tt(handler):
    list_a = get_random_arrays(some_2d_shapes + some_nd_shapes)
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.random.randn(*a.shape).astype(ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'mult_add_tt', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_mult_st(handler):
    list_a = [0, 0.5, -1]
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(b, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'mult_st', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_mult_add_st(handler):
    list_a = [0, 0.5, -1]
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.random.randn(*b.shape).astype(ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'mult_add_st', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_add_tt(handler):
    list_a = get_random_arrays(some_2d_shapes + some_nd_shapes)
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'add_tt', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_add_st(handler):
    list_a = [0, 0.5, -1]
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(b, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'add_st', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_subtract_tt(handler):
    list_a = get_random_arrays(some_2d_shapes + some_nd_shapes)
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'subtract_tt', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_subtract_mv(handler):
    # Only checking with row vectors
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b[0, :].reshape((1, -1)).copy() for b in list_b]

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'subtract_mv', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_add_mv(handler):
    # Only checking with row vectors
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b[0, :].reshape((1, -1)).copy() for b in list_b]

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'add_mv', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_broadcast_t(handler):
    args_to_check = [
        ([1], 0, [3]),
        ([1], 0, [1]),
        ([1, 2], 0, [3, 2]),
        ([3, 1], 1, [3, 2]),
        ([1, 2, 5], 0, [3, 2, 5]),
        ([3, 1, 5], 1, [3, 2, 5]),
        ([3, 2, 1], 2, [3, 2, 5])
    ]
    a_shapes, axes, out_shapes = list(zip(*args_to_check))

    list_a = get_random_arrays(a_shapes)
    list_out = get_random_arrays(out_shapes)
    for ref_args in zip(list_a, axes, list_out):
        assert operation_check(handler, 'broadcast_t', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_clip_t(handler):
    list_a = get_random_arrays(some_nd_shapes)
    list_clip_min = [-0.4, 0, 0.2]
    list_clip_max = [-0.1, 0, 0.3]

    for a, clip_min, clip_max in itertools.product(list_a, list_clip_min,
                                                   list_clip_max):
        if clip_max >= clip_min:
            out = np.zeros_like(a, dtype=ref_dtype)
            ref_args = (a, clip_min, clip_max, out)
            assert operation_check(handler, 'clip_t', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_log_t(handler):
    list_a = get_random_arrays(some_nd_shapes)

    for a in list_a:
        a += 10  # to remove negatives
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(handler, 'log_t', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_sqrt_t(handler):
    list_a = get_random_arrays(some_nd_shapes)

    for a in list_a:
        a += 10  # to remove negatives
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(handler, 'sqrt_t', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_abs_t(handler):
    list_a = get_random_arrays(some_nd_shapes)

    for a in list_a:
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(handler, 'abs_t', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_sign_t(handler):
    list_a = get_random_arrays(some_nd_shapes)
    list_a += [np.random.random_integers(-2, 2, (3, 3))]
    for a in list_a:
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(handler, 'sign_t', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_divide_tt(handler):
    list_a = get_random_arrays(some_2d_shapes + some_nd_shapes)
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'divide_tt', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_divide_mv(handler):
    # Only checking with row vectors
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b[0, :].reshape((1, -1)).copy() for b in list_b]

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'divide_mv', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_mult_mv(handler):
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b[0, :].reshape((1, -1)).copy() for b in list_b]

    # print("==================================")
    # print("Testing mult_mv() with row vectors")
    # print("==================================")
    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(handler, 'mult_mv', ref_args)

    # print("=====================================")
    # print("Testing mult_mv() with column vectors")
    # print("=====================================")
    list_b = get_random_arrays()
    list_b = [b[:, 0].reshape((-1, 1)).copy() for b in list_b]
    for a, b in zip(list_a, list_b):
        # print('-' * 40)
        # print("a:\n", a)
        # print("b:\n", b)
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)
        assert operation_check(handler, 'mult_mv', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_binarize_v(handler):
    v = np.random.random_integers(0, 4, (10, 1)).astype(ref_dtype)
    out = np.random.random_sample((10, 5))
    ref_args = (v, out)
    assert operation_check(handler, 'binarize_v', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_index_m_by_v(handler):
    m_list = get_random_arrays()
    for m in m_list:
        v = np.random.random_integers(0, m.shape[1] - 1, (m.shape[0], 1))
        out = np.random.random_sample(v.shape)
        ref_args = (m, v, out)
        assert operation_check(handler, 'index_m_by_v', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_sigmoid(handler):
    list_a = get_random_arrays(some_nd_shapes)

    for a in list_a:
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(handler, 'sigmoid', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_sigmoid_deriv(handler):
    list_x = get_random_arrays(some_nd_shapes)
    list_y = get_random_arrays(some_nd_shapes)
    list_dy = get_random_arrays(some_nd_shapes)

    for x, y, dy in zip(list_x, list_y, list_dy):
        dx = np.zeros_like(x, dtype=ref_dtype)
        ref_args = (x, y, dy, dx)
        assert operation_check(handler, 'sigmoid_deriv', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_tanh(handler):
    list_a = get_random_arrays(some_nd_shapes)

    for a in list_a:
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(handler, 'tanh', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_tanh_deriv(handler):
    list_x = get_random_arrays(some_nd_shapes)
    list_y = get_random_arrays(some_nd_shapes)
    list_dy = get_random_arrays(some_nd_shapes)

    for x, y, dy in zip(list_x, list_y, list_dy):
        dx = np.zeros_like(x, dtype=ref_dtype)
        ref_args = (x, y, dy, dx)
        assert operation_check(handler, 'tanh_deriv', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_rel(handler):
    list_a = get_random_arrays(some_nd_shapes)

    for a in list_a:
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(handler, 'rel', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_rel_deriv(handler):
    list_x = get_random_arrays(some_nd_shapes)
    list_y = get_random_arrays(some_nd_shapes)
    list_dy = get_random_arrays(some_nd_shapes)

    for x, y, dy in zip(list_x, list_y, list_dy):
        dx = np.zeros_like(x, dtype=ref_dtype)
        ref_args = (x, y, dy, dx)
        assert operation_check(handler, 'rel_deriv', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_conv2d_forward(handler):
    img_shapes = [(1, 3, 3, 1), (3, 8, 8, 1), (2, 6, 4, 3), (1, 3, 4, 2)]
    w_shapes = [(1, 1, 1), (3, 3, 3), (6, 2, 2), (2, 1, 3)]

    list_x = get_random_arrays(img_shapes)
    stride = (1, 1)
    padding = 1

    for ws in w_shapes:
        for x in list_x:
            w_shape = (ws[0], ws[1], ws[2], x.shape[3])
            w = np.random.uniform(size=w_shape).astype(ref_dtype)
            b = np.random.uniform(size=(w.shape[0],)).astype(ref_dtype)
            oh = (x.shape[1] + 2 * padding - w.shape[1]) / stride[0] + 1
            ow = (x.shape[2] + 2 * padding - w.shape[2]) / stride[1] + 1
            out = np.zeros((x.shape[0], oh, ow, w.shape[0]), dtype=ref_dtype)
            ref_args = (x, w, b, out, padding, stride)

            passed = operation_check(handler, 'conv2d_forward_batch', ref_args,
                                     atol=1e-6)
            if not passed:
                print(x.shape, w.shape)
            assert passed


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_conv2d_backward(handler):
    img_shapes = [(1, 3, 3, 1), (4, 8, 8, 1), (3, 6, 4, 10), (1, 3, 4, 2)]
    w_shapes = [(1, 1, 1), (3, 1, 1), (6, 2, 3), (2, 1, 3)]

    list_x = get_random_arrays(img_shapes)
    stride = (1, 1)
    padding = 1

    for ws in w_shapes:
        for x in list_x:
            w_shape = (ws[0], ws[1], ws[2], x.shape[3])
            w = np.random.uniform(size=w_shape).astype(ref_dtype)
            b = np.random.uniform(size=(w.shape[0],)).astype(ref_dtype)
            oh = (x.shape[1] + 2 * padding - w.shape[1]) / stride[0] + 1
            ow = (x.shape[2] + 2 * padding - w.shape[2]) / stride[1] + 1
            out_shape = (x.shape[0], oh, ow, w.shape[0])
            o_deltas = np.random.uniform(size=out_shape).astype(ref_dtype)
            i_deltas = np.zeros_like(x, dtype=ref_dtype)
            w_deltas = np.zeros_like(w, dtype=ref_dtype)
            b_deltas = np.zeros_like(b, dtype=ref_dtype)

            ref_args = (x, w, padding, stride, i_deltas,
                        o_deltas, w_deltas, b_deltas)
            passed = operation_check(handler, 'conv2d_backward_batch',
                                     ref_args, atol=1e-6)
            if not passed:
                print(x.shape, w.shape)
            assert passed


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_maxpool2d_forward(handler):
    img_shapes = [(1, 5, 5, 1), (1, 8, 8, 3), (3, 6, 4, 2), (1, 6, 9, 2)]
    window_list = [(2, 2), (3, 3), (4, 4), (2, 1), (1, 2)]
    strides_list = [(1, 1), (2, 2), (1, 2), (2, 1)]
    list_x = get_random_arrays(img_shapes)

    for x in list_x:
        for padding in (0, 1, 2):
            for strides in strides_list:
                for window in window_list:
                    out_shape = (
                        x.shape[0],
                        (x.shape[1] + 2*padding - window[0]) // strides[0] + 1,
                        (x.shape[2] + 2*padding - window[1]) // strides[1] + 1,
                        x.shape[3]
                    )
                    outputs = np.zeros(out_shape, dtype=ref_dtype)
                    argmax = np.zeros(out_shape, dtype=ref_dtype)
                    ref_args = (x, window, outputs, padding, strides, argmax)
                    passed = operation_check(handler,
                                             'maxpool2d_forward_batch',
                                             ref_args)
                    if not passed:
                        print(x.shape, window, outputs.shape, padding, strides)
                    assert passed


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_maxpool2d_backward(handler):
    img_shapes = [(1, 5, 5, 1), (1, 8, 8, 3), (3, 6, 4, 2), (1, 6, 9, 2)]
    window_list = [(2, 2), (3, 3), (4, 4), (2, 1), (1, 2)]
    strides_list = [(1, 1), (2, 2), (1, 2), (2, 1)]
    list_x = get_random_arrays(img_shapes)

    for x in list_x:
        for padding in (0, 1, 2):
            for strides in strides_list:
                for window in window_list:
                    out_shape = (
                        x.shape[0],
                        (x.shape[1] + 2*padding - window[0]) // strides[0] + 1,
                        (x.shape[2] + 2*padding - window[1]) // strides[1] + 1,
                        x.shape[3]
                    )
                    outputs = np.zeros(out_shape, dtype=ref_dtype)
                    o_deltas = np.random.normal(size=out_shape)
                    o_deltas = o_deltas.astype(ref_dtype)
                    i_deltas = np.zeros_like(x, dtype=ref_dtype)
                    argmax = np.zeros(out_shape, dtype=ref_dtype)

                    # initialize argmax
                    ref.maxpool2d_forward_batch(x, window, outputs, padding,
                                                strides, argmax)
                    ref_args = (x, window, outputs, padding, strides, argmax,
                                i_deltas, o_deltas)

                    passed = operation_check(handler,
                                             'maxpool2d_backward_batch',
                                             ref_args,
                                             atol=1e-6)
                    if not passed:
                        print(x.shape, window, outputs.shape, padding, strides)
                    assert passed


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_avgpool2d_forward(handler):
    img_shapes = [(1, 5, 5, 1), (10, 32, 32, 3), (10, 6, 4, 10), (1, 6, 9, 2)]
    window_list = [(2, 2), (3, 3), (4, 4), (2, 1), (1, 2)]
    strides_list = [(1, 1), (2, 2), (1, 2), (2, 1)]
    list_x = get_random_arrays(img_shapes)

    for x in list_x:
        for padding in (0, 1, 2):
            for strides in strides_list:
                for window in window_list:
                    out_shape = (
                        x.shape[0],
                        (x.shape[1] + 2*padding - window[0]) // strides[0] + 1,
                        (x.shape[2] + 2*padding - window[1]) // strides[1] + 1,
                        x.shape[3]
                    )
                    outputs = np.zeros(out_shape, dtype=ref_dtype)
                    ref_args = (x, window, outputs, padding, strides)
                    passed = operation_check(handler,
                                             'avgpool2d_forward_batch',
                                             ref_args, atol=1e-6)
                    if not passed:
                        print(x.shape, window, outputs.shape, padding, strides)
                    assert passed


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_avgpool2d_backward(handler):
    img_shapes = [(1, 5, 5, 1), (10, 32, 32, 3), (10, 6, 4, 10), (1, 6, 9, 2)]
    window_list = [(2, 2), (3, 3), (4, 4), (2, 1), (1, 2)]
    strides_list = [(1, 1), (2, 2), (1, 2), (2, 1)]
    list_x = get_random_arrays(img_shapes)

    for x in list_x:
        for padding in (0, 1, 2):
            for strides in strides_list:
                for window in window_list:
                    out_shape = (
                        x.shape[0],
                        (x.shape[1] + 2*padding - window[0]) // strides[0] + 1,
                        (x.shape[2] + 2*padding - window[1]) // strides[1] + 1,
                        x.shape[3]
                    )
                    outputs = np.zeros(out_shape, dtype=ref_dtype)
                    o_deltas = np.random.normal(size=out_shape)
                    o_deltas = o_deltas.astype(ref_dtype)
                    i_deltas = np.zeros_like(x, dtype=ref_dtype)

                    ref.avgpool2d_forward_batch(x, window, outputs, padding,
                                                strides)
                    ref_args = (x, window, outputs, padding, strides,
                                i_deltas, o_deltas)
                    passed = operation_check(handler,
                                             'avgpool2d_backward_batch',
                                             ref_args, atol=1e-6)
                    if not passed:
                        print(x.shape, window, outputs.shape, padding, strides)
                    assert passed


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_softmax_m(handler):
    list_a = get_random_arrays(some_2d_shapes)

    for m in list_a:
        out = np.zeros_like(m, dtype=ref_dtype)
        ref_args = (m, out)
        assert operation_check(handler, 'softmax_m', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_merge_tt(handler):
    shapes = [((5, 4), (5, 3)),
              ((1, 2, 1), (1, 2, 2)),
              ((10, 4, 3), (10, 4, 7)),
              ((1, 2, 3, 4), (1, 2, 3, 5)),
              ((2049, 3), (2049, 1025))]
    for sa, sb in shapes:
        a = np.random.randn(*sa).astype(ref_dtype)
        b = np.random.randn(*sb).astype(ref_dtype)
        sout = list(sa)
        sout[-1] = sa[-1] + sb[-1]
        out = np.zeros(sout, dtype=ref_dtype)
        ref_args = (a, b, out)
        assert operation_check(handler, 'merge_tt', ref_args)


@pytest.mark.parametrize("handler", non_default_handlers, ids=handler_ids)
def test_split_add_tt(handler):
    shapes = [((5, 4), (5, 3)),
              ((1, 2, 1), (1, 2, 2)),
              ((10, 4, 3), (10, 4, 7)),
              ((1, 2, 3, 4), (1, 2, 3, 5)),
              ((2049, 3), (2049, 1025))]
    for sa, sb in shapes:
        a = np.zeros(sa, dtype=ref_dtype)
        b = np.ones(sb, dtype=ref_dtype)
        sx = list(sa)
        sx[-1] = sa[-1] + sb[-1]
        x = np.random.randn(*sx).astype(ref_dtype)
        ref_args = (x, a, b)
        assert operation_check(handler, 'split_add_tt', ref_args)
