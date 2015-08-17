#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import itertools
import numpy as np
from brainstorm.handlers import NumpyHandler, PyCudaHandler

# np.random.seed(1234)
ref_dtype = np.float32
ref = NumpyHandler(np.float32)
handler = PyCudaHandler()
some_2d_shapes = [[1, 1], (5, 5), [3, 4]]
some_nd_shapes = [[1, 1, 4], [1, 1, 3, 3], [3, 4, 2, 1]]


def operation_check(ref_op, op, ref_args, args):
    print("--" * 80)
    ref_op(*ref_args)
    op(*args)
    check = True
    for (ref_arg, arg) in zip(ref_args, args):
        if type(ref_arg) is ref.array_type:
            arg_ref = handler.get_numpy_copy(arg)
            print("\narray ref:\n", ref_arg)
            print("\narray arg:\n", arg)
            check = np.allclose(ref_arg, arg_ref)
        else:
            print("\nref:\n", ref_arg)
            print("\narg:\n", arg)
            check = (ref_arg == arg)
    return check


def get_args_from_ref_args(ref_args):
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


def test_sum_t():
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

        assert operation_check(ref.sum_t, handler.sum_t, ref_args,
                               get_args_from_ref_args(ref_args))


def test_dot_mm():
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b.T.copy() for b in list_b]

    for a, b in zip(list_a, list_b):
        out = np.zeros((a.shape[0], a.shape[0]), dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(ref.dot_mm, handler.dot_mm, ref_args,
                               get_args_from_ref_args(ref_args))


def test_add_dot_mm():
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b.T.copy() for b in list_b]

    for a, b in zip(list_a, list_b):
        out = np.random.randn(a.shape[0], a.shape[0]).astype(np.float32)
        ref_args = (a, b, out)

        assert operation_check(ref.dot_mm, handler.dot_mm, ref_args,
                               get_args_from_ref_args(ref_args))


def test_mult_tt():
    list_a = get_random_arrays(some_2d_shapes + some_nd_shapes)
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(ref.mult_tt, handler.mult_tt, ref_args,
                               get_args_from_ref_args(ref_args))


def test_mult_add_tt():
    list_a = get_random_arrays(some_2d_shapes + some_nd_shapes)
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.random.randn(*a.shape).astype(np.float32)
        ref_args = (a, b, out)

        assert operation_check(ref.mult_add_tt, handler.mult_add_tt, ref_args,
                               get_args_from_ref_args(ref_args))


def test_mult_st():
    list_a = [0, 0.5, -1]
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(b, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(ref.mult_st, handler.mult_st, ref_args,
                               get_args_from_ref_args(ref_args))


def test_add_tt():
    list_a = get_random_arrays(some_2d_shapes + some_nd_shapes)
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(ref.add_tt, handler.add_tt, ref_args,
                               get_args_from_ref_args(ref_args))


def test_add_st():
    list_a = [0, 0.5, -1]
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(b, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(ref.add_st, handler.add_st, ref_args,
                               get_args_from_ref_args(ref_args))


def test_subtract_tt():
    list_a = get_random_arrays(some_2d_shapes + some_nd_shapes)
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(ref.subtract_tt, handler.subtract_tt, ref_args,
                               get_args_from_ref_args(ref_args))


def test_add_mv():
    # Only checking with row vectors
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b[0, :].reshape((1, -1)).copy() for b in list_b]

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(ref.add_mv, handler.add_mv, ref_args,
                               get_args_from_ref_args(ref_args))


def test_broadcast_features_t():
    shapes_to_check = [[1, 1, 1], [1, 2, 1], [3, 2, 1], [4, 1, 1]]

    list_a = get_random_arrays(shapes_to_check)
    shapes_to_add = [(1,), (2, 2), (3, 1, 1)]

    for a, shape_to_add in itertools.product(list_a, shapes_to_add):
        out = np.zeros(a.shape + shape_to_add, dtype=ref_dtype)
        ref_args = (a, out)

        assert operation_check(ref.broadcast_features_t,
                               handler.broadcast_features_t, ref_args,
                               get_args_from_ref_args(ref_args))


def test_clip_t():
    list_a = get_random_arrays(some_nd_shapes)
    list_clip_min = [-0.4, 0, 0.2]
    list_clip_max = [-0.1, 0, 0.3]

    for a, clip_min, clip_max in itertools.product(list_a, list_clip_min,
                                                   list_clip_max):
        if clip_max >= clip_min:
            out = np.zeros_like(a, dtype=ref_dtype)
            ref_args = (a, clip_min, clip_max, out)
            assert operation_check(ref.clip_t, handler.clip_t, ref_args,
                                   get_args_from_ref_args(ref_args))


def test_log_t():
    list_a = get_random_arrays(some_nd_shapes)

    for a in list_a:
        a += 10  # to remove negatives
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(ref.log_t, handler.log_t, ref_args,
                               get_args_from_ref_args(ref_args))


def test_divide_tt():
    list_a = get_random_arrays(some_2d_shapes + some_nd_shapes)
    list_b = get_random_arrays(some_2d_shapes + some_nd_shapes)

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(ref.divide_tt, handler.divide_tt, ref_args,
                               get_args_from_ref_args(ref_args))


def test_divide_mv():
    # Only checking with row vectors
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b[0, :].reshape((1, -1)).copy() for b in list_b]

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(ref.divide_mv, handler.divide_mv, ref_args,
                               get_args_from_ref_args(ref_args))


def test_mult_mv():
    # Only checking with row vectors
    list_a = get_random_arrays()
    list_b = get_random_arrays()
    list_b = [b[0, :].reshape((1, -1)).copy() for b in list_b]

    for a, b in zip(list_a, list_b):
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, b, out)

        assert operation_check(ref.mult_mv, handler.mult_mv, ref_args,
                               get_args_from_ref_args(ref_args))


def test_binarize_v(): # TODO
    pass


def test_index_m_by_v(): # TODO
    pass


def test_sigmoid():
    list_a = get_random_arrays(some_nd_shapes)

    for a in list_a:
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(ref.sigmoid, handler.sigmoid, ref_args,
                               get_args_from_ref_args(ref_args))


def test_sigmoid_deriv():
    list_x = get_random_arrays(some_nd_shapes)
    list_y = get_random_arrays(some_nd_shapes)
    list_dy = get_random_arrays(some_nd_shapes)

    for x, y, dy in zip(list_x, list_y, list_dy):
        dx = np.zeros_like(x, dtype=ref_dtype)
        ref_args = (x, y, dy, dx)
        assert operation_check(ref.sigmoid_deriv, handler.sigmoid_deriv,
                               ref_args, get_args_from_ref_args(ref_args))


def test_tanh():
    list_a = get_random_arrays(some_nd_shapes)

    for a in list_a:
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(ref.tanh, handler.tanh, ref_args,
                               get_args_from_ref_args(ref_args))


def test_tanh_deriv():
    list_x = get_random_arrays(some_nd_shapes)
    list_y = get_random_arrays(some_nd_shapes)
    list_dy = get_random_arrays(some_nd_shapes)

    for x, y, dy in zip(list_x, list_y, list_dy):
        dx = np.zeros_like(x, dtype=ref_dtype)
        ref_args = (x, y, dy, dx)
        assert operation_check(ref.tanh_deriv, handler.tanh_deriv,
                               ref_args, get_args_from_ref_args(ref_args))


def test_rel():
    list_a = get_random_arrays(some_nd_shapes)

    for a in list_a:
        out = np.zeros_like(a, dtype=ref_dtype)
        ref_args = (a, out)
        assert operation_check(ref.rel, handler.rel, ref_args,
                               get_args_from_ref_args(ref_args))


def test_rel_deriv():
    list_x = get_random_arrays(some_nd_shapes)
    list_y = get_random_arrays(some_nd_shapes)
    list_dy = get_random_arrays(some_nd_shapes)

    for x, y, dy in zip(list_x, list_y, list_dy):
        dx = np.zeros_like(x, dtype=ref_dtype)
        ref_args = (x, y, dy, dx)
        assert operation_check(ref.rel_deriv, handler.rel_deriv,
                               ref_args, get_args_from_ref_args(ref_args))
