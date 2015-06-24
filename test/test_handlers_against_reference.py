#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.handlers import NumpyHandler, PyCudaHandler

# np.random.seed(1234)
ref = NumpyHandler(np.float32)
handler = PyCudaHandler()

def operation_check(ref_op, op, ref_args, args):
    ref_op(*ref_args)
    op(*args)
    for (ref_arg, arg) in zip(ref_args, args):
        if type(ref_arg) is ref.array_type:
            arg_ref = handler.get_numpy_copy(arg)
            print("ref: ", ref_arg)
            print("arg: ", arg)
            return np.allclose(ref_arg, arg_ref)
        else:
            return True

def test_sum_t():
    a = np.random.rand(3, 4, 5)
    a = a.astype(np.float32)
    axis = 0
    out = np.zeros((1, 4, 5), dtype=np.float32)
    ref_args = (a, axis, out)

    args = []
    for ref_arg in ref_args:
        if type(ref_arg) is ref.array_type:
            temp = handler.create_from_numpy(ref_arg)
            print(temp)
            args.append(temp)
        else:
            args.append(ref_arg)
    assert operation_check(ref.sum_t, handler.sum_t, ref_args, args)
