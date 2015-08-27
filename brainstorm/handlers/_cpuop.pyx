# coding=utf-8
from __future__ import division, print_function
import cython
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np

#ctypedef np.float32_t DTYPE_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

cdef inline DTYPE_t dtype_t_max(DTYPE_t a, DTYPE_t b) nogil:
    return a if a >= b else b

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b



@cython.boundscheck(False)
@cython.wraparound(False)
def maxpool_forward(DTYPE_t[:, :, :, ::1] inputs not None,
            tuple window not None,
            DTYPE_t[:, :, :, ::1] outputs not None,
            int pad,
            tuple strides not None,
            DTYPE_t[:, :, :, :, ::1] argmax not None):
    cdef int pool_h = window[0]
    cdef int pool_w = window[1]
    cdef int stride_x = strides[1]
    cdef int stride_y = strides[1]
    cdef int n_inputs = inputs.shape[0]
    cdef int n_filters = inputs.shape[1]
    cdef int in_h = inputs.shape[2]
    cdef int in_w = inputs.shape[3]
    cdef int out_w = outputs.shape[2]
    cdef int out_h = outputs.shape[3]
    cdef int i, c, y, x, y_out, x_out
    cdef int y_min, y_max, x_min, x_max
    cdef int in_y, in_x,
    cdef int in_y_max = 0
    cdef int in_x_max = 0
    cdef DTYPE_t value, new_value
    with nogil:
        for i in range(n_inputs):
            for c in range(n_filters):
                for y_out in range(out_h):
                    y = y_out*stride_y-pad
                    y_min = int_max(y, 0)
                    y_max = int_min(y+pool_h, in_h)
                    for x_out in range(out_w):
                        x = x_out*stride_x-pad
                        x_min = int_max(x, 0)
                        x_max = int_min(x+pool_w, in_w)
                        value = -1e38
                        for in_y in range(y_min, y_max):
                            for in_x in range(x_min, x_max):
                                new_value = inputs[i, c, in_y, in_x,]
                                if new_value > value:
                                    value = new_value
                                    in_y_max = in_y
                                    in_x_max = in_x
                        outputs[i, c, y_out, x_out] = value
                        argmax[i, c, y_out, x_out, 0] = <DTYPE_t>(in_y_max)
                        argmax[i, c, y_out, x_out, 1] = <DTYPE_t>(in_x_max)

@cython.boundscheck(False)
@cython.wraparound(False)
def maxpool_backward(DTYPE_t[:, :, :, ::1] inputs not None,
                     tuple window not None,
                     DTYPE_t[:, :, :, ::1] outputs not None,
                     const int pad,
                     tuple strides not None,
                     DTYPE_t[:, :, :, :, ::1] argmax not None,
                     DTYPE_t[:, :, :, ::1] in_deltas not None,
                     DTYPE_t[:, :, :, ::1] out_deltas not None):
    cdef int pool_h = window[0]
    cdef int pool_w = window[1]
    cdef int stride_x = strides[1]
    cdef int stride_y = strides[1]
    cdef int n_inputs = inputs.shape[0]
    cdef int n_filters = inputs.shape[1]
    cdef int in_h = inputs.shape[2]
    cdef int in_w = inputs.shape[3]
    cdef int out_w = outputs.shape[2]
    cdef int out_h = outputs.shape[3]
    cdef int i, c, y, x, in_y, in_x
    with nogil:
        for i in range(n_inputs):
            for c in range(n_filters):
                for y in range(out_h):
                    for x in range(out_w):
                        in_y = <int>(argmax[i, c, y, x, 0])
                        in_x = <int>(argmax[i, c, y, x, 1])
                        in_deltas[i, c, in_y, in_x] += out_deltas[i, c, y, x]
