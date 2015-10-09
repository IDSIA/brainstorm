#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function

import numpy as np
import pycuda
import skcuda.linalg as culinalg
import skcuda.misc as cumisc
from pycuda import cumath, gpuarray
from pycuda.compiler import SourceModule
from pycuda.curandom import XORWOWRandomNumberGenerator
from pycuda.elementwise import ElementwiseKernel

from brainstorm.handlers.base_handler import Handler
from brainstorm.optional import has_cudnn
from brainstorm.randomness import global_rnd

culinalg.init()

if has_cudnn:
    import ctypes
    import libcudnn as cudnn


class PyCudaHandler(Handler):
    __undescribed__ = {'context', 'dtype', 'EMPTY', 'rnd',
                       'cudnn_context', 'cudnn_tensor_format',
                       'cudnn_data_type', 'cudnn_convmode', 'cudnn_convpref',
                       'cudnn_addmode'}

    def __init__(self, seed=None, init_cudnn=True):
        self.dtype = np.float32
        self.context = cumisc._global_cublas_handle
        self.EMPTY = gpuarray.zeros((), dtype=self.dtype)
        if seed is None:
            seed = global_rnd.generate_seed()

        def get_seeds(n):
            return gpuarray.to_gpu(np.ones(n, np.int32) * seed)
        self.rnd = XORWOWRandomNumberGenerator(seed_getter=get_seeds)

        if init_cudnn:
            if not has_cudnn:
                raise ImportError("cudnn-python-wrappers package is "
                                  "required to use cuDNN but could not be "
                                  "imported.")
            self.init_cudnn = init_cudnn
            self.cudnn_context = cudnn.cudnnCreate()
            self.cudnn_tensor_format = cudnn.cudnnTensorFormat[
                'CUDNN_TENSOR_NCHW']
            self.cudnn_data_type = cudnn.cudnnDataType[
                'CUDNN_DATA_FLOAT']
            self.cudnn_convmode = cudnn.cudnnConvolutionMode[
                'CUDNN_CROSS_CORRELATION']
            # TODO we should use use PREFER_FASTEST eventually!
            self.cudnn_convpref = cudnn.cudnnConvolutionFwdPreference[
                # 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST']
                'CUDNN_CONVOLUTION_FWD_NO_WORKSPACE']
            self.cudnn_addmode = cudnn.cudnnAddMode['CUDNN_ADD_SAME_C']

    array_type = pycuda.gpuarray.GPUArray

    def __init_from_description__(self, description):
        self.__init__()

    # ------------------------- Allocate new memory ------------------------- #

    def allocate(self, size):
        return gpuarray.zeros(size, dtype=self.dtype)

    def ones(self, shape):
        a = self.zeros(shape)
        self.fill(a, 1.0)
        return a

    def zeros(self, shape):
        return gpuarray.zeros(shape=shape, dtype=self.dtype)

    # ---------------------------- Copy and Fill ---------------------------- #

    def copy_to(self, dest, src):
        # Copy data from src to dest (both must be GPUArrays)
        pycuda.driver.memcpy_dtod(dest.gpudata, src.gpudata, dest.nbytes)

    def create_from_numpy(self, arr):
        return gpuarray.to_gpu(arr.astype(self.dtype))

    def fill(self, mem, val):
        mem.fill(val)

    def get_numpy_copy(self, mem):
        assert type(mem) == self.array_type
        return mem.get()

    def set_from_numpy(self, mem, arr):
        assert mem.shape == arr.shape, "Shape of destination ({}) != Shape " \
                                       "of source ({})".format(mem.shape,
                                                               arr.shape)
        mem.set(arr.astype(self.dtype))

    # ---------------------------- Debug helpers ---------------------------- #

    def is_fully_finite(self, a):
        temp = gpuarray.zeros_like(a)
        check_inf_or_nan_kernel(a, temp)
        return np.all(temp.get())

    # ----------------------- Mathematical operations ----------------------- #

    def abs_t(self, a, out):
        cumath.fabs(a, out=out)

    def add_mv(self, m, v, out):
        cumisc.add_matvec(m, v, out=out)

    def add_st(self, s, t, out):
        add_st_kernel(s, t, out)

    def add_tt(self, a, b, out):
        add_mm_kernel(a, b, out)

    def avgpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, in_deltas, out_deltas):
        pool_mode = cudnn.cudnnPoolingMode[
            'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING']
        self._pool2d_backward_batch(inputs, window, outputs, padding,
                                    stride, None, in_deltas, out_deltas,
                                    pool_mode)

    def avgpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride):
        pool_mode = cudnn.cudnnPoolingMode[
            'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING']
        self._pool2d_forward_batch(inputs, window, outputs, padding,
                                   stride, None, pool_mode)

    def binarize_v(self, v, out):
        binarize_v_kernel(out, v, out.shape[0], out.shape[1])

    def broadcast_features_t(self, a, out):
        assert len(a.shape) == 3
        assert a.shape[2] == 1
        assert len(out.shape) > 2
        a_flat = a.reshape(a.size)
        out_flat = out.reshape(out.size)
        broadcast_features_kernel(out_flat, a_flat, np.prod(out.shape[2:]))

    def clip_t(self, a, a_min, a_max, out):
        clip_kernel(a, out, a_min, a_max)

    def conv2d_backward_batch(self, inputs, weights, padding, stride,
                              in_deltas, out_deltas, weight_deltas,
                              bias_deltas):
        upscalex, upscaley = 1, 1  # currently not exposed to API

        x_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(x_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, *inputs.shape)
        id_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(id_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type,
                                         *in_deltas.shape)
        od_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(od_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type,
                                         *out_deltas.shape)
        w_desc = cudnn.cudnnCreateFilterDescriptor()
        cudnn.cudnnSetFilter4dDescriptor(w_desc, self.cudnn_data_type,
                                         *weights.shape)
        dw_desc = cudnn.cudnnCreateFilterDescriptor()
        cudnn.cudnnSetFilter4dDescriptor(dw_desc, self.cudnn_data_type,
                                         *weight_deltas.shape)
        db_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(db_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, 1,
                                         bias_deltas.size, 1, 1)
        conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetConvolution2dDescriptor(conv_desc, padding, padding,
                                              stride[0], stride[1], upscalex,
                                              upscaley, self.cudnn_convmode)

        alpha, beta = 1.0, 0.0
        x_data = ctypes.c_void_p(int(inputs.gpudata))
        w_data = ctypes.c_void_p(int(weights.gpudata))
        id_data = ctypes.c_void_p(int(in_deltas.gpudata))
        od_data = ctypes.c_void_p(int(out_deltas.gpudata))
        dw_data = ctypes.c_void_p(int(weight_deltas.gpudata))
        db_data = ctypes.c_void_p(int(bias_deltas.gpudata))

        cudnn.cudnnConvolutionBackwardFilter(self.cudnn_context, alpha,
                                             x_desc, x_data, od_desc, od_data,
                                             conv_desc, beta,
                                             dw_desc, dw_data)

        cudnn.cudnnConvolutionBackwardBias(self.cudnn_context, alpha,
                                           od_desc, od_data, beta, db_desc,
                                           db_data)
        beta = 1.0  # Gradients w.r.t. inputs should be added
        cudnn.cudnnConvolutionBackwardData(self.cudnn_context, alpha,
                                           w_desc, w_data, od_desc, od_data,
                                           conv_desc, beta,
                                           id_desc, id_data)
        cudnn.cudnnDestroyTensorDescriptor(x_desc)
        cudnn.cudnnDestroyFilterDescriptor(w_desc)
        cudnn.cudnnDestroyTensorDescriptor(id_desc)
        cudnn.cudnnDestroyTensorDescriptor(od_desc)
        cudnn.cudnnDestroyFilterDescriptor(dw_desc)
        cudnn.cudnnDestroyFilterDescriptor(db_desc)
        cudnn.cudnnDestroyConvolutionDescriptor(conv_desc)

    def conv2d_forward_batch(self, inputs, weights, bias, outputs,
                             padding, stride):
        upscalex, upscaley = 1, 1  # currently not exposed to API

        x_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(x_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, *inputs.shape)

        w_desc = cudnn.cudnnCreateFilterDescriptor()
        cudnn.cudnnSetFilter4dDescriptor(w_desc, self.cudnn_data_type,
                                         *weights.shape)

        b_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(b_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, 1, bias.size, 1,
                                         1)

        conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetConvolution2dDescriptor(conv_desc, padding, padding,
                                              stride[0], stride[1], upscalex,
                                              upscaley, self.cudnn_convmode)

        # TODO: remove this sanity check once implementation works
        outshape = cudnn.cudnnGetConvolution2dForwardOutputDim(
            conv_desc, x_desc, w_desc)
        assert (outshape == outputs.shape)
        assert (weights.shape[0] == bias.size)
        assert (outputs.shape[1] == bias.size)

        y_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(y_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, *outputs.shape)

        # TODO: we hardcode a memory limit of zero for cudnn
        algo = cudnn.cudnnGetConvolutionForwardAlgorithm(
            self.cudnn_context, x_desc, w_desc, conv_desc, y_desc,
            self.cudnn_convpref, 0)

        alpha, beta = 1.0, 0.0
        x_data = ctypes.c_void_p(int(inputs.gpudata))
        w_data = ctypes.c_void_p(int(weights.gpudata))
        b_data = ctypes.c_void_p(int(bias.gpudata))
        y_data = ctypes.c_void_p(int(outputs.gpudata))
        cudnn.cudnnConvolutionForward(self.cudnn_context, alpha, x_desc,
                                      x_data, w_desc, w_data, conv_desc, algo,
                                      None, 0, beta, y_desc,
                                      y_data)
        beta = 1.0
        cudnn.cudnnAddTensor(self.cudnn_context, self.cudnn_addmode, alpha,
                             b_desc, b_data, beta, y_desc, y_data)

        cudnn.cudnnDestroyTensorDescriptor(x_desc)
        cudnn.cudnnDestroyTensorDescriptor(y_desc)
        cudnn.cudnnDestroyFilterDescriptor(w_desc)
        cudnn.cudnnDestroyTensorDescriptor(b_desc)
        cudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
        # cudnn.cudnnDestroy(cudnn_context)

    def dot_add_mm(self, a, b, out, transa=False, transb=False):
        transa = 'T' if transa else 'N'
        transb = 'T' if transb else 'N'
        culinalg.add_dot(a, b, out, transa, transb)

    def dot_mm(self, a, b, out, transa=False, transb=False):
        transa = 'T' if transa else 'N'
        transb = 'T' if transb else 'N'
        culinalg.dot(a, b, transa=transa, transb=transb, out=out)

    def divide_mv(self, m, v, out):
        cumisc.div_matvec(m, v, out=out)

    def divide_tt(self, a, b, out):
        div_kernel(a, b, out)

    def fill_gaussian(self, mean, std, out):
        self.rnd.fill_normal(out)
        self.mult_st(std, out, out=out)
        self.add_st(mean, out, out=out)

    def generate_probability_mask(self, mask, probability):
        self.rnd.fill_uniform(mask)
        create_probabilistic_mask_kernel(mask, probability, mask)

    def index_m_by_v(self, m, v, out):
        index_m_by_v_kernel(out, v, m, m.shape[0], m.shape[1])

    def log_t(self, a, out):
        cumath.log(a, out=out)

    def maxpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, argmax, in_deltas, out_deltas):
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_MAX']
        self._pool2d_backward_batch(inputs, window, outputs, padding, stride,
                                    argmax, in_deltas, out_deltas,
                                    pool_mode)

    def maxpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride, argmax):
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_MAX']
        self._pool2d_forward_batch(inputs, window, outputs, padding,
                                   stride, argmax, pool_mode)

    def mult_add_st(self, s, t, out):
        mult_add_st_kernel(s, t, out)

    def mult_add_tt(self, a, b, out):
        mult_add_kernel(a, b, out)

    def mult_mv(self, m, v, out):
        if m.shape == v.shape:
            self.mult_tt(m, v, out=out)
        else:
            cumisc.mult_matvec(m, v, out=out)

    def mult_st(self, s, t, out):
        mult_st_kernel(s, t, out)

    def mult_tt(self, a, b, out):
        mult_tt_kernel(a, b, out)

    def sign_t(self, a, out):
        sign_kernel(a, out)

    def sqrt_t(self, a, out):
        cumath.sqrt(a, out)

    def subtract_mv(self, m, v, out):
        cumisc.binaryop_matvec('-', m, v, None, out, None)

    def subtract_tt(self, a, b, out):
        subtract_mm_kernel(a, b, out)

    def sum_t(self, a, axis, out):
        if len(a.shape) < 3 and (axis == 0 or axis == 1):
            cumisc.sum(a, axis, out)
        elif axis is None:
            self.copy_to(out, cumisc.sum(a))
        else:
            raise NotImplementedError

    def _pool2d_forward_batch(self, inputs, window, outputs, padding,
                              stride, argmax, pooling_mode):
        pool_desc = cudnn.cudnnCreatePoolingDescriptor()
        cudnn.cudnnSetPooling2dDescriptor(pool_desc, pooling_mode,
                                          window[0], window[1], padding,
                                          padding, stride[0], stride[1])

        x_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(x_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, *inputs.shape)
        y_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(y_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, *outputs.shape)

        # TODO: remove this sanity check once implementation works
        # outshape = cudnn.cudnnGetPooling2dForwardOutputDim(
        #    conv_desc, x_desc)
        # assert(outshape == outputs.shape)
        x_data = ctypes.c_void_p(int(inputs.gpudata))
        y_data = ctypes.c_void_p(int(outputs.gpudata))
        alpha, beta = 1.0, 0.0
        cudnn.cudnnPoolingForward(self.cudnn_context, pool_desc, alpha,
                                  x_desc, x_data, beta, y_desc, y_data)

        cudnn.cudnnDestroyTensorDescriptor(x_desc)
        cudnn.cudnnDestroyTensorDescriptor(y_desc)
        cudnn.cudnnDestroyPoolingDescriptor(pool_desc)

    def _pool2d_backward_batch(self, inputs, window, outputs, padding, stride,
                               argmax, in_deltas, out_deltas, pooling_mode):
        pool_desc = cudnn.cudnnCreatePoolingDescriptor()
        cudnn.cudnnSetPooling2dDescriptor(pool_desc, pooling_mode,
                                          window[0], window[1], padding,
                                          padding, stride[0], stride[1])

        x_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(x_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, *inputs.shape)
        y_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(y_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, *outputs.shape)
        id_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(id_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type,
                                         *in_deltas.shape)
        od_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(od_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type,
                                         *out_deltas.shape)

        x_data = ctypes.c_void_p(int(inputs.gpudata))
        y_data = ctypes.c_void_p(int(outputs.gpudata))
        id_data = ctypes.c_void_p(int(in_deltas.gpudata))
        od_data = ctypes.c_void_p(int(out_deltas.gpudata))
        alpha, beta = 1.0, 1.0
        cudnn.cudnnPoolingBackward(self.cudnn_context, pool_desc, alpha,
                                   y_desc, y_data, od_desc, od_data, x_desc,
                                   x_data, beta,
                                   id_desc, id_data)

        cudnn.cudnnDestroyTensorDescriptor(x_desc)
        cudnn.cudnnDestroyTensorDescriptor(y_desc)
        cudnn.cudnnDestroyTensorDescriptor(id_desc)
        cudnn.cudnnDestroyTensorDescriptor(od_desc)
        cudnn.cudnnDestroyPoolingDescriptor(pool_desc)

    # ------------------------ Activation functions ------------------------- #

    def rel(self, x, y):
        rel_kernel(x, y)

    def rel_deriv(self, x, y, dy, dx):
        rel_deriv_kernel(x, y, dy, dx)

    def sigmoid(self, x, y):
        sigmoid_kernel(x, y)

    def sigmoid_deriv(self, x, y, dy, dx):
        sigmoid_deriv_kernel(x, y, dy, dx)

    def softmax_m(self, m, out):
        n, k = m.shape
        tmp = gpuarray.empty((1, n), dtype=m.dtype)
        _softmax_impl(m, tmp.gpudata, out, np.int32(n),
                      np.int32(k), block=(32, 1, 1), grid=(n, 1, 1))

    def tanh(self, x, y):
        tanh_kernel(x, y)

    def tanh_deriv(self, x, y, dy, dx):
        tanh_deriv_kernel(x, y, dy, dx)


# -------------------------------- Kernels ---------------------------------- #

add_mm_kernel = ElementwiseKernel(
    "float* x, float* y, float *out",
    "out[i] = x[i] + y[i]",
    "add_mm_kernel"
)

add_st_kernel = ElementwiseKernel(
    "float x, float* y, float *out",
    "out[i] = x + y[i]",
    "add_st_kernel"
)

binarize_v_kernel = ElementwiseKernel(
    "float* out, float* v, int nrows, int ncols",
    "out[i] = v[i / ncols] == (i % ncols) ? 1.0f : 0.0f",
    "binarize_v_kernel"
)

broadcast_features_kernel = ElementwiseKernel(
    "float* out, float* a, unsigned int broadcast_size",
    "out[i] = a[i / broadcast_size]",
    "bc_features_kernel"
)

check_inf_or_nan_kernel = ElementwiseKernel(
    b"float* inp, float* result",
    b"if (isnan(inp[i]) || isinf(inp[i])) result[i] = 1;",
    b"check_inf_or_nan_kernel"
)

clip_kernel = ElementwiseKernel(
    "float* a, float* out, float a_min, float a_max",
    "out[i] = fminf(fmaxf(a[i], a_min), a_max);",
    "clip_kernel"
)

create_probabilistic_mask_kernel = ElementwiseKernel(
    "float* inp, float prob, float* mask",
    "if (inp[i] < prob) mask[i] = 1; else mask[i] = 0;",
    "create_probabilistic_mask_kernel"
)

div_kernel = ElementwiseKernel(
    "float* a, float* b, float* out",
    "out[i] = a[i] / b[i];",
    "div_kernel"
)

index_m_by_v_kernel = ElementwiseKernel(
    "float* out, float* v, float* m, int nrows, int ncols",
    "out[i] = m[i * ncols + int(v[i])]",
    "index_m_by_v_kernel"
)

mult_add_kernel = ElementwiseKernel(
    "float* x, float* y, float *out",
    "out[i] += x[i] * y[i]",
    "mult_add_kernel"
)

mult_add_st_kernel = ElementwiseKernel(
    "float x, float* y, float *out",
    "out[i] += x * y[i]",
    "mult_add_st_kernel"
)

mult_st_kernel = ElementwiseKernel(
    "float x, float* y, float *out",
    "out[i] = x * y[i]",
    "mult_st_kernel"
)

mult_tt_kernel = ElementwiseKernel(
    "float* x, float* y, float *out",
    "out[i] = x[i] * y[i]",
    "mult_tt_kernel"
)

rel_deriv_kernel = ElementwiseKernel(
    "float* x, float* y, float* dy, float* dx",
    "if (y[i] > 0) dx[i] = dy[i]; else dx[i] = 0.0;",
    "rel_deriv_kernel"
)

rel_kernel = ElementwiseKernel(
    "float* x, float* y",
    "if (x[i] > 0) y[i] = x[i]; else y[i] = 0.0;",
    "rel_kernel"
)

sigmoid_deriv_kernel = ElementwiseKernel(
    "float* x, float* y, float* dy, float* dx",
    "dx[i] = dy[i] * y[i] * (1.0 - y[i])",
    "sigmoid_deriv_kernel"
)

sigmoid_kernel = ElementwiseKernel(
    "float* x, float* y",
    "y[i] = (x[i]>=0) ? 1.0/(1.0 + exp(-1.0*x[i])) : exp(1.0*x[i])/(1.0 + exp(1.0*x[i]))",
    "sigmoid_kernel"
)

sign_kernel = ElementwiseKernel(
    "float* a, float* out",
    "out[i] = (a[i] > 0) - (a[i] < 0);",
    "sign_kernel"
)

subtract_mm_kernel = ElementwiseKernel(
    "float* x, float* y, float *out",
    "out[i] = x[i] - y[i]",
    "subtract_mm_kernel"
)

tanh_deriv_kernel = ElementwiseKernel(
    "float* x, float* y, float* dy, float* dx",
    "dx[i] = dy[i] * (1.0 - y[i] * y[i])",
    "tanh_deriv_kernel"
)

tanh_kernel = ElementwiseKernel(
    "float* x, float* y",
    "y[i] = tanh(x[i])",
    "tanh_kernel"
)

__softmax_kernel_code = """
    #include "float.h"

    __global__ void softmax_kernel(float* mat, float* tmp, float* out,
                                   unsigned int height, unsigned int width) {
          __shared__ float max_vals[32];
        float cur_max = -FLT_MAX;
        float val = 0;

        for (unsigned int i = threadIdx.x; i < width; i += 32) {
            val = mat[blockIdx.x * width + i];
            if (val > cur_max)
                cur_max = val;
        }

        max_vals[threadIdx.x] = cur_max;
        __syncthreads();
        if (threadIdx.x == 0) {
            cur_max = -FLT_MAX;
            for (unsigned int i = 0; i < 32; i++) {
                if (max_vals[i] > cur_max)
                    cur_max = max_vals[i];
            }
            tmp[blockIdx.x] = cur_max;
        }
        __syncthreads();


        float sum = 0.0;
        for (unsigned int i = threadIdx.x; i < width; i += 32) {
            float x =  __expf(mat[blockIdx.x * width + i] - tmp[blockIdx.x]);
            out[blockIdx.x * width + i] = x;
            sum += x;
        }
        max_vals[threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            sum = 0.0;
            for (unsigned int i = 0; i < 32; i++)
                sum += max_vals[i];
            tmp[blockIdx.x] = sum;
        }
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < width; i += 32) {
            out[blockIdx.x * width + i] /= tmp[blockIdx.x];
        }
    }
    """
_mod = SourceModule(__softmax_kernel_code)
_softmax_impl = _mod.get_function("softmax_kernel")
