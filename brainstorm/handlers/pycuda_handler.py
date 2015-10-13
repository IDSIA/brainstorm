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
from brainstorm.utils import flatten_all_but_last

culinalg.init()

if has_cudnn:
    import ctypes
    import libcudnn as cudnn


NUM_CUDA_THREADS = 1024


def get_blocks(n):
    return (n + NUM_CUDA_THREADS - 1) // NUM_CUDA_THREADS


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
                'CUDNN_TENSOR_NHWC']
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

    def copy_to(self, src, dest):
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
        pass
        # pool_mode = cudnn.cudnnPoolingMode[
        #     'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING']
        # self._pool2d_backward_batch(inputs, window, outputs, padding,
        #                             stride, None, in_deltas, out_deltas,
        #                             pool_mode)

    def avgpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride):
        n, h, w, c = inputs.shape
        o_h, o_w = outputs.shape[1], outputs.shape[2]
        _avepool_fwd_fp32_impl(np.int32(outputs.size), inputs.gpudata,
                               np.int32(n), np.int32(h),
                               np.int32(w), np.int32(c),
                               np.int32(o_h), np.int32(o_w),
                               np.int32(window[0]), np.int32(window[1]),
                               np.int32(stride[0]), np.int32(stride[1]),
                               np.int32(padding), np.int32(padding),
                               outputs.gpudata,
                               block=(get_blocks(outputs.size), 1, 1),
                               grid=(NUM_CUDA_THREADS, 1, 1))
        # pool_mode = cudnn.cudnnPoolingMode[
        #     'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING']
        # self._pool2d_forward_batch(inputs, window, outputs, padding,
        #                            stride, None, pool_mode)

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

    def conv2d_backward_batch(self, inputs, params, padding, stride,
                              in_deltas, out_deltas, dparams, dbias):
        num_filters = params.shape[0]
        num_images, input_rows, input_cols, num_input_maps = inputs.shape
        kernel_shape = params.shape[1:]
        num_output_pixels = out_deltas.shape[1] * out_deltas.shape[2]
        num_kernel_params = np.prod(kernel_shape)

        dparams.fill(0.0)
        dbias.fill(0.0)
        tmp = self.zeros(dbias.shape)
        col = self.zeros((num_output_pixels, num_kernel_params))

        for i in range(num_images):
            num_cuda_kernels = num_output_pixels * num_input_maps

            _im2col_fp32_impl(np.int32(num_cuda_kernels), inputs[i],
                              np.int32(input_rows), np.int32(input_cols),
                              np.int32(kernel_shape[0]),
                              np.int32(kernel_shape[1]),
                              np.int32(padding), np.int32(padding),
                              np.int32(stride[0]), np.int32(stride[1]),
                              np.int32(out_deltas.shape[2]),
                              np.int32(num_input_maps),
                              col.gpudata,
                              block=(get_blocks(num_cuda_kernels), 1, 1),
                              grid=(NUM_CUDA_THREADS, 1, 1))

            # Compute gradients
            reshaped_dparams = dparams.reshape(num_filters, num_kernel_params)
            reshaped_out_deltas = out_deltas[i].reshape((num_output_pixels,
                                                         num_filters))
            self.dot_add_mm(reshaped_out_deltas, col, out=reshaped_dparams,
                            transa=True)

            self.sum_t(reshaped_out_deltas, axis=0, out=tmp)
            self.add_tt(tmp, dbias, out=dbias)

            # Compute in_deltas
            reshaped_params = params.reshape((num_filters, num_kernel_params))
            self.dot_mm(reshaped_out_deltas, reshaped_params, out=col)
            num_cuda_kernels = input_rows * input_cols * num_input_maps
            _col2im_fp32_impl(np.int32(num_cuda_kernels), col.gpudata,
                              np.int32(input_cols), np.int32(num_input_maps),
                              np.int32(kernel_shape[0]),
                              np.int32(kernel_shape[1]),
                              np.int32(padding), np.int32(padding),
                              np.int32(stride[0]), np.int32(stride[1]),
                              np.int32(out_deltas.shape[1]),
                              np.int32(out_deltas.shape[2]),
                              in_deltas[i],
                              block=(get_blocks(num_cuda_kernels), 1, 1),
                              grid=(NUM_CUDA_THREADS, 1, 1))

        # upscalex, upscaley = 1, 1  # currently not exposed to API
        #
        # n, h, w, c = inputs.shape
        # x_desc = cudnn.cudnnCreateTensorDescriptor()
        # cudnn.cudnnSetTensor4dDescriptor(x_desc, self.cudnn_tensor_format,
        #                                  self.cudnn_data_type, n, c, h, w)
        # id_desc = cudnn.cudnnCreateTensorDescriptor()
        # cudnn.cudnnSetTensor4dDescriptor(id_desc, self.cudnn_tensor_format,
        #                                  self.cudnn_data_type,
        #                                  n, c, h, w)
        # n, h, w, c = out_deltas.shape
        # od_desc = cudnn.cudnnCreateTensorDescriptor()
        # cudnn.cudnnSetTensor4dDescriptor(od_desc, self.cudnn_tensor_format,
        #                                  self.cudnn_data_type,
        #                                  n, c, h, w)
        # w_desc = cudnn.cudnnCreateFilterDescriptor()
        # cudnn.cudnnSetFilter4dDescriptor(w_desc, self.cudnn_data_type,
        #                                  *weights.shape)
        # dw_desc = cudnn.cudnnCreateFilterDescriptor()
        # cudnn.cudnnSetFilter4dDescriptor(dw_desc, self.cudnn_data_type,
        #                                  *weight_deltas.shape)
        # db_desc = cudnn.cudnnCreateTensorDescriptor()
        # cudnn.cudnnSetTensor4dDescriptor(db_desc, self.cudnn_tensor_format,
        #                                  self.cudnn_data_type, 1,
        #                                  bias_deltas.size, 1, 1)
        # conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        # cudnn.cudnnSetConvolution2dDescriptor(conv_desc, padding, padding,
        #                                       stride[0], stride[1], upscalex,
        #                                       upscaley, self.cudnn_convmode)
        #
        # alpha, beta = 1.0, 0.0
        # x_data = ctypes.c_void_p(int(inputs.gpudata))
        # w_data = ctypes.c_void_p(int(weights.gpudata))
        # id_data = ctypes.c_void_p(int(in_deltas.gpudata))
        # od_data = ctypes.c_void_p(int(out_deltas.gpudata))
        # dw_data = ctypes.c_void_p(int(weight_deltas.gpudata))
        # db_data = ctypes.c_void_p(int(bias_deltas.gpudata))
        #
        # cudnn.cudnnConvolutionBackwardFilter(self.cudnn_context, alpha,
        #                                      x_desc, x_data, od_desc, od_data,
        #                                      conv_desc, beta,
        #                                      dw_desc, dw_data)
        #
        # cudnn.cudnnConvolutionBackwardBias(self.cudnn_context, alpha,
        #                                    od_desc, od_data, beta, db_desc,
        #                                    db_data)
        # beta = 1.0  # Gradients w.r.t. inputs should be added
        # cudnn.cudnnConvolutionBackwardData(self.cudnn_context, alpha,
        #                                    w_desc, w_data, od_desc, od_data,
        #                                    conv_desc, beta,
        #                                    id_desc, id_data)
        # cudnn.cudnnDestroyTensorDescriptor(x_desc)
        # cudnn.cudnnDestroyFilterDescriptor(w_desc)
        # cudnn.cudnnDestroyTensorDescriptor(id_desc)
        # cudnn.cudnnDestroyTensorDescriptor(od_desc)
        # cudnn.cudnnDestroyFilterDescriptor(dw_desc)
        # cudnn.cudnnDestroyFilterDescriptor(db_desc)
        # cudnn.cudnnDestroyConvolutionDescriptor(conv_desc)

    def conv2d_forward_batch(self, inputs, params, bias, outputs,
                             padding, stride):
        num_filters = params.shape[0]
        num_images, input_rows, input_cols, num_input_maps = inputs.shape
        kernel_shape = params.shape[1:]
        num_output_pixels = outputs.shape[1] * outputs.shape[2]
        num_kernel_params = np.prod(kernel_shape)
        out_shape = (num_output_pixels, num_filters)
        num_cuda_kernels = num_output_pixels * num_input_maps

        for i in range(num_images):
            col = self.zeros((num_output_pixels, num_kernel_params))
            _im2col_fp32_impl(np.int32(num_cuda_kernels), inputs[i],
                              np.int32(input_rows), np.int32(input_cols),
                              np.int32(kernel_shape[0]),
                              np.int32(kernel_shape[1]),
                              np.int32(padding), np.int32(padding),
                              np.int32(stride[0]), np.int32(stride[1]),
                              np.int32(outputs.shape[2]),
                              np.int32(num_input_maps),
                              col.gpudata,
                              block=(get_blocks(num_cuda_kernels), 1, 1),
                              grid=(NUM_CUDA_THREADS, 1, 1))

            reshaped_params = params.reshape(num_filters, num_kernel_params)
            culinalg.dot(col, reshaped_params, transb='T',
                         out=outputs[i].reshape(out_shape))

        flat_outputs = flatten_all_but_last(outputs)
        self.add_mv(flat_outputs, bias, flat_outputs)
        # outputs += bias.reshape((1, 1, 1, num_filters))

        # upscalex, upscaley = 1, 1  # currently not exposed to API
        #
        # n, h, w, c = inputs.shape
        # x_desc = cudnn.cudnnCreateTensorDescriptor()
        # cudnn.cudnnSetTensor4dDescriptor(x_desc, self.cudnn_tensor_format,
        #                                  self.cudnn_data_type, n, c, h, w)
        #
        # n, h, w, c = outputs.shape
        # y_desc = cudnn.cudnnCreateTensorDescriptor()
        # cudnn.cudnnSetTensor4dDescriptor(y_desc, self.cudnn_tensor_format,
        #                                  self.cudnn_data_type, n, c, h, w)
        #
        # w_desc = cudnn.cudnnCreateFilterDescriptor()
        # cudnn.cudnnSetFilter4dDescriptor(w_desc, self.cudnn_data_type,
        #                                  *weights.shape)
        #
        # b_desc = cudnn.cudnnCreateTensorDescriptor()
        # cudnn.cudnnSetTensor4dDescriptor(b_desc, self.cudnn_tensor_format,
        #                                  self.cudnn_data_type, 1, bias.size, 1,
        #                                  1)
        #
        # conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        # cudnn.cudnnSetConvolution2dDescriptor(conv_desc, padding, padding,
        #                                       stride[0], stride[1], upscalex,
        #                                       upscaley, self.cudnn_convmode)
        #
        # # TODO: remove this sanity check once implementation works
        # outshape = cudnn.cudnnGetConvolution2dForwardOutputDim(
        #     conv_desc, x_desc, w_desc)
        # print(inputs.shape, weights.shape, outputs.shape)
        # print(outshape)
        # assert (outshape == outputs.shape)
        # assert (weights.shape[0] == bias.size)
        # assert (outputs.shape[1] == bias.size)
        #
        # # TODO: we hardcode a memory limit of zero for cudnn
        # algo = cudnn.cudnnGetConvolutionForwardAlgorithm(
        #     self.cudnn_context, x_desc, w_desc, conv_desc, y_desc,
        #     self.cudnn_convpref, 0)
        #
        # alpha, beta = 1.0, 0.0
        # x_data = ctypes.c_void_p(int(inputs.gpudata))
        # w_data = ctypes.c_void_p(int(weights.gpudata))
        # b_data = ctypes.c_void_p(int(bias.gpudata))
        # y_data = ctypes.c_void_p(int(outputs.gpudata))
        # cudnn.cudnnConvolutionForward(self.cudnn_context, alpha, x_desc,
        #                               x_data, w_desc, w_data, conv_desc, algo,
        #                               None, 0, beta, y_desc,
        #                               y_data)
        # beta = 1.0
        # cudnn.cudnnAddTensor(self.cudnn_context, self.cudnn_addmode, alpha,
        #                      b_desc, b_data, beta, y_desc, y_data)
        #
        # cudnn.cudnnDestroyTensorDescriptor(x_desc)
        # cudnn.cudnnDestroyTensorDescriptor(y_desc)
        # cudnn.cudnnDestroyFilterDescriptor(w_desc)
        # cudnn.cudnnDestroyTensorDescriptor(b_desc)
        # cudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
        # # cudnn.cudnnDestroy(cudnn_context)

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
            self.copy_to(cumisc.sum(a), out)
        else:
            raise NotImplementedError

    def _pool2d_forward_batch(self, inputs, window, outputs, padding,
                              stride, argmax, pooling_mode):
        pool_desc = cudnn.cudnnCreatePoolingDescriptor()
        cudnn.cudnnSetPooling2dDescriptor(pool_desc, pooling_mode,
                                          window[0], window[1], padding,
                                          padding, stride[0], stride[1])

        n, h, w, c = inputs.shape
        x_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(x_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, n, c, h, w)
        n, h, w, c = outputs.shape
        y_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(y_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, n, c, h, w)

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

        n, h, w, c = inputs.shape
        x_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(x_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, n, c, h, w)
        id_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(id_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, n, c, h, w)
        n, h, w, c = outputs.shape
        y_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(y_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, n, c, h, w)
        od_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(od_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, n, c, h, w)

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
        return out

    def tanh(self, x, y):
        tanh_kernel(x, y)

    def tanh_deriv(self, x, y, dy, dx):
        tanh_deriv_kernel(x, y, dy, dx)


# -------------------------- Activation functions --------------------------- #

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
_mod_softmax = SourceModule(__softmax_kernel_code)
_softmax_impl = _mod_softmax.get_function("softmax_kernel")

# ----------------------------- Caffe2 Kernels ------------------------------ #
# Please see Third Party License file for license information

__im2col_fp32_kernel_code = """
    __global__ void im2col_fp32_kernel(const int n, const float* data_im,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_t, const int pad_l,
        const int stride_h, const int stride_w,
        const int width_col, const int channels,
        float* data_col) {
      for (int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < (n);
           index += blockDim.x * gridDim.x) {
        int channel_in = index % channels;
        int w_out = index / channels % width_col;
        int h_out = index / channels / width_col;
        int h_in = h_out * stride_h - pad_t;
        int w_in = w_out * stride_w - pad_l;
        float* local_data_col = data_col +
            ((h_out * width_col) + w_out) * channels * kernel_h * kernel_w
            + channel_in;
        for (int i = 0; i < kernel_h; ++i) {
          int h = h_in + i;
          for (int j = 0; j < kernel_w; ++j) {
            int w = w_in + j;
            *local_data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
                data_im[(h * width + w) * channels + channel_in] : 0;
            local_data_col += channels;
          }
        }
      }
    }
    """

_mod_im2col_fp32 = SourceModule(__im2col_fp32_kernel_code)
_im2col_fp32_impl = _mod_im2col_fp32.get_function("im2col_fp32_kernel")

__col2im_fp32_kernel_code = """
    __global__ void col2im_fp32_kernel(const int n, const float* data_col,
        const int width, const int channels,
        const int patch_h, const int patch_w,
        const int pad_t, const int pad_l,
        const int stride_h, const int stride_w,
        const int height_col, const int width_col,
        float* data_im) {
      for (int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < (n);
           index += blockDim.x * gridDim.x) {
        float val = 0;
        int c = index % channels;
        int w = index / channels % width + pad_l;
        int h = index / channels / width + pad_t;
        // compute the start and end of the output
        int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
        int w_col_end = min(w / stride_w + 1, width_col);
        int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
        int h_col_end = min(h / stride_h + 1, height_col);
        int channels_col = patch_h * patch_w * channels;
        /*
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
          for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
            int c_col = ((h - h_col * stride_h) * patch_w + w - w_col * stride_w) * channels + c;
            val += data_col[(h_col * width_col + w_col) * channels_col + c_col];
          }
        }
        */
        // Equivalent of above
        int offset = (h * patch_w + w) * channels + c;
        int coeff_h_col = width_col * channels_col - stride_h * patch_w * channels;
        int coeff_w_col = channels_col - stride_w * channels;
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
          for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
            val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
          }
        }
        data_im[index] += val;
      }
    }
    """

_mod_col2im_fp32 = SourceModule(__col2im_fp32_kernel_code)
_col2im_fp32_impl = _mod_col2im_fp32.get_function("col2im_fp32_kernel")

__maxpool_fwd_fp32_kernel = """
    #include "float.h"
    __global__ void max_pool_fwd(const int nthreads, const float*
    bottom_data,
        const int height, const int width,
        const int channels, const int pooled_height, const int pooled_width,
        const int kernel_h, const int kernel_w, const int stride_h,
        const int stride_w, const int pad_t, const int pad_l, float* top_data,
        float* mask) {
      for (int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < (nthreads);
           index += blockDim.x * gridDim.x) {
        int n = index;
        int c = n % channels;
        n /= channels;
        int wstart = (n % pooled_width) * stride_w - pad_l;
        n /= pooled_width;
        int hstart = (n % pooled_height) * stride_h - pad_t;
        n /= pooled_height;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        float maxval = -FLT_MAX;
        int maxidx = -1;
        bottom_data += n * height * width * channels;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int idx = (h * width + w) * channels + c;
            if (bottom_data[idx] > maxval) {
              maxidx = idx;
              maxval = bottom_data[idx];
            }
          }
        }
        top_data[index] = maxval;
        mask[index] = maxidx;
      }
    }
    """
_mod_maxpool_fwd_fp32 = SourceModule(__maxpool_fwd_fp32_kernel)
_maxpool_fwd_fp32_impl = _mod_maxpool_fwd_fp32.get_function("max_pool_fwd")

__maxpool_bwd_fp32_kernel = """
    __global__ void max_pool_bwd(
        const int nthreads, const float* top_diff, const float* mask,
        const int top_offset, const int bottom_offset, float* bottom_diff) {
      for (int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < (nthreads);
           index += blockDim.x * gridDim.x) {
        int image_id = (index / top_offset);
        atomicAdd(bottom_diff + image_id * bottom_offset + int(mask[index]),
                  top_diff[index]);
      }
    }
    """
_mod_maxpool_bwd_fp32 = SourceModule(__maxpool_bwd_fp32_kernel)
_maxpool_bwd_fp32_impl = _mod_maxpool_bwd_fp32.get_function("max_pool_bwd")

__avepool_fwd_fp32_kernel = """
    __global__ void ave_pool_fwd(
        const int nthreads, const float* bottom_data,
        const int num, const int height, const int width,
        const int channels, const int pooled_height, const int pooled_width,
        const int kernel_h, const int kernel_w, const int stride_h,
        const int stride_w, const int pad_t, const int pad_l, float* top_data) {
      for (int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < (nthreads);
           index += blockDim.x * gridDim.x) {
        int c = index % channels;
        int pw = (index / channels) % pooled_width;
        int ph = (index / channels / pooled_width) % pooled_height;
        int n = index / channels / pooled_width / pooled_height;
        int hstart = ph * stride_h - pad_t;
        int wstart = pw * stride_w - pad_l;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        float output = 0;
        bottom_data += n * height * width * channels;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            output += bottom_data[(h * width + w) * channels + c];
          }
        }
        // Make sure that all pixels were not padding
        int pool_size = max((hend - hstart) * (wend - wstart), 1);
        top_data[index] = output / pool_size;
      }
    }
    """
_mod_avepool_fwd_fp32 = SourceModule(__avepool_fwd_fp32_kernel)
_avepool_fwd_fp32_impl = _mod_avepool_fwd_fp32.get_function("ave_pool_fwd")

__avepool_bwd_fp32_kernel = """
    __global__ void ave_pool_bwd(const int nthreads,
        const float* const top_diff, const int num, const int height,
        const int width, const int channels, const int pooled_height,
        const int pooled_width, const int kernel_h, const int kernel_w,
        const int stride_h, const int stride_w, const int pad_t,
        const int pad_l, float* const bottom_diff) {
      for (int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < (nthreads);
           index += blockDim.x * gridDim.x) {
        // find out the local index
        // find out the local offset
        const int c = index % channels;
        const int w = index / channels % width + pad_l;
        const int h = (index / channels / width) % height + pad_t;
        const int n = index / channels / width / height;
        const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        const int phend = min(h / stride_h + 1, pooled_height);
        const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        const int pwend = min(w / stride_w + 1, pooled_width);
        float gradient = 0;
        const float* const top_diff_slice =
            top_diff + n * pooled_height * pooled_width * channels + c;
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            // figure out the pooling size
            int hstart = ph * stride_h - pad_t;
            int wstart = pw * stride_w - pad_l;
            int hend = min(hstart + kernel_h, height);
            int wend = min(wstart + kernel_w, width);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            // Make sure that all pixels were not padding
            int pool_size = max((hend - hstart) * (wend - wstart), 1);
            gradient +=
                top_diff_slice[(ph * pooled_width + pw) * channels] / pool_size;
          }
        }
        bottom_diff[index] += gradient;
      }
    }
    """
_mod_avepool_bwd_fp32 = SourceModule(__avepool_bwd_fp32_kernel)
_avepool_bwd_fp32_impl = _mod_avepool_bwd_fp32.get_function("ave_pool_bwd")
