#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function

from brainstorm.handlers.pycuda_handler import PyCudaHandler
from brainstorm.optional import has_cudnn

if has_cudnn:
    import ctypes
    import libcudnn as cudnn


NUM_CUDA_THREADS = 1024


def get_blocks(n):
    return (n + NUM_CUDA_THREADS - 1) // NUM_CUDA_THREADS


class CuDnnHandler(PyCudaHandler):
    __undescribed__ = {'context', 'dtype', 'EMPTY', 'rnd',
                       'cudnn_context', 'cudnn_tensor_format',
                       'cudnn_data_type', 'cudnn_convmode', 'cudnn_convpref',
                       'cudnn_addmode'}

    def __init__(self, seed=None):
        super(CuDnnHandler, self).__init__(seed=seed)

        if not has_cudnn:
            raise ImportError("cudnn-python-wrappers package is "
                              "required to use cuDNN but could not be "
                              "imported.")
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

    def __init_from_description__(self, description):
        self.__init__()

    # ----------------------- Mathematical operations ----------------------- #

    def avgpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, in_deltas, out_deltas):
        pool_mode = cudnn.cudnnPoolingMode[
            'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING']
        self._pool2d_backward_batch(inputs, window, outputs, padding,
                                    stride, None, in_deltas, out_deltas,
                                    pool_mode)

    def avgpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride):
        pool_mode = cudnn.cudnnPoolingMode[
            'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING']
        self._pool2d_forward_batch(inputs, window, outputs, padding,
                                   stride, None, pool_mode)

    def conv2d_backward_batch(self, inputs, params, padding, stride,
                              in_deltas, out_deltas, dparams, dbias):
        upscalex, upscaley = 1, 1  # currently not exposed to API

        n, h, w, c = inputs.shape
        x_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(x_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, n, c, h, w)
        id_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(id_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type,
                                         n, c, h, w)
        n, h, w, c = out_deltas.shape
        od_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(od_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type,
                                         n, c, h, w)
        w_desc = cudnn.cudnnCreateFilterDescriptor()
        cudnn.cudnnSetFilter4dDescriptor(w_desc, self.cudnn_data_type,
                                         *params.shape)
        dw_desc = cudnn.cudnnCreateFilterDescriptor()
        cudnn.cudnnSetFilter4dDescriptor(dw_desc, self.cudnn_data_type,
                                         *dparams.shape)
        db_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(db_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, 1,
                                         dbias.size, 1, 1)
        conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetConvolution2dDescriptor(conv_desc, padding, padding,
                                              stride[0], stride[1], upscalex,
                                              upscaley, self.cudnn_convmode)

        alpha, beta = 1.0, 0.0
        x_data = ctypes.c_void_p(int(inputs.gpudata))
        w_data = ctypes.c_void_p(int(params.gpudata))
        id_data = ctypes.c_void_p(int(in_deltas.gpudata))
        od_data = ctypes.c_void_p(int(out_deltas.gpudata))
        dw_data = ctypes.c_void_p(int(dparams.gpudata))
        db_data = ctypes.c_void_p(int(dbias.gpudata))

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

    def conv2d_forward_batch(self, inputs, params, bias, outputs,
                             padding, stride):
        upscalex, upscaley = 1, 1  # currently not exposed to API

        n, h, w, c = inputs.shape
        x_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(x_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, n, c, h, w)

        n, h, w, c = outputs.shape
        y_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnSetTensor4dDescriptor(y_desc, self.cudnn_tensor_format,
                                         self.cudnn_data_type, n, c, h, w)

        w_desc = cudnn.cudnnCreateFilterDescriptor()
        cudnn.cudnnSetFilter4dDescriptor(w_desc, self.cudnn_data_type,
                                         *params.shape)

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
        print(inputs.shape, params.shape, outputs.shape)
        print(outshape)
        assert (outshape == outputs.shape)
        assert (params.shape[0] == bias.size)
        assert (outputs.shape[1] == bias.size)

        # TODO: we hardcode a memory limit of zero for cudnn
        algo = cudnn.cudnnGetConvolutionForwardAlgorithm(
            self.cudnn_context, x_desc, w_desc, conv_desc, y_desc,
            self.cudnn_convpref, 0)

        alpha, beta = 1.0, 0.0
        x_data = ctypes.c_void_p(int(inputs.gpudata))
        w_data = ctypes.c_void_p(int(params.gpudata))
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
