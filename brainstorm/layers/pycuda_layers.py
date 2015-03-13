#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.python_layers import LayerBase
try:
    import pycuda.cumath
    import pycuda.gpuarray
    import scikits.cuda.linalg as linalg
    linalg.init()

    class PyCudaFFLayer(LayerBase):
        def __init__(self, size, in_size, sink_layers, source_layers, kwargs):
            super(PyCudaFFLayer, self).__init__(size, in_size, sink_layers,
                                                source_layers, kwargs)
            self.act_func = pycuda.cumath.tanh
            self.act_func_deriv = lambda y: 1 - linalg.multiply(y, y)

        def get_parameter_structure(self):
            return [
                ('W', (self.in_size, self.out_size)),
                ('b', self.out_size)
            ]

        def forward_pass(self, parameters, input_buffer, output_buffer):
            W, b = parameters['W'], parameters['b']
            for t in range(input_buffer.shape[0]):
                self.act_func(linalg.dot(input_buffer[t], W), out=output_buffer[t])

        def backward_pass(self, parameters, input_buffer, output_buffer,
                          in_delta_buffer, out_delta_buffer, gradient_buffer):
            W = parameters.W
            for t in range(input_buffer.shape[0]):
                d_z = linalg.multiply(self.act_func_deriv(output_buffer[t]),
                                      out_delta_buffer[t])
                in_delta_buffer[t, :].set_from_numpy(linalg.dot(d_z, W, transb='t'))

            dW, db = gradient_buffer
            for t in range(input_buffer.shape[0]):
                dz = linalg.multiply(self.act_func_deriv(output_buffer[t]),
                                     out_delta_buffer[t])
                dW += linalg.dot(input_buffer[t], dz, transa='t')


except ImportError as e:
    print(e)