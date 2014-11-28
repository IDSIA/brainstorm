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

        def get_parameter_size(self):
            return self.in_size * self.out_size + self.out_size

        def create_param_view(self, buffer):
            w_size = self.in_size * self.out_size
            W = buffer[:w_size].reshape(self.in_size, self.out_size)
            b = buffer[w_size:]
            return {'W': W, 'b': b}

        def forward_pass(self, parameters, input_buffer, output_buffer):
            W, b = parameters['W'], parameters['b']
            for t in range(input_buffer.shape[0]):
                self.act_func(linalg.dot(input_buffer[t], W), out=output_buffer[t])

        def backward_pass(self, parameters, input_buffer, output_buffer,
                          in_delta_buffer, out_delta_buffer):
            W = parameters['W']
            for t in range(input_buffer.shape[0]):
                d_z = linalg.multiply(self.act_func_deriv(output_buffer[t]), out_delta_buffer[t])
                in_delta_buffer[t, :].set(linalg.dot(d_z, W, transb='t'))

        def calculate_gradient(self, parameters, input_buffer, output_buffer,
                               out_delta_buffer, gradient_buffer):
            d_W, d_b = gradient_buffer['W'], gradient_buffer['b']
            for t in range(input_buffer.shape[0]):
                d_z = linalg.multiply(self.act_func_deriv(output_buffer[t]), out_delta_buffer[t])
                d_W += linalg.dot(input_buffer[t], d_z, transa='t')

except ImportError as e:
    print(e)