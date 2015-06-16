#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class ConvolutionLayerImpl(LayerBaseImpl):
    expected_kwargs = {'num_filters', 'kernel_size', 'stride', 'pad',
                       'activation_function'}

    def __init__(self, name, in_shapes, incoming_connections,
                 outgoing_connections, **kwargs):
        super(ConvolutionLayerImpl, self).__init__(
            name, in_shapes, incoming_connections, outgoing_connections,
            **kwargs)
        self.act_func = None
        self.act_func_deriv = None
        self.kwargs = kwargs
        assert 'num_filters' in kwargs, "num_filters must be specified for " \
                                        "ConvolutionLayer"
        assert 'kernel_size' in kwargs, "kernel_size must be specified for " \
                                        "ConvolutionLayer"
        self.num_filters = kwargs['num_filters']
        self.kernel_size = kwargs['kernel_size']

    def set_handler(self, new_handler):
        super(ConvolutionLayerImpl, self).set_handler(new_handler)

        # Assign act_func and act_dunc_derivs
        activation_functions = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(y, x),
                       lambda x, y, dy, dx: self.handler.copy_to(dx, dy)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activation_functions[
            self.kwargs.get('activation_function', 'linear')]

    def get_internal_structure(self):
       self.num_filters = self.out_shapes['default'].feature_size

       internals = OrderedDict()
       internals['Ha'] = ShapeTemplate('T', 'B', size)
       return internals

    def get_parameter_structure(self):
        in_size = self.in_shapes['default'].feature_size
        out_size = self.out_shapes['default'].feature_size

        parameters = OrderedDict()
        parameters['W'] = ShapeTemplate(in_size, out_size)
        parameters['b'] = ShapeTemplate(out_size)
        return parameters

    def _get_output_shapes(self):
        s = self.kwargs.get('size', self.in_shapes['default'].feature_size)
        if not isinstance(s, int):
            raise LayerValidationError('size must be int but was {}'.format(s))

        return {'default': ShapeTemplate('T', 'B', s)}