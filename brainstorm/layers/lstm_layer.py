#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class LstmLayerImpl(LayerBaseImpl):
    expected_kwargs = {'size', 'activation_function'}

    def _setup_hyperparameters(self):
        self.act_func = lambda x, y: None
        self.act_func_deriv = lambda x, y, dy, dx: None
        self.size = self.kwargs.get('size',
                                    self.in_shapes['default'].feature_size)
        if not isinstance(self.size, int):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))

    def set_handler(self, new_handler):
        super(LstmLayerImpl, self).set_handler(new_handler)

        # Assign act_func and act_dunc_derivs
        activation_functions = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(y, x),
                       lambda x, y, dy, dx: self.handler.copy_to(dx, dy)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activation_functions[
            self.kwargs.get('activation_function', 'tanh')]

    def get_parameter_structure(self):
        in_size = self.in_shapes['default'].feature_size
        
        parameters = OrderedDict()
        parameters['Wz'] = ShapeTemplate(self.size, in_size)
        parameters['Wi'] = ShapeTemplate(self.size, in_size)
        parameters['Wf'] = ShapeTemplate(self.size, in_size)
        parameters['Wo'] = ShapeTemplate(self.size, in_size)

        parameters['Rz'] = ShapeTemplate(self.size, self.size)
        parameters['Ri'] = ShapeTemplate(self.size, self.size)
        parameters['Rf'] = ShapeTemplate(self.size, self.size)
        parameters['Ro'] = ShapeTemplate(self.size, self.size)

        parameters['bz'] = ShapeTemplate(self.size)
        parameters['bi'] = ShapeTemplate(self.size)
        parameters['bf'] = ShapeTemplate(self.size)
        parameters['bo'] = ShapeTemplate(self.size)

        return parameters

    def get_internal_structure(self):
        internals = OrderedDict()

        internals['Za'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['Zb'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['Ia'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['Ib'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['Fa'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['Fb'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['Oa'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['Ob'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['Ca'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['Cb'] = ShapeTemplate('T', 'B', self.size, context_size=1)

        return internals

    def _get_output_shapes(self):
        s = self.kwargs.get('size', self.in_shapes['default'].feature_size)
        if not isinstance(s, int):
            raise LayerValidationError('size must be int but was {}'.format(s))

        return {'default': ShapeTemplate('T', 'B', s, context_size=1)}

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        (Wz, Wi, Wf, Wo,
         Rz, Ri, Rf, Ro,
         bz, bi, bf, bo) = forward_buffers.parameters
        Za, Zb, Ia, Ib, Fa, Fb, Oa, Ob, Ca, Cb = forward_buffers.internals
        x = forward_buffers.inputs.default
        y = forward_buffers.outputs.default

        time_size, batch_size, in_size = x.shape

        for t in range(time_size):
            # Block input
            _h.dot_mm(x[t], Wz, Za[t], transb='T')
            _h.dot_add_mm(y[t - 1], Rz, Za[t])
            _h.add_mv(Za[t], bz, Za[t])
            self.act_func(Za[t], Zb[t])

            # Input Gate
            _h.dot_mm(x[t], Wi, Ia[t], transb='T')
            _h.dot_add_mm(y[t - 1], Ri, Ia[t])
            _h.add_mv(Ia[t], bi, Ia[t])
            _h.sigmoid(Ia[t], Ib[t])

            # Forget Gate
            _h.dot_mm(x[t], Wf, Fa[t], transb='T')
            _h.dot_add_mm(y[t - 1], Rf, Fa[t])
            _h.add_mv(Fa[t], bf, Fa[t])
            _h.sigmoid(Fa[t], Fb[t])

            # Cell
            _h.mult_tt(Ib[t], Zb[t], Ca[t])
            _h.mult_add_tt(Fb[t], Ca[t - 1], Ca[t])

            # Output Gate
            _h.dot_mm(x[t], Wo, Oa[t], transb='T')
            _h.dot_add_mm(y[t - 1], Ro, Oa[t])
            _h.add_mv(Oa[t], bo, Oa[t])
            _h.sigmoid(Oa[t], Ob[t])

            # Block output
            self.act_func(Ca[t], Cb[t])
            _h.mult_tt(Ob[t], Cb[t], y[t])

    def backward_pass(self, forward_buffers, backward_buffers):
        # prepare
        _h = self.handler
        (Wz, Wi, Wf, Wo,
         Rz, Ri, Rf, Ro,
         bz, bi, bf, bo) = forward_buffers.parameters
        (dWz, dWi, dWf, dWo,
         dRz, dRi, dRf, dRo,
         dbz, dbi, dbf, dbo) = backward_buffers.parameters

        Za, Zb, Ia, Ib, Fa, Fb, Oa, Ob, Ca, Cb = forward_buffers.internals
        dZa, dZb, dIa, dIb, dFa, dFb, dOa, dOb, dCa, dCb = backward_buffers.internals

        x = forward_buffers.inputs.default
        dx = backward_buffers.inputs.default
        y = forward_buffers.outputs.default
        deltas = backward_buffers.outputs.default

        dy = _h.allocate(y.shape)

        time_size, batch_size, in_size = x.shape
        for t in range(time_size - 1, -1, - 1):
            # cumulate recurrent deltas
            _h.copy_to(dy[t], deltas[t])
            _h.dot_add_mm(dIa[t + 1], Ri, dy[t], transb='T')
            _h.dot_add_mm(dFa[t + 1], Rf, dy[t], transb='T')
            _h.dot_add_mm(dOa[t + 1], Ro, dy[t], transb='T')
            _h.dot_add_mm(dZa[t + 1], Rz, dy[t], transb='T')

            # Output Gate
            _h.mult_tt(dy[t], Cb[t], dOb)
            _h.sigmoid_deriv(Oa[t], Ob[t], dOb[t], dOa[t])

            # Cell
            _h.mult_tt(dy[t], Ob[t], dCb)
            self.act_func_deriv(Ca[t], Cb[t], dCb[t], dCa[t])
            _h.mult_add_tt(dCa[t + 1], Fb[t + 1], dCa[t])

            # Forget Gate
            _h.mult_tt(dCa[t], Ca[t - 1], dFb)
            _h.sigmoid_deriv(Fa[t], Fb[t], dFb[t], dFa[t])

            # Input Gate
            _h.mult_tt(dCa[t], Zb[t], dIb)
            _h.sigmoid_deriv(Ia[t], Ib[t], dIb[t], dIa[t])

            # Block Input
            _h.mult_tt(dCa[t], Ib[t], dZb)
            self.act_func_deriv(Za[t], Zb[t], dZb[t], dZa[t])

            # Input Deltas
            _h.dot_add_mm(dIa[t], Wi, dx[t])
            _h.dot_add_mm(dFa[t], Wf, dx[t])
            _h.dot_add_mm(dOa[t], Wo, dx[t])
            _h.dot_add_mm(dZa[t], Wz, dx[t])

            # Gradients for the input weights
            _h.dot_add_mm(dIa[t], x[t], dWi, transa='T')
            _h.dot_add_mm(dFa[t], x[t], dWf, transa='T')
            _h.dot_add_mm(dOa[t], x[t], dWo, transa='T')
            _h.dot_add_mm(dZa[t], x[t], dWz, transa='T')

            # Gradient for the recurrent weights
            _h.dot_add_mm(y[t], dIa[t + 1], dRi, transa='T')
            _h.dot_add_mm(y[t], dFa[t + 1], dRf, transa='T')
            _h.dot_add_mm(y[t], dOa[t + 1], dRo, transa='T')
            _h.dot_add_mm(y[t], dZa[t + 1], dRz, transa='T')

            # biases
            bias_tmp = _h.allocate(dbz.shape)
            _h.sum_t(dIa[t], axis=0, out=bias_tmp)
            _h.add_tt(bias_tmp, dbi, dbi)
            _h.sum_t(dFa[t], axis=0, out=bias_tmp)
            _h.add_tt(bias_tmp, dbf, dbf)
            _h.sum_t(dOa[t], axis=0, out=bias_tmp)
            _h.add_tt(bias_tmp, dbo, dbo)
            _h.sum_t(dZa[t], axis=0, out=bias_tmp)
            _h.add_tt(bias_tmp, dbz, dbz)
