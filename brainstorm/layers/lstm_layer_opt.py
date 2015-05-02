#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class LstmOptLayerImpl(LayerBaseImpl):
    expected_kwargs = {'size', 'activation_function'}

    def __init__(self, name, in_shapes, incoming_connections,
                 outgoing_connections, **kwargs):
        super(LstmOptLayerImpl, self).__init__(
            name, in_shapes, incoming_connections, outgoing_connections,
            **kwargs)
        self.act_func = lambda x, y: None
        self.act_func_deriv = lambda x, y, dy, dx: None
        self.kwargs = kwargs

    def set_handler(self, new_handler):
        super(LstmOptLayerImpl, self).set_handler(new_handler)

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

    def get_parameter_structure(self):
        in_size = self.in_shapes['default'].feature_size
        out_size = self.out_shapes['default'].feature_size

        parameters = OrderedDict()
        parameters['Wz'] = ShapeTemplate(in_size, out_size)
        parameters['Wi'] = ShapeTemplate(in_size, out_size)
        parameters['Wf'] = ShapeTemplate(in_size, out_size)
        parameters['Wo'] = ShapeTemplate(in_size, out_size)

        parameters['Rz'] = ShapeTemplate(out_size, out_size)
        parameters['Ri'] = ShapeTemplate(out_size, out_size)
        parameters['Rf'] = ShapeTemplate(out_size, out_size)
        parameters['Ro'] = ShapeTemplate(out_size, out_size)

        parameters['bz'] = ShapeTemplate(out_size)
        parameters['bi'] = ShapeTemplate(out_size)
        parameters['bf'] = ShapeTemplate(out_size)
        parameters['bo'] = ShapeTemplate(out_size)

        return parameters

    def get_internal_structure(self):
        out_size = self.out_shapes['default'].feature_size
        internals = OrderedDict()

        internals['Z'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['I'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['F'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['O'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['Ca'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['Cb'] = ShapeTemplate('T', 'B', out_size, context_size=1)

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
        Z, I, F, O, Ca, Cb = forward_buffers.internals
        x = forward_buffers.inputs.default
        y = forward_buffers.outputs.default

        time_size, batch_size, in_size = x.shape
        flat_size = time_size * batch_size
        flat_x = x.reshape((flat_size, in_size))

        flat_Zb = Z[:-1].reshape((flat_size, Z.shape[2]))
        flat_Ib = I[:-1].reshape((flat_size, I.shape[2]))
        flat_Fb = F[:-1].reshape((flat_size, F.shape[2]))
        flat_Ob = O[:-1].reshape((flat_size, O.shape[2]))

        _h.dot_mm(flat_x, Wz, flat_Zb)
        _h.dot_mm(flat_x, Wi, flat_Ib)
        _h.dot_mm(flat_x, Wf, flat_Fb)
        _h.dot_mm(flat_x, Wo, flat_Ob)

        _h.add_mv(flat_Zb, bz, flat_Zb)
        _h.add_mv(flat_Ib, bi, flat_Ib)
        _h.add_mv(flat_Fb, bf, flat_Fb)
        _h.add_mv(flat_Ob, bo, flat_Ob)

        for t in range(time_size):
            _h.dot_add_mm(y[t - 1], Rz, Z[t])
            _h.dot_add_mm(y[t - 1], Ri, I[t])
            _h.dot_add_mm(y[t - 1], Rf, F[t])
            _h.dot_add_mm(y[t - 1], Ro, O[t])

            # Activations for Z I F O
            self.act_func(Z[t], Z[t])
            _h.sigmoid(I[t], I[t])
            _h.sigmoid(F[t], F[t])
            _h.sigmoid(O[t], O[t])

            # Cell
            _h.mult_tt(I[t], Z[t], Ca[t])
            _h.mult_add_tt(F[t], Ca[t - 1], Ca[t])

            # Block output
            self.act_func(Ca[t], Cb[t])
            _h.mult_tt(O[t], Cb[t], y[t])

    def backward_pass(self, forward_buffers, backward_buffers):
        # prepare
        _h = self.handler
        (Wz, Wi, Wf, Wo,
         Rz, Ri, Rf, Ro,
         bz, bi, bf, bo) = forward_buffers.parameters
        (dWz, dWi, dWf, dWo,
         dRz, dRi, dRf, dRo,
         dbz, dbi, dbf, dbo) = backward_buffers.parameters

        Z, I, F, O, Ca, Cb = forward_buffers.internals
        dZb, dIb, dFb, dOb, dCa, dCb = backward_buffers.internals

        x = forward_buffers.inputs.default
        dx = backward_buffers.inputs.default
        y = forward_buffers.outputs.default
        deltas = backward_buffers.outputs.default

        dy = _h.allocate(y.shape)

        time_size, batch_size, in_size = x.shape
        flat_size = time_size * batch_size
        flat_dx = dx.reshape((flat_size, in_size))
        flat_x = x.reshape((flat_size, in_size))
        flat_dZb = dZb[:-1].reshape((flat_size, Z.shape[2]))
        flat_dIb = dIb[:-1].reshape((flat_size, I.shape[2]))
        flat_dFb = dFb[:-1].reshape((flat_size, F.shape[2]))
        flat_dOb = dOb[:-1].reshape((flat_size, O.shape[2]))

        _h.copy_to(dy, deltas)

        for t in range(time_size - 1, -1, - 1):
            # cumulate recurrent deltas
            _h.dot_add_mm(dZb[t + 1], Rz, dy[t], transb='T')
            _h.dot_add_mm(dIb[t + 1], Ri, dy[t], transb='T')
            _h.dot_add_mm(dFb[t + 1], Rf, dy[t], transb='T')
            _h.dot_add_mm(dOb[t + 1], Ro, dy[t], transb='T')

            # Cell
            _h.mult_tt(dy[t], O[t], dCb[t])
            self.act_func_deriv(Ca[t], Cb[t], dCb[t], dCa[t])
            _h.mult_add_tt(dCa[t + 1], F[t + 1], dCa[t])

            # Block Input and Gates
            _h.mult_tt(dCa[t], I[t], dZb[t])
            _h.mult_tt(dCa[t], Z[t], dIb[t])
            _h.mult_tt(dCa[t], Ca[t - 1], dFb[t])
            _h.mult_tt(dy[t], Cb[t], dOb[t])

            # Activation functions
            self.act_func_deriv(None, Z[t], dZb[t], dZb[t])
            _h.sigmoid_deriv(None, I[t], dIb[t], dIb[t])
            _h.sigmoid_deriv(None, F[t], dFb[t], dFb[t])
            _h.sigmoid_deriv(None, O[t], dOb[t], dOb[t])


        flat_y = y[:-2].reshape(((time_size - 1) * batch_size, y.shape[2]))
        # Gradient for the recurrent weights
        _h.dot_add_mm(flat_y, flat_dZb[batch_size:], dRz, transa='T')
        _h.dot_add_mm(flat_y, flat_dIb[batch_size:], dRi, transa='T')
        _h.dot_add_mm(flat_y, flat_dFb[batch_size:], dRf, transa='T')
        _h.dot_add_mm(flat_y, flat_dOb[batch_size:], dRo, transa='T')
        _h.dot_add_mm(y[-1], dZb[0], dRz, transa='T')
        _h.dot_add_mm(y[-1], dIb[0], dRi, transa='T')
        _h.dot_add_mm(y[-1], dFb[0], dRf, transa='T')
        _h.dot_add_mm(y[-1], dOb[0], dRo, transa='T')

        # biases
        bias_tmp = _h.allocate(dbz.shape)
        _h.sum_t(flat_dZb, axis=0, out=bias_tmp)
        _h.add_tt(bias_tmp, dbz, dbz)
        _h.sum_t(flat_dIb, axis=0, out=bias_tmp)
        _h.add_tt(bias_tmp, dbi, dbi)
        _h.sum_t(flat_dFb, axis=0, out=bias_tmp)
        _h.add_tt(bias_tmp, dbf, dbf)
        _h.sum_t(flat_dOb, axis=0, out=bias_tmp)
        _h.add_tt(bias_tmp, dbo, dbo)

        # Gradients for the input weights
        _h.dot_add_mm(flat_x, flat_dZb, dWz, transa='T')
        _h.dot_add_mm(flat_x, flat_dIb, dWi, transa='T')
        _h.dot_add_mm(flat_x, flat_dFb, dWf, transa='T')
        _h.dot_add_mm(flat_x, flat_dOb, dWo, transa='T')

        # Input Deltas
        _h.dot_add_mm(flat_dZb, Wz, flat_dx, transb='T')
        _h.dot_add_mm(flat_dIb, Wi, flat_dx, transb='T')
        _h.dot_add_mm(flat_dFb, Wf, flat_dx, transb='T')
        _h.dot_add_mm(flat_dOb, Wo, flat_dx, transb='T')
