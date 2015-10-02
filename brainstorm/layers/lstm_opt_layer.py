#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.layers.base_layer import BaseLayerImpl
from brainstorm.structure.buffer_structure import (StructureTemplate,
                                                   BufferStructure)


def LstmOpt(size, activation='tanh', name=None):
    """Create an LSTMOpt layer."""
    return ConstructionWrapper.create('LstmOpt', size=size, name=name,
                                      activation=activation)


# noinspection PyPep8Naming
class LstmOptLayerImpl(BaseLayerImpl):
    
    expected_inputs = {'default': StructureTemplate('T', 'B', 'F')}
    expected_kwargs = {'size', 'activation'}

    def setup(self, kwargs, in_shapes):
        self.act_func = lambda x, y: None
        self.act_func_deriv = lambda x, y, dy, dx: None
        in_size = in_shapes['default'].feature_size
        self.size = kwargs.get('size', in_size)

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', self.size,
                                             context_size=1)

        parameters = OrderedDict()
        parameters['W'] = BufferStructure(self.size * 4, in_size)
        parameters['R'] = BufferStructure(self.size * 4, self.size)
        parameters['b'] = BufferStructure(self.size * 4)

        internals = OrderedDict()
        internals['S'] = BufferStructure('T', 'B', self.size * 4,
                                         context_size=1)
        internals['Ca'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['Cb'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['dS'] = BufferStructure('T', 'B', self.size * 4,
                                          context_size=1,
                                          is_backward_only=True)
        internals['dCa'] = BufferStructure('T', 'B', self.size, context_size=1,
                                           is_backward_only=True)
        internals['dCb'] = BufferStructure('T', 'B', self.size, context_size=1,
                                           is_backward_only=True)
        return outputs, parameters, internals

    def set_handler(self, new_handler):
        super(LstmOptLayerImpl, self).set_handler(new_handler)

        # Assign act_func and act_func_derivs
        activations = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(y, x),
                       lambda x, y, dy, dx: self.handler.copy_to(dx, dy)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activations[
            self.kwargs.get('activation', 'tanh')]

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, R, b = buffers.parameters
        S, Ca, Cb, dS, dCa, dCb = buffers.internals
        x = buffers.inputs.default
        y = buffers.outputs.default

        time_size, batch_size, in_size = x.shape
        out_size = y.shape[2]
        flat_size = time_size * batch_size
        flat_x = x.reshape((flat_size, in_size))

        flat_S = S[:-1].reshape((flat_size, S.shape[2]))

        Z = S[:, :, :out_size]
        gates = S[:, :, out_size:]
        I = S[:, :, out_size:2 * out_size]
        F = S[:, :, out_size * 2:out_size * 3]
        O = S[:, :, out_size * 3:]

        _h.dot_mm(flat_x, W, flat_S, transb=True)  # all inputs times weights
        _h.add_mv(flat_S, b.reshape((1, b.shape[0])), flat_S)  # all biases

        for t in range(time_size):
            # Recurrent Connections
            _h.dot_add_mm(y[t - 1], R, S[t], transb=True)

            # Activations for Z and gates
            self.act_func(Z[t], Z[t])
            _h.sigmoid(gates[t], gates[t])

            # Cell
            _h.mult_tt(I[t], Z[t], Ca[t])
            _h.mult_add_tt(F[t], Ca[t - 1], Ca[t])

            # Block output
            self.act_func(Ca[t], Cb[t])
            _h.mult_tt(O[t], Cb[t], y[t])

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        W, R, b = buffers.parameters
        dW, dR, db = buffers.gradients

        S, Ca, Cb, dS, dCa, dCb = buffers.internals

        x = buffers.inputs.default
        dx = buffers.input_deltas.default
        y = buffers.outputs.default
        deltas = buffers.output_deltas.default

        dy = _h.allocate(y.shape)

        time_size, batch_size, in_size = x.shape
        out_size = y.shape[2]
        flat_size = time_size * batch_size
        flat_dx = dx.reshape((flat_size, in_size))
        flat_x = x.reshape((flat_size, in_size))
        flat_dS = dS[:-1].reshape((flat_size, S.shape[2]))

        gates = S[:, :, out_size:]
        Z = S[:, :, :out_size]
        I = S[:, :, out_size:2 * out_size]
        F = S[:, :, out_size * 2:out_size * 3]
        O = S[:, :, out_size * 3:]

        dgates = dS[:, :, out_size:]
        dZ = dS[:, :, :out_size]
        dI = dS[:, :, out_size:2 * out_size]
        dF = dS[:, :, out_size * 2:out_size * 3]
        dO = dS[:, :, out_size * 3:]

        _h.copy_to(dy, deltas)

        for t in range(time_size - 1, -1, - 1):
            # cumulate recurrent deltas
            _h.dot_add_mm(dS[t + 1], R, dy[t])

            # Cell
            _h.mult_tt(dy[t], O[t], dCb[t])
            self.act_func_deriv(Ca[t], Cb[t], dCb[t], dCa[t])
            _h.mult_add_tt(dCa[t + 1], F[t + 1], dCa[t])

            # Block Input and Gates
            _h.mult_tt(dCa[t], I[t], dZ[t])
            _h.mult_tt(dCa[t], Z[t], dI[t])
            _h.mult_tt(dCa[t], Ca[t - 1], dF[t])
            _h.mult_tt(dy[t], Cb[t], dO[t])

            # Activation functions
            self.act_func_deriv(None, Z[t], dZ[t], dZ[t])
            _h.sigmoid_deriv(None, gates[t], dgates[t], dgates[t])

        # Gradient for the recurrent weights
        flat_y = y[:-2].reshape(((time_size - 1) * batch_size, y.shape[2]))
        _h.dot_add_mm(flat_dS[batch_size:], flat_y, dR, transa=True)
        _h.dot_add_mm(dS[0], y[-1], dR, transa=True)

        # biases
        bias_tmp = _h.allocate(db.shape)
        _h.sum_t(flat_dS, axis=0, out=bias_tmp)
        _h.add_tt(bias_tmp, db, db)

        # Gradients for the input weights
        _h.dot_add_mm(flat_dS, flat_x, dW, transa=True)

        # Input Deltas
        _h.dot_add_mm(flat_dS, W, flat_dx)
