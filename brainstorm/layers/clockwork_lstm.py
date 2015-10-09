#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_time
from brainstorm.layers.base_layer import BaseLayerImpl
from brainstorm.structure.buffer_structure import BufferStructure, StructureTemplate

def ClockworkLstm(size, timing, activation='tanh', name=None):
    return ConstructionWrapper.create('ClockworkLstm',
                                      size=size,
                                      timing=timing,
                                      name=name,
                                      activation=activation)

class ClockworkLstmLayerImpl(BaseLayerImpl):
    expected_kwargs = {'size', 'timing', 'activation'}
    expected_inputs = {'default': StructureTemplate('T', 'B', 'F')}

    def setup(self, kwargs, in_shapes):
        self.act_func = None
        self.act_func_deriv = None
        self.size = kwargs.get('size', in_shapes['default'].feature_size)

        if not isinstance(self.size, int):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))

        in_size = in_shapes['default'].feature_size

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', self.size, context_size=1)

        parameters = OrderedDict()

        parameters['Wz'] = BufferStructure(self.size, in_size)
        parameters['Wi'] = BufferStructure(self.size, in_size)
        parameters['Wf'] = BufferStructure(self.size, in_size)
        parameters['Wo'] = BufferStructure(self.size, in_size)

        parameters['Rz'] = BufferStructure(self.size, self.size)
        parameters['Ri'] = BufferStructure(self.size, self.size)
        parameters['Rf'] = BufferStructure(self.size, self.size)
        parameters['Ro'] = BufferStructure(self.size, self.size)

        parameters['bz'] = BufferStructure(self.size)
        parameters['bi'] = BufferStructure(self.size)
        parameters['bf'] = BufferStructure(self.size)
        parameters['bo'] = BufferStructure(self.size)

        parameters['timing'] = BufferStructure(self.size)

        internals = OrderedDict()
        internals['Za'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['Zb'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['Ia'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['Ib'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['Fa'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['Fb'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['Oa'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['Ob'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['Ca'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['Cb'] = BufferStructure('T', 'B', self.size, context_size=1)

        internals['dZa'] = BufferStructure('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dZb'] = BufferStructure('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dIa'] = BufferStructure('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dIb'] = BufferStructure('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dFa'] = BufferStructure('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dFb'] = BufferStructure('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dOa'] = BufferStructure('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dOb'] = BufferStructure('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dCa'] = BufferStructure('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dCb'] = BufferStructure('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)

        return outputs, parameters, internals

    def _setup_hyperparameters(self):
        self.act_func = None
        self.act_func_deriv = None
        self.size = self.kwargs.get('size',
                                    self.in_shapes['default'].feature_size)

        if not isinstance(self.size, int):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))

    def set_handler(self, new_handler):
        super(ClockworkLstmLayerImpl, self).set_handler(new_handler)

        # Assign act_func and act_func_derivs
        activation_functions = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(y, x),
                       lambda x, y, dy, dx: self.handler.copy_to(dx, dy)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activation_functions[
            self.kwargs.get('activation_function', 'tanh')]


    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        (Wz, Wi, Wf, Wo,
         Rz, Ri, Rf, Ro,
         bz, bi, bf, bo,
         timing) = buffers.parameters
        (Za, Zb,
         Ia, Ib,
         Fa, Fb,
         Oa, Ob,
         Ca, Cb,
         dZa, dZb,
         dIa, dIb,
         dFa, dFb,
         dOa, dOb,
         dCa, dCb) = buffers.internals
        x = buffers.inputs.default
        y = buffers.outputs.default
        time_size, batch_size, in_size = x.shape

        feature_size = timing.shape[0]

        flat_x = flatten_time(x)
        flat_Za = flatten_time(Za[:-1])
        flat_Ia = flatten_time(Ia[:-1])
        flat_Fa = flatten_time(Fa[:-1])
        flat_Oa = flatten_time(Oa[:-1])
        _h.dot_mm(flat_x, Wz, flat_Za, transb=True)
        _h.dot_mm(flat_x, Wi, flat_Ia, transb=True)
        _h.dot_mm(flat_x, Wf, flat_Fa, transb=True)
        _h.dot_mm(flat_x, Wo, flat_Oa, transb=True)

        # Temporary variable to be filled with the current value of time t
        tmp = _h.zeros(timing.shape)

        for t in range(time_size):

            # Block input
            _h.dot_add_mm(y[t - 1], Rz, Za[t])
            _h.add_mv(Za[t], bz.reshape((1, self.size)), Za[t])
            self.act_func(Za[t], Zb[t])

            # Input Gate
            _h.dot_add_mm(y[t - 1], Ri, Ia[t])
            _h.add_mv(Ia[t], bi.reshape((1, self.size)), Ia[t])
            _h.sigmoid(Ia[t], Ib[t])

            # Forget Gate
            _h.dot_add_mm(y[t - 1], Rf, Fa[t])
            _h.add_mv(Fa[t], bf.reshape((1, self.size)), Fa[t])
            _h.sigmoid(Fa[t], Fb[t])

            # Cell
            _h.mult_tt(Ib[t], Zb[t], Ca[t])
            _h.mult_add_tt(Fb[t], Ca[t - 1], Ca[t])

            # Output Gate
            _h.dot_add_mm(y[t - 1], Ro, Oa[t])
            _h.add_mv(Oa[t], bo.reshape((1, self.size)), Oa[t])
            _h.sigmoid(Oa[t], Ob[t])

            # Block output
            self.act_func(Ca[t], Cb[t])
            _h.mult_tt(Ob[t], Cb[t], y[t])

            if t > 0:
                _h.fill(tmp, t)
                _h.modulo_mm(tmp, timing, tmp)
            # -----------------------------------
            # Clockwork part: Undo updates:
            # -----------------------------------
            # Reset Cell
                _h.clw_undo_update(batch_size, feature_size, tmp, Ca[t-1], Ca[t])
            # Reset Block output
                _h.clw_undo_update(batch_size, feature_size, tmp, y[t-1], y[t])

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler

        (dWz, dWi, dWf, dWo,
         dRz, dRi, dRf, dRo,
         dbz, dbi, dbf, dbo,
         dtiming) = buffers.gradients

        (Wz, Wi, Wf, Wo,
        Rz, Ri, Rf, Ro,
        bz, bi, bf, bo,
        timing) = buffers.parameters

        (Za, Zb,
        Ia, Ib,
        Fa, Fb,
        Oa, Ob,
        Ca, Cb,
        dZa, dZb,
        dIa, dIb,
        dFa, dFb,
        dOa, dOb,
        dCa, dCb) = buffers.internals

        x = buffers.inputs.default
        dx = buffers.input_deltas.default
        y = buffers.outputs.default
        deltas = buffers.output_deltas.default

        dy = _h.allocate(y.shape)

        feature_size = timing.shape[0]

        time_size, batch_size, in_size = x.shape

        # Temporary variable to be filled with the current value of time t
        tmp = _h.zeros(timing.shape)
        _h.fill(dCa, 0.0)  # zero initialization. important for backward pass

        for t in range(time_size - 1, -1, - 1):
            # Cumulate recurrent deltas
            _h.add_tt(dy[t], deltas[t], dy[t])

            _h.fill(tmp, t)
            _h.modulo_mm(tmp, timing, tmp)

            _h.dot_add_mm(dIa[t + 1], Ri, dy[t], transb=True)
            _h.dot_add_mm(dFa[t + 1], Rf, dy[t], transb=True)
            _h.dot_add_mm(dOa[t + 1], Ro, dy[t], transb=True)
            _h.dot_add_mm(dZa[t + 1], Rz, dy[t], transb=True)

            # Output Gate
            _h.mult_tt(dy[t], Cb[t], dOb[t])
            _h.sigmoid_deriv(Oa[t], Ob[t], dOb[t], dOa[t])

            # Cell
            _h.mult_tt(dy[t], Ob[t], dCb[t])
            self.act_func_deriv(Ca[t], Cb[t], dCb[t], dCb[t])  # Important change to standard LSTM
            _h.clw_set_inactive_to_zero(batch_size, feature_size, tmp, dCb[t])
            _h.add_tt(dCa[t], dCb[t], dCa[t])
            _h.mult_add_tt(dCa[t + 1], Fb[t + 1], dCa[t])

            # Forget Gate
            _h.mult_tt(dCa[t], Ca[t - 1], dFb[t])
            _h.sigmoid_deriv(Fa[t], Fb[t], dFb[t], dFa[t])

            # Input Gate
            _h.mult_tt(dCa[t], Zb[t], dIb[t])
            _h.sigmoid_deriv(Ia[t], Ib[t], dIb[t], dIa[t])

            # Block Input
            _h.mult_tt(dCa[t], Ib[t], dZb[t])
            self.act_func_deriv(Za[t], Zb[t], dZb[t], dZa[t])

            # Copy over the error from previous inactive nodes:
            _h.clw_copy_add_act_of_inactive(batch_size, feature_size, tmp, dy[t], dy[t-1])
            _h.clw_undo_update(batch_size, feature_size, tmp, dCa[t], dCa[t-1])
            # Undo updates to inactive nodes:
            _h.clw_set_inactive_to_zero(batch_size, feature_size, tmp, dIa[t])
            _h.clw_set_inactive_to_zero(batch_size, feature_size, tmp, dFa[t])
            _h.clw_set_inactive_to_zero(batch_size, feature_size, tmp, dOa[t])
            _h.clw_set_inactive_to_zero(batch_size, feature_size, tmp, dZa[t])
            _h.clw_set_inactive_to_zero(batch_size, feature_size, tmp, Fb[t])

        # Same as for standard RNN:
        flat_inputs = flatten_time(x)
        flat_dinputs = flatten_time(dx)

        flat_dIa = flatten_time(dIa[:-1])
        flat_dFa = flatten_time(dFa[:-1])
        flat_dOa = flatten_time(dOa[:-1])
        flat_dZa = flatten_time(dZa[:-1])

        # calculate in_deltas and gradients
        _h.dot_add_mm(flat_dIa, Wi, flat_dinputs)
        _h.dot_add_mm(flat_dFa, Wf, flat_dinputs)
        _h.dot_add_mm(flat_dOa, Wo, flat_dinputs)
        _h.dot_add_mm(flat_dZa, Wz, flat_dinputs)

        _h.dot_add_mm(flat_dIa, flat_inputs, dWi, transa=True)
        _h.dot_add_mm(flat_dFa, flat_inputs, dWf, transa=True)
        _h.dot_add_mm(flat_dOa, flat_inputs, dWo, transa=True)
        _h.dot_add_mm(flat_dZa, flat_inputs, dWz, transa=True)

        dbias_tmp = _h.allocate(dbz.shape)
        _h.sum_t(flat_dIa, axis=0, out=dbias_tmp)
        _h.add_tt(dbi, dbias_tmp, dbi)
        _h.sum_t(flat_dFa, axis=0, out=dbias_tmp)
        _h.add_tt(dbf, dbias_tmp, dbf)
        _h.sum_t(flat_dOa, axis=0, out=dbias_tmp)
        _h.add_tt(dbo, dbias_tmp, dbo)
        _h.sum_t(flat_dZa, axis=0, out=dbias_tmp)
        _h.add_tt(dbz, dbias_tmp, dbz)

        flat_outputs = flatten_time(y[:-2])
        flat_dIa = flatten_time(dIa[1:-1])
        flat_dFa = flatten_time(dFa[1:-1])
        flat_dOa = flatten_time(dOa[1:-1])
        flat_dZa = flatten_time(dZa[1:-1])

        _h.dot_add_mm(flat_outputs, flat_dIa, dRi, transa=True)
        _h.dot_add_mm(flat_outputs, flat_dFa, dRf, transa=True)
        _h.dot_add_mm(flat_outputs, flat_dOa, dRo, transa=True)
        _h.dot_add_mm(flat_outputs, flat_dZa, dRz, transa=True)

        _h.dot_add_mm(dy[-1], dIa[0], dRi, transa=True)
        _h.dot_add_mm(dy[-1], dFa[0], dRf, transa=True)
        _h.dot_add_mm(dy[-1], dOa[0], dRo, transa=True)
        _h.dot_add_mm(dy[-1], dZa[0], dRz, transa=True)