#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_time, \
    flatten_time_and_features


def Lstm(size, activation='tanh', name=None):
    """Create an LSTM layer."""
    return ConstructionWrapper.create(LstmLayerImpl, size=size,
                                      name=name, activation=activation)


class LstmLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'size', 'activation'}

    def setup(self, kwargs, in_shapes):
        self.activation = kwargs.get('activation', 'tanh')
        in_size = in_shapes['default'].feature_size
        self.size = kwargs.get('size', in_size)
        if not isinstance(self.size, int):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', self.size,
                                             context_size=1)

        parameters = OrderedDict()
        parameters['Wz'] = BufferStructure(self.size, in_size)
        parameters['Wi'] = BufferStructure(self.size, in_size)
        parameters['Wf'] = BufferStructure(self.size, in_size)
        parameters['Wo'] = BufferStructure(self.size, in_size)

        parameters['pi'] = BufferStructure(1, self.size)
        parameters['pf'] = BufferStructure(1, self.size)
        parameters['po'] = BufferStructure(1, self.size)

        parameters['Rz'] = BufferStructure(self.size, self.size)
        parameters['Ri'] = BufferStructure(self.size, self.size)
        parameters['Rf'] = BufferStructure(self.size, self.size)
        parameters['Ro'] = BufferStructure(self.size, self.size)
        parameters['bz'] = BufferStructure(self.size)
        parameters['bi'] = BufferStructure(self.size)
        parameters['bf'] = BufferStructure(self.size)
        parameters['bo'] = BufferStructure(self.size)

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

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        (Wz, Wi, Wf, Wo,
         pi, pf, po,
         Rz, Ri, Rf, Ro,
         bz, bi, bf, bo) = buffers.parameters

        (Za, Zb, Ia, Ib, Fa, Fb, Oa, Ob, Ca, Cb,
         dZa, dZb, dIa, dIb, dFa, dFb, dOa, dOb, dCa, dCb) = buffers.internals
        x = buffers.inputs.default
        y = buffers.outputs.default

        time_size, batch_size = x.shape[0], x.shape[1]

        flat_x = flatten_time_and_features(x)
        flat_Za = flatten_time(Za[:-1])
        flat_Ia = flatten_time(Ia[:-1])
        flat_Fa = flatten_time(Fa[:-1])
        flat_Oa = flatten_time(Oa[:-1])
        _h.dot_mm(flat_x, Wz, flat_Za, transb=True)
        _h.dot_mm(flat_x, Wi, flat_Ia, transb=True)
        _h.dot_mm(flat_x, Wf, flat_Fa, transb=True)
        _h.dot_mm(flat_x, Wo, flat_Oa, transb=True)

        for t in range(time_size):
            # Block input
            _h.dot_add_mm(y[t - 1], Rz, Za[t], transb=True)
            _h.add_mv(Za[t], bz.reshape((1, self.size)), Za[t])
            _h.act_func[self.activation](Za[t], Zb[t])

            # Input Gate
            _h.dot_add_mm(y[t - 1], Ri, Ia[t], transb=True)
            _h.mult_add_mv(Ca[t - 1], pi, Ia[t])
            _h.add_mv(Ia[t], bi.reshape((1, self.size)), Ia[t])
            _h.sigmoid(Ia[t], Ib[t])

            # Forget Gate
            _h.dot_add_mm(y[t - 1], Rf, Fa[t], transb=True)
            _h.mult_add_mv(Ca[t - 1], pf, Fa[t])
            _h.add_mv(Fa[t], bf.reshape((1, self.size)), Fa[t])
            _h.sigmoid(Fa[t], Fb[t])

            # Cell
            _h.mult_tt(Ib[t], Zb[t], Ca[t])
            _h.mult_add_tt(Fb[t], Ca[t - 1], Ca[t])

            # Output Gate
            _h.dot_add_mm(y[t - 1], Ro, Oa[t], transb=True)
            _h.mult_add_mv(Ca[t], po, Oa[t])
            _h.add_mv(Oa[t], bo.reshape((1, self.size)), Oa[t])
            _h.sigmoid(Oa[t], Ob[t])

            # Block output
            _h.act_func[self.activation](Ca[t], Cb[t])
            _h.mult_tt(Ob[t], Cb[t], y[t])

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        (Wz, Wi, Wf, Wo,
         pi, pf, po,
         Rz, Ri, Rf, Ro,
         bz, bi, bf, bo) = buffers.parameters
        (dWz, dWi, dWf, dWo,
         dpi, dpf, dpo,
         dRz, dRi, dRf, dRo,
         dbz, dbi, dbf, dbo) = buffers.gradients

        (Za, Zb, Ia, Ib, Fa, Fb, Oa, Ob, Ca, Cb,
         dZa, dZb, dIa, dIb, dFa, dFb, dOa, dOb, dCa, dCb) = buffers.internals

        x = buffers.inputs.default
        dx = buffers.input_deltas.default
        y = buffers.outputs.default
        deltas = buffers.output_deltas.default

        dy = _h.allocate(y.shape)
        _h.fill(dCa, 0.0)

        time_size, batch_size = x.shape[0], x.shape[1]
        for t in range(time_size - 1, -1, - 1):
            # Accumulate recurrent deltas
            _h.copy_to(deltas[t], dy[t])
            _h.dot_add_mm(dIa[t + 1], Ri, dy[t])
            _h.dot_add_mm(dFa[t + 1], Rf, dy[t])
            _h.dot_add_mm(dOa[t + 1], Ro, dy[t])
            _h.dot_add_mm(dZa[t + 1], Rz, dy[t])

            # Peephole connection part:
            _h.mult_add_mv(dIa[t + 1], pi, dCa[t])
            _h.mult_add_mv(dFa[t + 1], pf, dCa[t])

            # Output Gate
            _h.mult_tt(dy[t], Cb[t], dOb[t])
            _h.sigmoid_deriv(Oa[t], Ob[t], dOb[t], dOa[t])
            # Peephole connection
            _h.mult_add_mv(dOa[t], po, dCa[t])

            # Cell
            _h.mult_tt(dy[t], Ob[t], dCb[t])
            _h.act_func_deriv[self.activation](Ca[t], Cb[t], dCb[t], dCb[t])
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
            _h.act_func_deriv[self.activation](Za[t], Zb[t], dZb[t], dZa[t])

        flat_inputs = flatten_time_and_features(x)
        flat_dinputs = flatten_time_and_features(dx)

        flat_dIa = flatten_time(dIa[:-1])
        flat_dFa = flatten_time(dFa[:-1])
        flat_dOa = flatten_time(dOa[:-1])
        flat_dZa = flatten_time(dZa[:-1])

        # Calculate in_deltas and gradients
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
        flat_cell = flatten_time(Ca[:-2])
        flat_cell2 = flatten_time(Ca[:-1])

        dWco_tmp = _h.allocate(flat_cell2.shape)
        dWc_tmp = _h.allocate(dpo.shape)

        # Output gate Peephole
        _h.mult_tt(flat_cell2, flat_dOa, dWco_tmp)
        _h.sum_t(dWco_tmp, axis=0, out=dWc_tmp)
        _h.add_tt(dpo, dWc_tmp, dpo)

        flat_dIa = flatten_time(dIa[1:-1])
        flat_dFa = flatten_time(dFa[1:-1])
        flat_dOa = flatten_time(dOa[1:-1])
        flat_dZa = flatten_time(dZa[1:-1])

        _h.dot_add_mm(flat_dIa, flat_outputs, dRi, transa=True)
        _h.dot_add_mm(flat_dFa, flat_outputs, dRf, transa=True)
        _h.dot_add_mm(flat_dOa, flat_outputs, dRo, transa=True)
        _h.dot_add_mm(flat_dZa, flat_outputs, dRz, transa=True)

        _h.dot_add_mm(dIa[0], dy[-1], dRi, transa=True)
        _h.dot_add_mm(dFa[0], dy[-1], dRf, transa=True)
        _h.dot_add_mm(dOa[0], dy[-1], dRo, transa=True)
        _h.dot_add_mm(dZa[0], dy[-1], dRz, transa=True)

        # Other Peephole connections
        dWcif_tmp = _h.allocate(flat_cell.shape)
        _h.mult_tt(flat_cell, flat_dIa, dWcif_tmp)
        _h.sum_t(dWcif_tmp, axis=0, out=dWc_tmp)
        _h.add_tt(dpi, dWc_tmp, dpi)
        _h.mult_tt(flat_cell, flat_dFa, dWcif_tmp)
        _h.sum_t(dWcif_tmp, axis=0, out=dWc_tmp)
        _h.add_tt(dpf, dWc_tmp, dpf)

        dWcif_tmp = _h.allocate(dIa[0].shape)
        _h.mult_tt(dCa[-1], dIa[0], dWcif_tmp)
        _h.sum_t(dWcif_tmp, axis=0, out=dWc_tmp)
        _h.add_tt(dpi, dWc_tmp, dpi)
        _h.mult_tt(dCa[-1], dIa[0], dWcif_tmp)
        _h.sum_t(dWcif_tmp, axis=0, out=dWc_tmp)
        _h.add_tt(dpf, dWc_tmp, dpf)
