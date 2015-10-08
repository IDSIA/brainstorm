#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_time
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


def LstmPeephole(size, activation_function='tanh', name=None):
    return ConstructionWrapper.create('LstmPeephole',
                                      size=size,
                                      name=name,
                                      activation_function=activation_function)


class LstmPeepholeLayerImpl(LayerBaseImpl):
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
        super(LstmPeepholeLayerImpl, self).set_handler(new_handler)

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

    def get_parameter_structure(self):
        in_size = self.in_shapes['default'].feature_size
        
        parameters = OrderedDict()
        parameters['Wz'] = ShapeTemplate(self.size, in_size)
        parameters['Wi'] = ShapeTemplate(self.size, in_size)
        parameters['Wf'] = ShapeTemplate(self.size, in_size)
        parameters['Wo'] = ShapeTemplate(self.size, in_size)
        # Peephole connection matrices:
        parameters['Wci'] = ShapeTemplate(self.size, self.size)
        parameters['Wcf'] = ShapeTemplate(self.size, self.size)
        parameters['Wco'] = ShapeTemplate(self.size, self.size)

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

        internals['dZa'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dZb'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dIa'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dIb'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dFa'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dFb'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dOa'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dOb'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dCa'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dCb'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)

        return internals

    def _get_output_shapes(self):
        s = self.kwargs.get('size', self.in_shapes['default'].feature_size)
        if not isinstance(s, int):
            raise LayerValidationError('size must be int but was {}'.format(s))

        return {'default': ShapeTemplate('T', 'B', s, context_size=1)}

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        (Wz, Wi, Wf, Wo,
         Wci, Wcf, Wco,
         Rz, Ri, Rf, Ro,
         bz, bi, bf, bo) = buffers.parameters

        (Za, Zb, Ia, Ib, Fa, Fb, Oa, Ob, Ca, Cb,
         dZa, dZb, dIa, dIb, dFa, dFb, dOa, dOb, dCa, dCb) = buffers.internals
        x = buffers.inputs.default
        y = buffers.outputs.default

        time_size, batch_size, in_size = x.shape

        flat_x = flatten_time(x)
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
            _h.dot_add_mm(y[t - 1], Rz, Za[t])
            _h.add_mv(Za[t], bz.reshape((1, self.size)), Za[t])
            self.act_func(Za[t], Zb[t])

            # Input Gate
            _h.dot_add_mm(y[t - 1], Ri, Ia[t])
            _h.dot_add_mm(Ca[t - 1], Wci, Ia[t])  # ADDED PEEPHOLE CONNECTION
            _h.add_mv(Ia[t], bi.reshape((1, self.size)), Ia[t])
            _h.sigmoid(Ia[t], Ib[t])

            # Forget Gate
            _h.dot_add_mm(y[t - 1], Rf, Fa[t])
            _h.dot_add_mm(Ca[t - 1], Wcf, Fa[t])  # ADDED PEEPHOLE CONNECTION
            _h.add_mv(Fa[t], bf.reshape((1, self.size)), Fa[t])
            _h.sigmoid(Fa[t], Fb[t])

            # Cell
            _h.mult_tt(Ib[t], Zb[t], Ca[t])
            _h.mult_add_tt(Fb[t], Ca[t - 1], Ca[t])

            # Output Gate
            _h.dot_add_mm(y[t - 1], Ro, Oa[t])
            _h.dot_add_mm(Ca[t], Wco, Oa[t])  # ADDED PEEPHOLE CONNECTION
            _h.add_mv(Oa[t], bo.reshape((1, self.size)), Oa[t])
            _h.sigmoid(Oa[t], Ob[t])

            # Block output
            self.act_func(Ca[t], Cb[t])
            _h.mult_tt(Ob[t], Cb[t], y[t])

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        (Wz, Wi, Wf, Wo,
         Wci, Wcf, Wco,
         Rz, Ri, Rf, Ro,
         bz, bi, bf, bo) = buffers.parameters
        (dWz, dWi, dWf, dWo,
         dWci, dWcf, dWco,
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

        time_size, batch_size, in_size = x.shape
        for t in range(time_size - 1, -1, - 1):
            # Accumulate recurrent deltas
            _h.copy_to(dy[t], deltas[t])
            _h.dot_add_mm(dIa[t + 1], Ri, dy[t], transb=True)
            _h.dot_add_mm(dFa[t + 1], Rf, dy[t], transb=True)
            _h.dot_add_mm(dOa[t + 1], Ro, dy[t], transb=True)
            _h.dot_add_mm(dZa[t + 1], Rz, dy[t], transb=True)

            # Peephole connection part:
            _h.dot_mm(dIa[t + 1], Wci, dCa[t], transb=True)
            _h.dot_add_mm(dFa[t + 1], Wcf, dCa[t], transb=True)

            # Output Gate
            _h.mult_tt(dy[t], Cb[t], dOb[t])
            _h.sigmoid_deriv(Oa[t], Ob[t], dOb[t], dOa[t])
            # Peephole connection part:
            _h.dot_add_mm(dOa[t], Wco, dCa[t], transb=True)

            # Cell
            _h.mult_tt(dy[t], Ob[t], dCb[t])
            self.act_func_deriv(Ca[t], Cb[t], dCb[t], dCb[t])  # Slight change to accommodate peephole
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

        # Same as for standard LSTM:
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
        flat_cell = flatten_time(Ca[:-2])
        flat_cell2 = flatten_time(Ca[:-1])

        # Peephole connection output weight:
        _h.dot_add_mm(flat_cell2, flat_dOa, dWco, transa=True)

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

        # Peephole connection weights:
        _h.dot_add_mm(flat_cell, flat_dIa, dWci, transa=True)
        _h.dot_add_mm(flat_cell, flat_dFa, dWcf, transa=True)

        _h.dot_add_mm(dCa[-1], dIa[0], dWci, transa=True)
        _h.dot_add_mm(dCa[-1], dFa[0], dWcf, transa=True)