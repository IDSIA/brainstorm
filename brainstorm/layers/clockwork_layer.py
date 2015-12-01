#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_time, \
    flatten_time_and_features
from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import BufferStructure, \
    StructureTemplate


def Clockwork(size, activation='tanh', name=None):
    return ConstructionWrapper.create(ClockworkLayerImpl,
                                      size=size,
                                      name=name,
                                      activation=activation)


class ClockworkLayerImpl(Layer):
    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'size', 'activation'}

    computes_no_gradients_for = ['timing']

    def setup(self, kwargs, in_shapes):
        self.activation = kwargs.get('activation', 'tanh')
        self.size = kwargs.get('size', in_shapes['default'].feature_size)

        if not isinstance(self.size, int):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))

        in_size = self.in_shapes['default'].feature_size

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', self.size,
                                             context_size=1)

        parameters = OrderedDict()
        parameters['W'] = BufferStructure(self.size, in_size)
        parameters['R'] = BufferStructure(self.size, self.size)
        parameters['bias'] = BufferStructure(self.size)
        parameters['timing'] = BufferStructure(self.size)

        internals = OrderedDict()
        internals['Ha'] = BufferStructure('T', 'B', self.size, context_size=1)
        internals['dHa'] = BufferStructure('T', 'B', self.size, context_size=1,
                                           is_backward_only=True)
        internals['dHb'] = BufferStructure('T', 'B', self.size, context_size=1,
                                           is_backward_only=True)

        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, R, bias, timing = buffers.parameters
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        Ha = buffers.internals.Ha

        flat_inputs = flatten_time_and_features(inputs)
        flat_H = flatten_time(Ha[:-1])

        _h.dot_mm(flat_inputs, W, flat_H, transb=True)
        _h.add_mv(flat_H, bias.reshape((1, self.size)), flat_H)

        tmp = _h.zeros(timing.shape)
        cond = _h.zeros(outputs[0].shape)
        for t in range(inputs.shape[0]):
            _h.dot_add_mm(outputs[t - 1], R, Ha[t], transb=True)
            _h.act_func[self.activation](Ha[t], outputs[t])
            # Undo updates
            if t > 0:
                _h.fill(tmp, t)
                _h.modulo_tt(tmp, timing, tmp)
                _h.broadcast_t(tmp.reshape((1, tmp.shape[0])), 0, cond)
                _h.copy_to_if(outputs[t - 1], outputs[t], cond)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        W, R, bias, timing = buffers.parameters
        dW, dR, dbias, dtiming = buffers.gradients
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        dinputs = buffers.input_deltas.default
        doutputs = buffers.output_deltas.default
        Ha, dHa, dHb = buffers.internals

        tmp = _h.zeros(timing.shape)
        cond = _h.zeros(outputs[0].shape)

        _h.copy_to(doutputs, dHb)
        T = inputs.shape[0] - 1
        _h.act_func_deriv[self.activation](Ha[T], outputs[T], dHb[T], dHa[T])
        for t in range(T - 1, -1, -1):
            _h.fill(tmp, t + 1)
            _h.modulo_tt(tmp, timing, tmp)
            _h.broadcast_t(tmp.reshape((1, tmp.shape[0])), 0, cond)
            _h.add_into_if(dHb[t + 1], dHb[t], cond)
            _h.fill_if(dHa[t+1], 0.0, cond)
            _h.dot_add_mm(dHa[t + 1], R, dHb[t])
            _h.act_func_deriv[self.activation](Ha[t], outputs[t], dHb[t],
                                               dHa[t])

        flat_inputs = flatten_time_and_features(inputs)
        flat_dinputs = flatten_time_and_features(dinputs)
        flat_dHa = flatten_time(dHa[:-1])

        # Calculate in_deltas and gradients
        _h.dot_add_mm(flat_dHa, W, flat_dinputs)
        _h.dot_add_mm(flat_dHa, flat_inputs, dW, transa=True)
        dbias_tmp = _h.allocate(dbias.shape)
        _h.sum_t(flat_dHa, axis=0, out=dbias_tmp)
        _h.add_tt(dbias, dbias_tmp, dbias)

        flat_outputs = flatten_time(outputs[:-2])
        flat_dHa = flatten_time(dHa[1:-1])
        _h.dot_add_mm(flat_dHa, flat_outputs, dR, transa=True)
        _h.dot_add_mm(dHa[0], outputs[-1], dR, transa=True)
