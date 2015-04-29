#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class LstmLayerImpl(LayerBaseImpl):
    expected_kwargs = {'size', 'activation_function'}

    def __init__(self, name, in_shapes, incoming_connections,
                 outgoing_connections, **kwargs):
        super(LstmLayerImpl, self).__init__(
            name, in_shapes, incoming_connections, outgoing_connections,
            **kwargs)
        self.act_func = lambda x, y: None
        self.act_func_deriv = lambda x, y, dy, dx: None
        self.kwargs = kwargs

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

        internals['Za'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['Zb'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['Ia'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['Ib'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['Fa'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['Fb'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['Oa'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['Ob'] = ShapeTemplate('T', 'B', out_size, context_size=1)
        internals['C'] = ShapeTemplate('T', 'B', out_size, context_size=1)

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
        Za, Zb, Ia, Ib, Fa, Fb, Oa, Ob, C = forward_buffers.internals
        x = forward_buffers.inputs.default
        y = forward_buffers.outputs.default

        time_size, batch_size, in_size = x.shape

        for t in range(1, time_size + 1):
            # Block input
            _h.dot_mm(x[t - 1], Wz, Za[t])
            _h.dot_add_mm(y[t - 1], Rz, Za[t])
            _h.add_(Za[t], bz, Za[t])
            self.act_func(Za[t], Zb[t])

            # Input Gate
            _h.dot_mm(x[t - 1], Wi, Ia[t])
            _h.dot_add_mm(y[t - 1], Ri, Ia[t])
            _h.add_(Ia[t], bi, Ia[t])
            _h.sigmoid(Ia[t], Ib[t])

            # Forget Gate
            _h.dot_mm(x[t - 1], Wf, Fa[t])
            _h.dot_add_mm(y[t - 1], Rf, Fa[t])
            _h.add_(Fa[t], bf, Fa[t])
            _h.sigmoid(Fa[t], Fb[t])

            # Cell
            _h.mult_tt(Ib[t], Zb[t], C[t])
            _h.mult_add_tt(Fb[t], C[t-1], C[t])

            # Output Gate
            _h.dot_mm(x[t - 1], Wo, Oa[t])
            _h.dot_add_mm(y[t - 1], Ro, Oa[t])
            _h.add_(Oa[t], bo, Oa[t])
            _h.sigmoid(Oa[t], Ob[t])

            # Block output
            self.act_func(C[t], y[t])
            _h.mult_add_tt(Ob[t], y[t], y[t])

    def backward_pass(self, forward_buffers, backward_buffers):
        # prepare
        _h = self.handler
        (Wz, Wi, Wf, Wo,
         Rz, Ri, Rf, Ro,
         bz, bi, bf, bo) = forward_buffers.parameters
        (dWz, dWi, dWf, dWo,
         dRz, dRi, dRf, dRo,
         dbz, dbi, dbf, dbo) = backward_buffers.parameters

        Za, Zb, Ia, Ib, Fa, Fb, Oa, Ob, C = forward_buffers.internals
        dZa, dZb, dIa, dIb, dFa, dFb, dOa, dOb, dC = backward_buffers.internals

        x = forward_buffers.inputs.default
        dx = backward_buffers.inputs.default
        y = forward_buffers.outputs.default
        deltas = backward_buffers.outputs.default

        time_size, batch_size, in_size = x.shape
        # for t in reversed(range(1, time_size + 1)):
        for t in range(time_size, 0, - 1):
            dy[t] = deltas[t] + np.dot(di[t+1], Ri.T) + np.dot(df[t+1], Rf.T)  +\
                            np.dot(do[t+1], Ro.T) + np.dot(dz[t+1], Rz.T)
            do[t] = dy[t] * h(c[t]) * sigma_deriv(oa[t])
            dc[t] = dy[t] * o[t] * h_deriv(c[t])
            if t < T-1:
                dc[t] += dc[t+1] * f[t+1]
            if t > 0:
                df[t] = dc[t] * c[t-1] * sigma_deriv(fa[t])
            di[t] = dc[t] * z[t] * sigma_deriv(ia[t])
            dz[t] = dc[t] * i[t] * g_deriv(za[t])

            # Input Deltas
            dx[t] = np.dot(dz[t], Wz.T) + np.dot(di[t], Wi.T) + np.dot(df[t], Wf.T) + np.dot(do[t], Wo.T)

            # Gradients for the weights
            dWz += np.outer(x[t], dz[t])
            dWi += np.outer(x[t], di[t])
            dWf += np.outer(x[t], df[t])
            dWo += np.outer(x[t], do[t])
            dRz += np.outer(y[t], dz[t+1])
            dRi += np.outer(y[t], di[t+1])
            dRf += np.outer(y[t], df[t+1])
            dRo += np.outer(y[t], do[t+1])
            dbz += dz[t]
            dbi += di[t]
            dbf += df[t]
            dbo += do[t]
