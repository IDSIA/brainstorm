#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate
from brainstorm.utils import flatten_time_and_features


class BatchNormLayerImpl(LayerBaseImpl):

    inputs = {'default': ShapeTemplate('T', 'B', '...')}
    outputs = {'default': ShapeTemplate('T', 'B', '...')}

    def _setup_hyperparameters(self):
        self.epsilon = 1.0e-5

    def _get_output_shapes(self):
        return {'default': self.in_shapes['default']}

    def get_parameter_structure(self):
        parameters = OrderedDict()
        feature_shape = ShapeTemplate(self.in_shapes['default'].feature_size)
        parameters['gamma'] = feature_shape
        parameters['beta'] = feature_shape
        return parameters

    def get_internal_structure(self):
        internals = OrderedDict()
        feature_shape = ShapeTemplate(self.in_shapes['default'].feature_size)
        internals['std'] = feature_shape
        internals['centered'] = self.in_shapes['default']
        internals['x_hat'] = self.in_shapes['default']
        return internals

    def forward_pass(self, buffers, training_pass=True):
        _h = self.handler
        std, centered, x_hat = buffers.internals
        gamma, beta = buffers.parameters
        # Note: we flatten time for all buffers, so we skip the flat_ prefix
        inputs = flatten_time_and_features(buffers.inputs.default)
        centered = flatten_time_and_features(centered)
        x_hat = flatten_time_and_features(x_hat)
        out = flatten_time_and_features(buffers.outputs.default)
        m = inputs.shape[0]

        mu = std  # temporary use this with other name
        # Calculate the (negative) mean
        _h.sum_t(inputs, 0, mu)
        _h.mult_st(-1.0 / m, mu, mu)

        # Calculate the centered activations
        _h.add_mv(inputs, mu.reshape((1, mu.size)), centered)

        sigma2 = mu        # temporary use this with other name
        centered2 = x_hat  # temporary use this with other name
        # Calculate the variance
        _h.mult_tt(centered, centered, centered2)
        _h.sum_t(centered2, 0, sigma2)
        _h.mult_st(1.0 / m, sigma2, sigma2)  # TODO m-1 instead?
        _h.add_st(self.epsilon, sigma2, sigma2)  # (numerically stabilized)

        # Standard deviation
        _h.sqrt_t(sigma2, std)

        # compute normalized inputs
        _h.divide_mv(centered, std.reshape((1, std.size)), x_hat)

        # Compute outputs
        _h.mult_mv(x_hat, gamma.reshape((1, gamma.size)), out)
        _h.add_mv(out, beta.reshape((1, beta.size)), out)

    def backward_pass(self, buffers):
        _h = self.handler
        std, centered, x_hat = buffers.internals
        gamma = buffers.parameters.gamma
        dgamma, dbeta = buffers.gradients
        # Note: we flatten time for all buffers, so we skip the flat_ prefix
        x_hat = flatten_time_and_features(x_hat)
        outdeltas = flatten_time_and_features(buffers.output_deltas.default)
        indeltas = flatten_time_and_features(buffers.input_deltas.default)
        m = outdeltas.shape[0]

        big_tmp = _h.allocate(x_hat.shape)     # big
        small_tmp = _h.allocate(gamma.shape)  # small

        # ------------- Gradients ---------------
        # Calculate dgamma
        tmp = big_tmp
        dgamma_tmp = small_tmp
        _h.mult_tt(outdeltas, x_hat, tmp)
        _h.sum_t(tmp, axis=0, out=dgamma_tmp)
        _h.add_tt(dgamma_tmp, dgamma, dgamma)

        _h.mult_st(1 / m, dgamma_tmp, dgamma_tmp)
        term1 = big_tmp
        _h.mult_mv(x_hat, dgamma_tmp.reshape((1, gamma.size)), term1)

        # Calculate dbeta
        dbeta_tmp = small_tmp
        _h.sum_t(outdeltas, axis=0, out=dbeta_tmp)
        _h.add_tt(dbeta_tmp, dbeta, dbeta)
        _h.mult_st(1 / m, dbeta_tmp, dbeta_tmp)

        # ------------- Deltas ---------------
        term2 = big_tmp
        term3 = big_tmp
        _h.subtract_tt(outdeltas, term1, term2)
        _h.subtract_mv(term2, dbeta_tmp.reshape((1, dbeta.size)), term3)

        # get normalization factor (gamma / std)
        coeff = small_tmp
        _h.divide_tt(gamma, std, coeff)

        term4 = big_tmp
        _h.mult_mv(term3, coeff.reshape((1, coeff.size)), term4)
        _h.add_tt(term4, indeltas, indeltas)
