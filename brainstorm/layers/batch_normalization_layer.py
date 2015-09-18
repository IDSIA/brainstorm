#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate
from brainstorm.utils import flatten_time_and_features


def BatchNorm(name=None, decay=0.9, epsilon=1.0e-5):
    return ConstructionWrapper.create('BatchNorm', name=name, decay=decay,
                                      epsilon=epsilon)


class BatchNormLayerImpl(LayerBaseImpl):

    inputs = {'default': ShapeTemplate('T', 'B', '...')}
    outputs = {'default': ShapeTemplate('T', 'B', '...')}
    expected_kwargs = {'decay', 'epsilon'}

    def _setup_hyperparameters(self):
        self.epsilon = self.kwargs.get('epsilon', 1.0e-5)
        self.decay = self.kwargs.get('decay', 0.9)
        assert 0.0 <= self.decay <= 1.0

    def _get_output_shapes(self):
        return {'default': self.in_shapes['default']}

    def get_parameter_structure(self):
        parameters = OrderedDict()
        feature_shape = ShapeTemplate(self.in_shapes['default'].feature_size)
        parameters['gamma'] = feature_shape
        parameters['beta'] = feature_shape
        parameters['mu'] = feature_shape
        parameters['sigma'] = feature_shape
        return parameters

    def get_internal_structure(self):
        internals = OrderedDict()
        feature_shape = ShapeTemplate(self.in_shapes['default'].feature_size)
        internals['sigma_b'] = feature_shape
        internals['centered'] = self.in_shapes['default']
        internals['x_hat'] = self.in_shapes['default']
        return internals

    def forward_pass(self, buffers, training_pass=True):
        _h = self.handler
        sigma_b, centered, x_hat = buffers.internals
        gamma, beta, mu, sigma = buffers.parameters
        # Note: we flatten time for all buffers, so we skip the flat_ prefix
        inputs = flatten_time_and_features(buffers.inputs.default)
        centered = flatten_time_and_features(centered)
        x_hat = flatten_time_and_features(x_hat)
        out = flatten_time_and_features(buffers.outputs.default)
        m = inputs.shape[0]

        if training_pass:
            mu_b = sigma_b  # temporary use this with other name
            # Calculate the (negative) batch mean
            _h.sum_t(inputs, 0, mu_b)
            _h.mult_st(-1.0 / m, mu_b, mu_b)

            # Adjust mu as an exponential moving average
            # TODO: Find better way
            _h.mult_st(self.decay, mu, mu)
            _h.mult_add_st(1.0 - self.decay, mu_b, mu)

            mu = mu_b

        # Calculate the centered activations
        _h.add_mv(inputs, mu.reshape((1, mu.size)), centered)

        if training_pass:
            sigma2 = sigma_b        # temporary use this with other name
            centered2 = x_hat  # temporary use this with other name
            # Calculate the variance
            _h.mult_tt(centered, centered, centered2)
            _h.sum_t(centered2, 0, sigma2)
            _h.mult_st(1.0 / m, sigma2, sigma2)  # TODO m-1 instead?
            _h.add_st(self.epsilon, sigma2, sigma2)  # (numerically stabilized)

            # Standard deviation
            _h.sqrt_t(sigma2, sigma_b)

            # Adjust sigma as an exponential moving sigma
            # FIXME: This is clearly a hack and wrong
            _h.mult_st(self.decay, sigma, sigma)
            _h.mult_add_st(1.0 - self.decay, sigma_b, sigma)

            sigma = sigma_b

        # compute normalized inputs
        _h.divide_mv(centered, sigma.reshape((1, sigma.size)), x_hat)

        # Compute outputs
        _h.mult_mv(x_hat, gamma.reshape((1, gamma.size)), out)
        _h.add_mv(out, beta.reshape((1, beta.size)), out)

    def backward_pass(self, buffers):
        _h = self.handler
        sigma_b, centered, x_hat = buffers.internals
        gamma = buffers.parameters.gamma
        dgamma = buffers.gradients.gamma
        dbeta = buffers.gradients.beta
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

        # get normalization factor (gamma / sigma_b)
        coeff = small_tmp
        _h.divide_tt(gamma, sigma_b, coeff)

        term4 = big_tmp
        _h.mult_mv(term3, coeff.reshape((1, coeff.size)), term4)
        _h.add_tt(term4, indeltas, indeltas)
