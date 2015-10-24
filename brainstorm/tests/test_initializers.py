#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import mock
import numpy as np
import pytest

from brainstorm.initializers import (DenseSqrtFanIn, DenseSqrtFanInOut,
                                     EchoState, Gaussian, InitializationError,
                                     Initializer, SparseInputs, SparseOutputs,
                                     Uniform, evaluate_initializer)


@pytest.mark.parametrize("initializer", [Gaussian(),
                                         Uniform(),
                                         DenseSqrtFanIn(),
                                         DenseSqrtFanInOut()])
def test_initializers_return_correct_shape(initializer):
    shape = (4, 7)
    result = initializer(shape)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape


@pytest.mark.parametrize("initializer", [Gaussian(),
                                         Uniform()])
def test_universal_initializers_work_on_biases(initializer):
    shape = (5,)
    result = initializer(shape)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape


@pytest.mark.parametrize("initializer", [DenseSqrtFanInOut(),
                                         DenseSqrtFanIn(),
                                         SparseInputs(0),
                                         SparseOutputs(0),
                                         EchoState()])
def test_matrix_initializers_raise_on_1d_matrices(initializer):
    with pytest.raises(InitializationError):
        initializer((20,))


# ################ EchoState ##################################################

def test_echo_state_raises_if_matrix_non_square():
    with pytest.raises(InitializationError):
        EchoState()((3, 5))


def test_echostate_returns_correct_shape():
    result = EchoState()((4, 4))
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)


def test_echostate_has_correct_spectral_radius():
    result = EchoState(spectral_radius=1.7)((4, 4))
    spectral_radius = max(abs(np.linalg.eig(result)[0]))
    assert abs(spectral_radius - 1.7) < 1e-12


# ################ SparseInputs & SparseOutputs ###############################

def test_sparse_inputs_raises_if_input_dim_too_small():
    with pytest.raises(InitializationError):
        SparseInputs(1, connections=17)((15, 20))


def test_sparse_outputs_raises_if_output_dim_too_small():
    with pytest.raises(InitializationError):
        SparseOutputs(1, connections=17)((20, 15))


def test_sparse_inputs_has_correct_number_of_nonzero_rows():
    res = SparseInputs(1, connections=17)((25, 12))
    assert np.all(np.sum(res > 0, axis=0) == 17)


def test_sparse_outputs_has_correct_number_of_nonzero_cols():
    res = SparseOutputs(1, connections=17)((12, 25))
    assert np.all(np.sum(res > 0, axis=1) == 17)


# ########################## evaluate_initializer #############################

def test_evaluate_initializer_with_number():
    assert np.all(evaluate_initializer(1.4, (7, 5)) == 1.4)


def test_evaluate_initializer_calls_initializer():
    init = mock.create_autospec(Initializer())
    init.side_effect = lambda x: np.array(1)
    evaluate_initializer(init, (7, 5))
    init.assert_called_once_with((7, 5))


def test_evaluate_initializer_without_fallback_propagates_error():
    init = mock.create_autospec(Initializer())
    init.side_effect = InitializationError
    with pytest.raises(InitializationError):
        evaluate_initializer(init, (7, 5))


def test_evaluate_initializer_with_fallback_calls_fallback():
    init = mock.create_autospec(Initializer())
    fallback = mock.create_autospec(Initializer())
    fallback.side_effect = lambda x: np.array(1)
    init.side_effect = InitializationError
    evaluate_initializer(init, (7, 5), fallback)
    fallback.assert_called_once_with((7, 5))
