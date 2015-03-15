#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
from brainstorm.structure.architecture import combine_input_shapes, \
    validate_architecture, get_canonical_layer_order
from brainstorm.utils import InvalidArchitectureError


def test_combine_input_sizes_int():
    assert combine_input_shapes([7]) == (7,)
    assert combine_input_shapes([1, 2, 3, 4]) == (10,)


def test_combine_input_sizes_int_and_unituples():
    assert combine_input_shapes([1, 2, (3,), (4,), 5]) == (15,)


def test_combine_input_sizes_tuples():
    assert combine_input_shapes([(1, 4)]) == (1, 4)

    assert combine_input_shapes([(1, 4),
                                (3, 4),
                                (6, 4)]) == (10, 4)

    assert combine_input_shapes([(2, 3, 4),
                                (3, 3, 4),
                                (2, 3, 4)]) == (7, 3, 4)


@pytest.mark.parametrize('sizes', [
    [2, (1, 2)],
    [(2, 3), (2, 2)],
    [(2,), (1, 2)],
    [(2, 1, 3), (3, 1, 3), (2, 2, 3)],
    [(2, 1, 3), (3, 1, 3), (1, 1, 2)]
])
def test_combine_input_sizes_mismatch(sizes):
    with pytest.raises(ValueError):
        combine_input_shapes(sizes)


def test_validate_architecture_minimal():
    assert validate_architecture({
        'InputLayer': {
            '@type': 'DataLayer',
            '@outgoing_connections': []
        }})


def test_validate_architecture_raises_on_missing_type():
    with pytest.raises(InvalidArchitectureError):
        validate_architecture({
            'typeless': {
                '@outgoing_connections': []
            }
        })


def test_validate_architecture_raises_on_invalid_type():
    with pytest.raises(InvalidArchitectureError):
        validate_architecture({
            'wrong_type': {
                '@type': pytest,
                '@outgoing_connections': []
            }
        })


def test_validate_architecture_raises_on_invalid_name():
    with pytest.raises(InvalidArchitectureError):
        validate_architecture({
            '$invalid name': {
                '@type': 'DataLayer',
                '@outgoing_connections': []
            }
        })


def test_validate_architecture_raises_on_nonexisting_outgoing():
    with pytest.raises(InvalidArchitectureError):
        validate_architecture({
            'InputLayer': {
                '@type': 'DataLayer',
                '@outgoing_connections': ['missing_layer']
            }
        })


def test_validate_architecture_raises_on_no_data_layer():
    with pytest.raises(InvalidArchitectureError):
        validate_architecture({
            'fwd1': {
                '@type': 'FullyConnectedLayer',
                '@outgoing_connections': []
            }})


def test_validate_architecture_raises_inputs_to_data_layer():
    with pytest.raises(InvalidArchitectureError):
        validate_architecture({
            'InputLayer': {
                '@type': 'DataLayer',
                '@outgoing_connections': []
            },
            'fwd1': {
                '@type': 'FullyConnectedLayer',
                '@outgoing_connections': ['InputLayer']
            }})


def test_validate_architecture_full_network():
    assert validate_architecture({
        'InputLayer': {
            '@type': 'DataLayer',
            'shape': 784,
            '@outgoing_connections': ['HiddenLayer']
        },
        'HiddenLayer': {
            '@type': 'FullyConnectedLayer',
            'shape': 1000,
            '@outgoing_connections': ['OutputLayer']
        },
        'OutputLayer': {
            '@type': 'FullyConnectedLayer',
            'shape': 10,
            'activation_function': 'softmax',
            '@outgoing_connections': []
        }
    })


def test_validate_architecture_with_named_sinks():
    assert validate_architecture({
        'InputLayer': {
            '@type': 'DataLayer',
            'shape': 10,
            '@outgoing_connections': ['HiddenLayer', 'OutputLayer.A']
        },
        'HiddenLayer': {
            '@type': 'FullyConnectedLayer',
            'shape': 10,
            '@outgoing_connections': ['OutputLayer.B']
        },
        'OutputLayer': {
            '@type': 'PointwiseAdditionLayer',
            '@outgoing_connections': []
        }
    })


def test_validate_architecture_with_named_sources():
    assert validate_architecture({
        'InputLayer': {
            '@type': 'DataLayer',
            'shape': 10,
            '@outgoing_connections': ['SplitLayer']
        },
        'SplitLayer': {
            '@type': 'SplitLayer',
            'split_at': 5,
            '@outgoing_connections': {
                'left': ['OutputLayer.A'],
                'right': ['OutputLayer.B'],
            }
        },
        'OutputLayer': {
            '@type': 'PointwiseAdditionLayer',
            '@outgoing_connections': []
        }
    })


def test_get_canonical_architecture_order():
    arch = {
        'A': {
            '@type': 'DataLayer',
            '@outgoing_connections': {'B1', 'C'}
        },
        'B1': {
            '@type': 'layertype',
            '@outgoing_connections': {'B2'}
        },
        'B2': {
            '@type': 'layertype',
            '@outgoing_connections': {'D'}
        },
        'C': {
            '@type': 'layertype',
            '@outgoing_connections': {'D'}
        },
        'D': {
            '@type': 'layertype',
            '@outgoing_connections': set()
        }
    }
    assert get_canonical_layer_order(arch) == ['A', 'B1', 'B2', 'C', 'D']


def test_get_canonical_architecture_order_with_named_sinks():
    arch = {
        'A': {
            '@type': 'DataLayer',
            '@outgoing_connections': {'B1', 'C'}
        },
        'B1': {
            '@type': 'layertype',
            '@outgoing_connections': {'B2'}
        },
        'B2': {
            '@type': 'layertype',
            '@outgoing_connections': {'D.source1'}
        },
        'C': {
            '@type': 'layertype',
            '@outgoing_connections': {'D.source1'}
        },
        'D': {
            '@type': 'layertype',
            '@outgoing_connections': set()
        }
    }
    assert get_canonical_layer_order(arch) == ['A', 'B1', 'B2', 'C', 'D']


def test_get_canonical_architecture_order_with_named_sources():
    arch = {
        'A': {
            '@type': 'DataLayer',
            '@outgoing_connections': {'out1': ['B1'],
                                      'out2': ['C']}
        },
        'B1': {
            '@type': 'layertype',
            '@outgoing_connections': {'B2'}
        },
        'B2': {
            '@type': 'layertype',
            '@outgoing_connections': {'D.source1'}
        },
        'C': {
            '@type': 'layertype',
            '@outgoing_connections': {'D.source1'}
        },
        'D': {
            '@type': 'layertype',
            '@outgoing_connections': set()
        }
    }
    assert get_canonical_layer_order(arch) == ['A', 'B1', 'B2', 'C', 'D']