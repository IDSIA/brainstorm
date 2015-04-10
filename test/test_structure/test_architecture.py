#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
from brainstorm.structure.architecture import combine_input_shapes, \
    validate_architecture, get_canonical_layer_order
from brainstorm.utils import NetworkValidationError
from brainstorm.structure.shapes import ShapeTemplate


def test_combine_input_sizes_tuples():
    assert combine_input_shapes([ShapeTemplate(1, 4)]) == ShapeTemplate(1, 4)

    assert combine_input_shapes([ShapeTemplate(1, 4),
                                ShapeTemplate(3, 4),
                                ShapeTemplate(6, 4)]) == ShapeTemplate(10, 4)

    assert combine_input_shapes([ShapeTemplate(2, 3, 4),
                                ShapeTemplate(3, 3, 4),
                                ShapeTemplate(2, 3, 4)]) == \
        ShapeTemplate(7, 3, 4)


def test_combine_input_sizes_tuple_templates():
    assert combine_input_shapes([ShapeTemplate('B', 4)]) == ShapeTemplate('B', 4)
    assert combine_input_shapes([ShapeTemplate('B', 4), ShapeTemplate('B', 3)]) == ShapeTemplate('B', 7)
    assert combine_input_shapes([ShapeTemplate('T', 'B', 4)]) == ShapeTemplate('T', 'B', 4)
    assert combine_input_shapes([ShapeTemplate('T', 'B', 4), ShapeTemplate('T', 'B', 3)]) == \
        ShapeTemplate('T', 'B', 7)
    assert combine_input_shapes([ShapeTemplate('T', 'B', 4, 3, 2), ShapeTemplate('T', 'B', 3, 3, 2)]) ==\
        ShapeTemplate('T', 'B', 7, 3, 2)


@pytest.mark.parametrize('sizes', [
    [ShapeTemplate(2, 3), ShapeTemplate(2, 2)],
    [ShapeTemplate(2), ShapeTemplate(1, 2)],
    [ShapeTemplate(2, 1, 3), ShapeTemplate(3, 1, 3), ShapeTemplate(2, 2, 3)],
    [ShapeTemplate(2, 1, 3), ShapeTemplate(3, 1, 3), ShapeTemplate(1, 1, 2)]
])
def test_combine_input_sizes_mismatch(sizes):
    with pytest.raises(ValueError):
        combine_input_shapes(sizes)


def test_validate_architecture_minimal():
    assert validate_architecture({
        'InputLayer': {
            '@type': 'InputLayer',
            '@outgoing_connections': []
        }})


def test_validate_architecture_raises_on_missing_type():
    with pytest.raises(NetworkValidationError):
        validate_architecture({
            'typeless': {
                '@outgoing_connections': []
            }
        })


def test_validate_architecture_raises_on_invalid_type():
    with pytest.raises(NetworkValidationError):
        validate_architecture({
            'wrong_type': {
                '@type': pytest,
                '@outgoing_connections': []
            }
        })


def test_validate_architecture_raises_on_invalid_name():
    with pytest.raises(NetworkValidationError):
        validate_architecture({
            '$invalid name': {
                '@type': 'InputLayer',
                '@outgoing_connections': []
            }
        })


def test_validate_architecture_raises_on_nonexisting_outgoing():
    with pytest.raises(NetworkValidationError):
        validate_architecture({
            'InputLayer': {
                '@type': 'InputLayer',
                '@outgoing_connections': ['missing_layer']
            }
        })


def test_validate_architecture_raises_on_no_data_layer():
    with pytest.raises(NetworkValidationError):
        validate_architecture({
            'fwd1': {
                '@type': 'FullyConnectedLayer',
                '@outgoing_connections': []
            }})


def test_validate_architecture_raises_inputs_to_data_layer():
    with pytest.raises(NetworkValidationError):
        validate_architecture({
            'InputLayer': {
                '@type': 'InputLayer',
                '@outgoing_connections': []
            },
            'fwd1': {
                '@type': 'FullyConnectedLayer',
                '@outgoing_connections': ['InputLayer']
            }})


def test_validate_architecture_full_network():
    assert validate_architecture({
        'InputLayer': {
            '@type': 'InputLayer',
            'out_shapes': {'default': 784},
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
            '@type': 'InputLayer',
            'out_shapes': {'default': 10},
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
            '@type': 'InputLayer',
            'shape': {'default': 10},
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
            '@type': 'InputLayer',
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
            '@type': 'InputLayer',
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
            '@type': 'InputLayer',
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
