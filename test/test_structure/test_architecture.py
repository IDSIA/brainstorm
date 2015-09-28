#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
from brainstorm.structure.architecture import combine_input_shapes, \
    validate_architecture, get_canonical_layer_order
from brainstorm.utils import NetworkValidationError
from brainstorm.structure.shapes import StructureTemplate


def test_combine_input_sizes_tuples():
    assert combine_input_shapes([StructureTemplate(1, 4)]) == StructureTemplate(1, 4)

    assert combine_input_shapes([StructureTemplate(1, 4),
                                StructureTemplate(3, 4),
                                StructureTemplate(6, 4)]) == StructureTemplate(10, 4)

    assert combine_input_shapes([StructureTemplate(2, 3, 4),
                                StructureTemplate(3, 3, 4),
                                StructureTemplate(2, 3, 4)]) == \
        StructureTemplate(7, 3, 4)


def test_combine_input_sizes_tuple_templates():
    assert (combine_input_shapes([StructureTemplate('B', 4)]) ==
            StructureTemplate('B', 4))
    assert (combine_input_shapes([StructureTemplate('B', 4),
                                 StructureTemplate('B', 3)]) ==
            StructureTemplate('B', 7))
    assert (combine_input_shapes([StructureTemplate('T', 'B', 4)]) ==
            StructureTemplate('T', 'B', 4))
    assert (combine_input_shapes([StructureTemplate('T', 'B', 4),
                                 StructureTemplate('T', 'B', 3)]) ==
            StructureTemplate('T', 'B', 7))
    assert (combine_input_shapes([StructureTemplate('T', 'B', 4, 3, 2),
                                 StructureTemplate('T', 'B', 3, 3, 2)]) ==
            StructureTemplate('T', 'B', 7, 3, 2))


@pytest.mark.parametrize('sizes', [
    [StructureTemplate(2, 3), StructureTemplate(2, 2)],
    [StructureTemplate(2), StructureTemplate(1, 2)],
    [StructureTemplate(2, 1, 3), StructureTemplate(3, 1, 3), StructureTemplate(2, 2, 3)],
    [StructureTemplate(2, 1, 3), StructureTemplate(3, 1, 3), StructureTemplate(1, 1, 2)]
])
def test_combine_input_sizes_mismatch(sizes):
    with pytest.raises(ValueError):
        combine_input_shapes(sizes)


def test_validate_architecture_minimal():
    assert validate_architecture({
        'Input': {
            '@type': 'Input',
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
                '@type': 'Input',
                '@outgoing_connections': []
            }
        })


def test_validate_architecture_raises_on_nonexisting_outgoing():
    with pytest.raises(NetworkValidationError):
        validate_architecture({
            'Input': {
                '@type': 'Input',
                '@outgoing_connections': ['missing_layer']
            }
        })


def test_validate_architecture_raises_on_no_data_layer():
    with pytest.raises(NetworkValidationError):
        validate_architecture({
            'fwd1': {
                '@type': 'FullyConnected',
                '@outgoing_connections': []
            }})


def test_validate_architecture_raises_inputs_to_data_layer():
    with pytest.raises(NetworkValidationError):
        validate_architecture({
            'Input': {
                '@type': 'Input',
                '@outgoing_connections': []
            },
            'fwd1': {
                '@type': 'FullyConnected',
                '@outgoing_connections': ['Input']
            }})


def test_validate_architecture_full_network():
    assert validate_architecture({
        'Input': {
            '@type': 'Input',
            'out_shapes': {'default': 784},
            '@outgoing_connections': ['HiddenLayer']
        },
        'HiddenLayer': {
            '@type': 'FullyConnected',
            'shape': 1000,
            '@outgoing_connections': ['OutputLayer']
        },
        'OutputLayer': {
            '@type': 'FullyConnected',
            'shape': 10,
            'activation_function': 'softmax',
            '@outgoing_connections': []
        }
    })


def test_validate_architecture_with_named_sinks():
    assert validate_architecture({
        'Input': {
            '@type': 'Input',
            'out_shapes': {'default': 10},
            '@outgoing_connections': ['HiddenLayer', 'OutputLayer.A']
        },
        'HiddenLayer': {
            '@type': 'FullyConnected',
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
        'Input': {
            '@type': 'Input',
            'shape': {'default': 10},
            '@outgoing_connections': ['SplitLayer']
        },
        'SplitLayer': {
            '@type': 'Split',
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
            '@type': 'Input',
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
            '@type': 'Input',
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
            '@type': 'Input',
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
