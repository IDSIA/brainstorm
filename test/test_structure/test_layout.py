#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest

from brainstorm.structure.layout import (
    ParameterLayoutEntry,
    create_layout_stub, get_order, get_parameter_order, get_internal_order,
    get_forced_orders, get_connections, merge_connections, get_forward_closure,
    can_be_connected_with_single_buffer, flatten, convert_to_nested_indices,
    permute_rows, create_layout, gather_array_nodes)
from brainstorm.structure.architecture import (
    instantiate_layers_from_architecture)
from brainstorm.utils import InvalidArchitectureError


@pytest.fixture
def impossible_layers():
    arch = {
        'InputLayer': {
            '@type': 'DataLayer',
            'shape': 2,
            '@outgoing_connections': {'A', 'B'}
        },
        'A': {
            '@type': 'FeedForwardLayer',
            'shape': 2,
            '@outgoing_connections': {'C', 'D'}
        },
        'B': {
            '@type': 'FeedForwardLayer',
            '@outgoing_connections': {'C', 'E'}
        },
        'C': {
            '@type': 'FeedForwardLayer',
            '@outgoing_connections': {'D', 'E'}
        },
        'D': {
            '@type': 'FeedForwardLayer',
            '@outgoing_connections': {'out'}
        },
        'E': {
            '@type': 'FeedForwardLayer',
            '@outgoing_connections': {'out'}
        },
        'out': {
            '@type': 'FeedForwardLayer',
            '@outgoing_connections': set()
        }
    }
    return instantiate_layers_from_architecture(arch)


def test_get_order():
    order = get_order({
        'W': {'index': 0},
        'b': {'index': 2},
        'R': {'index': 1}
    })
    assert order == ('W', 'R', 'b')


def test_get_order_raises_on_missing_index():
    with pytest.raises(KeyError):
        order = get_order({
            'W': {'layout': {}},
            'b': {'index': 2},
            'R': {'index': 1}
        })


def test_get_parameter_order(layers):
    assert get_parameter_order('InputLayer', layers['InputLayer']) == ()
    assert get_parameter_order('A', layers['A']) == ('A.parameters.W',
                                                     'A.parameters.b')
    assert get_parameter_order('B', layers['B']) == ('B.parameters.W',
                                                     'B.parameters.b')


def test_get_internals_order(layers):
    assert get_internal_order('InputLayer', layers['InputLayer']) == ()
    assert get_internal_order('A', layers['A']) == ('A.internals.Ha',)
    assert get_internal_order('B', layers['B']) == ('B.internals.Ha',)


def test_get_forced_orders(layers):
    assert get_forced_orders(layers) == [
        ('A.parameters.W', 'A.parameters.b'),
        ('B.parameters.W', 'B.parameters.b'),
        ('C.parameters.W', 'C.parameters.b'),
        ('D.parameters.W', 'D.parameters.b'),
        ('A.internals.Ha',),
        ('B.internals.Ha',),
        ('C.internals.Ha',),
        ('D.internals.Ha',)
    ]


def test_get_connections(layers):
    assert get_connections(layers) == [
        ('A.outputs.default', 'C.inputs.default'),
        ('B.outputs.default', 'C.inputs.default'),
        ('B.outputs.default', 'D.inputs.default'),
        ('C.outputs.default', 'D.inputs.default'),
        ('InputLayer.outputs.default', 'A.inputs.default'),
        ('InputLayer.outputs.default', 'B.inputs.default')
    ]


def test_merge_connections():
    connections = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'E'),
        ('D', 'E'),
        ('E', 'F')
    ]
    forced_orders = [('A', 'B'), ('C', 'D')]
    assert merge_connections(connections, forced_orders) == [
        (('A', 'B'), ('A', 'B')),
        (('A', 'B'), ('C', 'D')),
        (('A', 'B'), 'E'),
        (('C', 'D'), 'E'),
        ('E', 'F')
    ]


def test_get_forward_closure():
    connections = [
        ('A', 'B'),  # #       /-> C -\
        ('B', 'C'),  # #      /        \
        ('B', 'D'),  # # A -> B         -> E
        ('C', 'E'),  # #      \        /
        ('D', 'E'),  # #       \-> D -/--> F
        ('D', 'F')
    ]
    assert get_forward_closure('A', connections) == ({'A'},      {'B'})
    assert get_forward_closure('B', connections) == ({'B'},      {'C', 'D'})
    assert get_forward_closure('C', connections) == ({'C', 'D'}, {'E', 'F'})
    assert get_forward_closure('D', connections) == ({'C', 'D'}, {'E', 'F'})


@pytest.mark.parametrize('col,expected', [
    ([0, 0, 0, 0, 0], True),
    ([1, 1, 0, 0, 0], True),
    ([0, 0, 0, 1, 1], True),
    ([0, 1, 1, 1, 0], True),
    ([1, 1, 1, 1, 1], True),
    ([0, 1, 0, 1, 0], False),
    ([1, 1, 0, 1, 0], False),
    ([1, 0, 0, 1, 0], False),
    ([0, 1, 0, 1, 1], False),
    ([0, 1, 0, 0, 1], False),
    ([1, 1, 0, 1, 1], False),
])
def test_col_can_be_connected_with_single_buffer(col, expected):
    assert can_be_connected_with_single_buffer(np.array(col).reshape(-1, 1)) == expected


def test_can_be_connected_with_single_buffer():
    con_table = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]]).T
    assert can_be_connected_with_single_buffer(con_table)

    con_table = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0],  # < the bad boy
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]]).T
    assert not can_be_connected_with_single_buffer(con_table)


def test_flatten():
    assert list(flatten([0, (1, 2, 3), 4, [5, (6, 7), 8]])) == list(range(9))


def test_convert_to_nested_indices():
    assert list(convert_to_nested_indices(
        ['a', ('b', 'c', 'a'), 'b', ['c', ('c', 'c'), 'b']])) == \
        [0, [1, 2, 3], 4, [5, [6, 7], 8]]


def test_permute_rows():
    con_table = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1]])
    perm = permute_rows(con_table, [0, 1, 2, [3, 4]])
    assert perm == [0, 2, 1, 3, 4]

    con_table = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1]])
    perm = permute_rows(con_table, [[0, 1], 2, 3, 4])
    assert perm == [0, 1, 3, 2, 4]


def test_create_layout_stub(layers):
    layout = create_layout_stub(layers)
    import pprint
    pprint.pprint(layout)
    assert layout == {
        'InputLayer': {'index': 0, 'layout': {
            'inputs': {'index': 0, 'layout': {}},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (2,)},
            }},
            'parameters': {'index': 2, 'layout': {}},
            'internals': {'index': 3, 'layout': {}},
        }},
        'A': {'index': 1, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (2,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (3,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, -1, -1), 'shape': (2, 3)},
                'b': {'index': 1, 'slice': (0, -1, -1), 'shape': (3,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, -1, -1), 'shape': (3,)}
            }},
        }},
        'B': {'index': 2, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (2,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (5,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, -1, -1), 'shape': (2, 5)},
                'b': {'index': 1, 'slice': (0, -1, -1), 'shape': (5,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, -1, -1), 'shape': (5,)}
            }},
        }},
        'C': {'index': 3, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (8,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (7,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, -1, -1), 'shape': (8, 7)},
                'b': {'index': 1, 'slice': (0, -1, -1), 'shape': (7,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, -1, -1), 'shape': (7,)}
            }},
        }},
        'D': {'index': 4, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (12,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (11,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, -1, -1), 'shape': (12, 11)},
                'b': {'index': 1, 'slice': (0, -1, -1), 'shape': (11,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, -1, -1), 'shape': (11,)}
            }},
        }}
    }


def test_traverse_layout():
    layout = {
        'inp': {'index': 0, 'layout': {
            'inputs': {'index': 0, 'layout': {}},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (2,)},
            }},
            'parameters': {'index': 2, 'layout': {}},
            'internals': {'index': 3, 'layout': {}},
        }},
        'A': {'index': 1, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (2,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (3,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, -1, -1), 'shape': (2, 3)},
                'b': {'index': 1, 'slice': (0, -1, -1), 'shape': (3,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, -1, -1), 'shape': (3,)}
            }},
        }}}
    assert set(gather_array_nodes(layout)) == {
        'inp.outputs.default', 'A.inputs.default', 'A.outputs.default',
        'A.parameters.W', 'A.parameters.b', 'A.internals.Ha'}


def test_create_layout(layers):
    sizes, layout = create_layout(layers)
    import pprint
    pprint.pprint(layout)
    assert layout == {
        'InputLayer': {'index': 0, 'layout': {
            'inputs': {'index': 0, 'layout': {}},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, 0, 2), 'shape': (2,)},
            }},
            'parameters': {'index': 2, 'layout': {}},
            'internals': {'index': 3, 'layout': {}},
        }},
        'A': {'index': 1, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, 0, 2), 'shape': (2,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, 2, 5), 'shape': (3,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, 0, 6), 'shape': (2, 3)},
                'b': {'index': 1, 'slice': (0, 6, 9), 'shape': (3,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, 17, 20), 'shape': (3,)}
            }},
        }},
        'B': {'index': 2, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, 0, 2), 'shape': (2,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, 5, 10), 'shape': (5,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, 9, 19), 'shape': (2, 5)},
                'b': {'index': 1, 'slice': (0, 19, 24), 'shape': (5,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, 20, 25), 'shape': (5,)}
            }},
        }},
        'C': {'index': 3, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, 2, 10), 'shape': (8,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, 10, 17), 'shape': (7,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, 24, 80), 'shape': (8, 7)},
                'b': {'index': 1, 'slice': (0, 80, 87), 'shape': (7,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, 25, 32), 'shape': (7,)}
            }},
        }},
        'D': {'index': 4, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, 5, 17), 'shape': (12,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, 32, 43), 'shape': (11,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, 87, 219), 'shape': (12, 11)},
                'b': {'index': 1, 'slice': (0, 219, 230), 'shape': (11,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, 43, 54), 'shape': (11,)}
            }},
        }}
    }
