#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest

from brainstorm.structure.layout import (
    create_layout_stub, get_order, get_parameter_order, get_internal_order,
    get_forced_orders, get_connections, merge_connections, get_forward_closure,
    can_be_connected_with_single_buffer,
    permute_rows, create_layout, gather_array_nodes)


def test_get_order():
    order = get_order({
        'W': {'@index': 0},
        'b': {'@index': 2},
        'R': {'@index': 1}
    })
    assert order == ('W', 'R', 'b')


def test_get_order_raises_on_missing_index():
    with pytest.raises(KeyError):
        order = get_order({
            'W': {},
            'b': {'@index': 2},
            'R': {'@index': 1}
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
        ('A.parameters.W', 'parameters'),
        ('A.parameters.b', 'parameters'),
        ('B.outputs.default', 'C.inputs.default'),
        ('B.outputs.default', 'D.inputs.default'),
        ('B.parameters.W', 'parameters'),
        ('B.parameters.b', 'parameters'),
        ('C.outputs.default', 'D.inputs.default'),
        ('C.parameters.W', 'parameters'),
        ('C.parameters.b', 'parameters'),
        ('D.parameters.W', 'parameters'),
        ('D.parameters.b', 'parameters'),
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
    assert get_forward_closure('A', connections) == ({'A'}, {'B'})
    assert get_forward_closure('B', connections) == ({'B'}, {'C', 'D'})
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
    assert can_be_connected_with_single_buffer(np.array(col).reshape(-1, 1))\
           == expected


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
    assert layout == {
        '@type': 'BufferView',
        'parameters': {
            '@type': 'array',
            '@index': 0
        },
        'InputLayer': {
            '@type': 'BufferView',
            '@index': 1,
            'inputs': {'@type': 'BufferView', '@index': 0},
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2)},
            },
            'parameters': {'@type': 'BufferView', '@index': 2},
            'internals': {'@type': 'BufferView', '@index': 3},
        },
        'A': {
            '@type': 'BufferView',
            '@index': 2,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 3)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (2, 3)},
                'b': {'@type': 'array', '@index': 1, '@shape': (3,)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                'Ha': {'@type': 'array', '@index': 0, '@shape': ('T', 'B', 3)}
            },
        },
        'B': {
            '@type': 'BufferView',
            '@index': 3,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 5)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (2, 5)},
                'b': {'@type': 'array', '@index': 1, '@shape': (5,)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                'Ha': {'@type': 'array', '@index': 0, '@shape': ('T', 'B', 5)}
            },
        },
        'C': {
            '@type': 'BufferView',
            '@index': 4,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 8)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 7)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (8, 7)},
                'b': {'@type': 'array', '@index': 1, '@shape': (7,)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                'Ha': {'@type': 'array', '@index': 0, '@shape': ('T', 'B', 7)}
            },
        },
        'D': {
            '@type': 'BufferView',
            '@index': 5,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 12)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 11)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (12, 11)},
                'b': {'@type': 'array', '@index': 1, '@shape': (11,)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                'Ha': {'@type': 'array', '@index': 0, '@shape': ('T', 'B', 11)}
            },
        }}


def test_traverse_layout():
    layout = {
        '@type': 'BufferView',
        'inp': {
            '@type': 'BufferView',
            '@index': 0,
            'inputs': {'@type': 'BufferView', '@index': 0},
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2)},
            },
            'parameters': {'@type': 'BufferView', '@index': 2},
            'internals': {'@type': 'BufferView', '@index': 3},
        },
        'A': {
            '@type': 'BufferView',
            '@index': 1,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 3)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (2, 3)},
                'b': {'@type': 'array', '@index': 1, '@shape': (3,)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                'Ha': {'@type': 'array', '@index': 0, '@shape': ('T', 'B', 3)}
            }
        }
    }
    assert set(gather_array_nodes(layout)) == {
        'inp.outputs.default', 'A.inputs.default', 'A.outputs.default',
        'A.parameters.W', 'A.parameters.b', 'A.internals.Ha'}


def test_create_layout(layers):
    sizes, max_context_size, layout = create_layout(layers)
    assert layout == {
        '@type': 'BufferView',
        'parameters': {
            '@type': 'array',
            '@index': 0,
            '@slice': (0, 230),
            '@shape': (230, ),
        },
        'InputLayer': {
            '@type': 'BufferView',
            '@index': 1,
            'inputs': {'@type': 'BufferView', '@index': 0},
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2), '@slice': (0, 2)},
            },
            'parameters': {'@type': 'BufferView', '@index': 2},
            'internals': {'@type': 'BufferView', '@index': 3},
        },
        'A': {
            '@type': 'BufferView',
            '@index': 2,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2), '@slice': (0, 2)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 3), '@slice': (2, 5)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (2, 3),
                      '@slice': (0, 6)},
                'b': {'@type': 'array', '@index': 1, '@shape': (3,),
                      '@slice': (6, 9)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                'Ha': {'@type': 'array', '@index': 0, '@shape': ('T', 'B', 3),
                       '@slice': (17, 20)}
            },
        },
        'B': {
            '@type': 'BufferView',
            '@index': 3,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2), '@slice': (0, 2)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 5), '@slice': (5, 10)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (2, 5),
                      '@slice': (9, 19)},
                'b': {'@type': 'array', '@index': 1, '@shape': (5,),
                      '@slice': (19, 24)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                'Ha': {'@type': 'array', '@index': 0, '@shape': ('T', 'B', 5),
                       '@slice': (20, 25)}
            },
        },
        'C': {
            '@type': 'BufferView',
            '@index': 4,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 8), '@slice': (2, 10)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 7), '@slice': (10, 17)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (8, 7),
                      '@slice': (24, 80)},
                'b': {'@type': 'array', '@index': 1, '@shape': (7,),
                      '@slice': (80, 87)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                'Ha': {'@type': 'array', '@index': 0, '@shape': ('T', 'B', 7),
                       '@slice': (25, 32)}
            },
        },
        'D': {
            '@type': 'BufferView',
            '@index': 5,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 12), '@slice': (5, 17)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 11), '@slice': (32, 43)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (12, 11),
                      '@slice': (87, 219)},
                'b': {'@type': 'array', '@index': 1, '@shape': (11,),
                      '@slice': (219, 230)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                'Ha': {'@type': 'array', '@index': 0, '@shape': ('T', 'B', 11),
                       '@slice': (43, 54)}
            },
        }}
