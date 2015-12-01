#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import pytest

from brainstorm.structure.layout import (Hub, create_layout,
                                         create_layout_stub,
                                         gather_array_nodes, get_all_sources,
                                         get_connections, get_forced_orders,
                                         get_forward_closure, get_order,
                                         get_parameter_order,
                                         merge_connections)


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
    assert get_parameter_order('Input', layers['Input']) == ()
    assert get_parameter_order('A', layers['A']) == ('A.parameters.W',
                                                     'A.parameters.bias')
    assert get_parameter_order('B', layers['B']) == ('B.parameters.W',
                                                     'B.parameters.bias')


def test_get_forced_orders(layers):
    assert get_forced_orders(layers) == [
        ('A.parameters.W', 'A.parameters.bias'),
        ('B.parameters.W', 'B.parameters.bias'),
        ('C.parameters.W', 'C.parameters.bias'),
        ('D.parameters.W', 'D.parameters.bias'),
        ('A.gradients.W', 'A.gradients.bias'),
        ('B.gradients.W', 'B.gradients.bias'),
        ('C.gradients.W', 'C.gradients.bias'),
        ('D.gradients.W', 'D.gradients.bias')
    ]


def test_get_connections(layers):
    assert get_connections(layers) == [
        ('A.gradients.W', 'gradients'),
        ('A.gradients.bias', 'gradients'),
        ('A.output_deltas.default', 'C.input_deltas.default'),
        ('A.outputs.default', 'C.inputs.default'),
        ('A.parameters.W', 'parameters'),
        ('A.parameters.bias', 'parameters'),

        ('B.gradients.W', 'gradients'),
        ('B.gradients.bias', 'gradients'),
        ('B.output_deltas.default', 'C.input_deltas.default'),
        ('B.output_deltas.default', 'D.input_deltas.default'),
        ('B.outputs.default', 'C.inputs.default'),
        ('B.outputs.default', 'D.inputs.default'),
        ('B.parameters.W', 'parameters'),
        ('B.parameters.bias', 'parameters'),

        ('C.gradients.W', 'gradients'),
        ('C.gradients.bias', 'gradients'),
        ('C.output_deltas.default', 'D.input_deltas.default'),
        ('C.outputs.default', 'D.inputs.default'),
        ('C.parameters.W', 'parameters'),
        ('C.parameters.bias', 'parameters'),

        ('D.gradients.W', 'gradients'),
        ('D.gradients.bias', 'gradients'),
        ('D.parameters.W', 'parameters'),
        ('D.parameters.bias', 'parameters'),

        ('Input.output_deltas.default', 'A.input_deltas.default'),
        ('Input.output_deltas.default', 'B.input_deltas.default'),
        ('Input.outputs.default', 'A.inputs.default'),
        ('Input.outputs.default', 'B.inputs.default')
    ]


def test_get_all_sinks_and_sources(layers):
    forced_orders = get_forced_orders(layers)
    connections = get_connections(layers)
    layout = create_layout_stub(layers)
    all_sources = get_all_sources(forced_orders, connections, layout)

    assert all_sources == [
        'Input.outputs.default',
        'Input.output_deltas.default',
        'A.outputs.default',
        ('A.parameters.W', 'A.parameters.bias'),
        'A.output_deltas.default',
        ('A.gradients.W', 'A.gradients.bias'),
        'B.outputs.default',
        ('B.parameters.W', 'B.parameters.bias'),
        'B.output_deltas.default',
        ('B.gradients.W', 'B.gradients.bias'),
        'C.outputs.default',
        ('C.parameters.W', 'C.parameters.bias'),
        'C.output_deltas.default',
        ('C.gradients.W', 'C.gradients.bias'),
        'D.outputs.default',
        ('D.parameters.W', 'D.parameters.bias'),
        'D.output_deltas.default',
        ('D.gradients.W', 'D.gradients.bias')]


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
        ('A', 'B'),  # #       /-> C -+
        ('B', 'C'),  # #      /       |
        ('B', 'D'),  # # A -> B       +-> E
        ('C', 'E'),  # #      \       |
        ('D', 'E'),  # #       \-> D -+--> F
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
    assert Hub.can_be_connected_with_single_buffer(np.array(col)
                                                   .reshape(-1, 1)) == expected


def test_can_be_connected_with_single_buffer():
    con_table = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]]).T
    assert Hub.can_be_connected_with_single_buffer(con_table)

    con_table = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0],  # < the bad boy
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]]).T
    assert not Hub.can_be_connected_with_single_buffer(con_table)


def test_permute_rows1():
    h = Hub([0, 1, 2, 3, 4], [0, 1, 2, [3, 4]], [1, 2, 3, 4, 5], 0)

    h.connection_table = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1]])
    h.permute_rows()
    assert h.flat_sources == [0, 2, 1, 3, 4]
    assert h.perm == [0, 2, 1, 3, 4]
    # noinspection PyTypeChecker
    assert np.all(h.connection_table == np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1]]))


def test_permute_rows2():
    h = Hub([0, 1, 2, 3, 4], [0, 1, 2, [3, 4]], [1, 2, 3, 4, 5], 0)

    h.connection_table = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1]])
    h.permute_rows()
    assert h.flat_sources == [0, 1, 3, 4, 2]
    assert h.perm == [0, 1, 3, 4, 2]
    # noinspection PyTypeChecker
    assert np.all(h.connection_table == np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1]]))


def test_create_layout_stub(layers):
    layout = create_layout_stub(layers)
    assert layout == {
        '@type': 'BufferView',
        'parameters': {
            '@type': 'array',
            '@index': 0
        },
        'gradients': {
            '@type': 'array',
            '@index': 1,
            '@is_backward_only': True
        },
        'Input': {
            '@type': 'BufferView',
            '@index': 2,
            'inputs': {'@type': 'BufferView', '@index': 0},
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2)},
            },
            'parameters': {'@type': 'BufferView', '@index': 2},
            'internals': {'@type': 'BufferView', '@index': 3},
            'input_deltas': {'@type': 'BufferView', '@index': 4},
            'output_deltas': {
                '@type': 'BufferView',
                '@index': 5,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2),
                            '@is_backward_only': True},
            },
            'gradients': {'@type': 'BufferView', '@index': 6},
        },
        'A': {
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
                            '@shape': ('T', 'B', 3)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (3, 2)},
                'bias': {'@type': 'array', '@index': 1, '@shape': (3,)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
            },
            'input_deltas': {
                '@type': 'BufferView',
                '@index': 4,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2),
                            '@is_backward_only': True}
            },
            'output_deltas': {
                '@type': 'BufferView',
                '@index': 5,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 3),
                            '@is_backward_only': True}
            },
            'gradients': {
                '@type': 'BufferView',
                '@index': 6,
                'W': {'@type': 'array', '@index': 0, '@shape': (3, 2),
                      '@is_backward_only': True},
                'bias': {'@type': 'array', '@index': 1, '@shape': (3,),
                         '@is_backward_only': True}
            },
        },
        'B': {
            '@type': 'BufferView',
            '@index': 4,
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
                'W': {'@type': 'array', '@index': 0, '@shape': (5, 2)},
                'bias': {'@type': 'array', '@index': 1, '@shape': (5,)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
            },
            'input_deltas': {
                '@type': 'BufferView',
                '@index': 4,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2),
                            '@is_backward_only': True}
            },
            'output_deltas': {
                '@type': 'BufferView',
                '@index': 5,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 5),
                            '@is_backward_only': True}
            },
            'gradients': {
                '@type': 'BufferView',
                '@index': 6,
                'W': {'@type': 'array', '@index': 0, '@shape': (5, 2),
                      '@is_backward_only': True},
                'bias': {'@type': 'array', '@index': 1, '@shape': (5,),
                         '@is_backward_only': True}
            },
        },
        'C': {
            '@type': 'BufferView',
            '@index': 5,
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
                'W': {'@type': 'array', '@index': 0, '@shape': (7, 8)},
                'bias': {'@type': 'array', '@index': 1, '@shape': (7,)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
            },
            'input_deltas': {
                '@type': 'BufferView',
                '@index': 4,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 8),
                            '@is_backward_only': True}
            },
            'output_deltas': {
                '@type': 'BufferView',
                '@index': 5,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 7),
                            '@is_backward_only': True}
            },
            'gradients': {
                '@type': 'BufferView',
                '@index': 6,
                'W': {'@type': 'array', '@index': 0, '@shape': (7, 8),
                      '@is_backward_only': True},
                'bias': {'@type': 'array', '@index': 1, '@shape': (7,),
                         '@is_backward_only': True}
            },
        },
        'D': {
            '@type': 'BufferView',
            '@index': 6,
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
                'W': {'@type': 'array', '@index': 0, '@shape': (11, 12)},
                'bias': {'@type': 'array', '@index': 1, '@shape': (11,)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3
            },
            'input_deltas': {
                '@type': 'BufferView',
                '@index': 4,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 12),
                            '@is_backward_only': True}
            },
            'output_deltas': {
                '@type': 'BufferView',
                '@index': 5,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 11),
                            '@is_backward_only': True}
            },
            'gradients': {
                '@type': 'BufferView',
                '@index': 6,
                'W': {'@type': 'array', '@index': 0, '@shape': (11, 12),
                      '@is_backward_only': True},
                'bias': {'@type': 'array', '@index': 1, '@shape': (11,),
                         '@is_backward_only': True}
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
                '@index': 3
            }
        }
    }
    assert set(gather_array_nodes(layout)) == {
        'inp.outputs.default', 'A.inputs.default', 'A.outputs.default',
        'A.parameters.W', 'A.parameters.b'}


def test_create_layout(layers):
    hubs, layout = create_layout(layers)
    assert layout == {
        '@type': 'BufferView',
        'parameters': {
            '@type': 'array',
            '@index': 0,
            '@hub': 0,
            '@slice': (0, 230),
            '@shape': (230, ),
        },
        'gradients': {
            '@type': 'array',
            '@index': 1,
            '@hub': 4,
            '@slice': (0, 230),
            '@shape': (230, ),
            '@is_backward_only': True
        },

        'Input': {
            '@type': 'BufferView',
            '@index': 2,
            'inputs': {'@type': 'BufferView', '@index': 0},
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2),
                            '@hub': 1, '@slice': (0, 2)},
            },
            'parameters': {'@type': 'BufferView', '@index': 2},
            'internals': {'@type': 'BufferView', '@index': 3},
            'input_deltas': {'@type': 'BufferView', '@index': 4},
            'output_deltas': {
                '@type': 'BufferView',
                '@index': 5,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2),
                            '@hub': 5, '@slice': (0, 2),
                            '@is_backward_only': True},
            },
            'gradients': {'@type': 'BufferView', '@index': 6},
        },
        'A': {
            '@type': 'BufferView',
            '@index': 3,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2),
                            '@hub': 1, '@slice': (0, 2)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 3),
                            '@hub': 2, '@slice': (0, 3)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (3, 2),
                      '@hub': 0, '@slice': (0, 6)},
                'bias': {'@type': 'array', '@index': 1, '@shape': (3,),
                         '@hub': 0, '@slice': (6, 9)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
            },
            'input_deltas': {
                '@type': 'BufferView',
                '@index': 4,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2),
                            '@hub': 5, '@slice': (0, 2),
                            '@is_backward_only': True}
            },
            'output_deltas': {
                '@type': 'BufferView',
                '@index': 5,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 3),
                            '@hub': 6, '@slice': (0, 3),
                            '@is_backward_only': True}
            },
            'gradients': {
                '@type': 'BufferView',
                '@index': 6,
                'W': {'@type': 'array', '@index': 0, '@shape': (3, 2),
                      '@hub': 4, '@slice': (0, 6),
                      '@is_backward_only': True},
                'bias': {'@type': 'array', '@index': 1, '@shape': (3,),
                         '@hub': 4, '@slice': (6, 9),
                         '@is_backward_only': True}
            },
        },
        'B': {
            '@type': 'BufferView',
            '@index': 4,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2),
                            '@hub': 1, '@slice': (0, 2)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 5),
                            '@hub': 2, '@slice': (3, 8)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (5, 2),
                      '@hub': 0, '@slice': (9, 19)},
                'bias': {'@type': 'array', '@index': 1, '@shape': (5,),
                         '@hub': 0, '@slice': (19, 24)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
            },
            'input_deltas': {
                '@type': 'BufferView',
                '@index': 4,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 2),
                            '@hub': 5, '@slice': (0, 2),
                            '@is_backward_only': True}
            },
            'output_deltas': {
                '@type': 'BufferView',
                '@index': 5,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 5),
                            '@hub': 6, '@slice': (3, 8),
                            '@is_backward_only': True}
            },
            'gradients': {
                '@type': 'BufferView',
                '@index': 6,
                'W': {'@type': 'array', '@index': 0, '@shape': (5, 2),
                      '@hub': 4, '@slice': (9, 19),
                      '@is_backward_only': True},
                'bias': {'@type': 'array', '@index': 1, '@shape': (5,),
                         '@hub': 4, '@slice': (19, 24),
                         '@is_backward_only': True}
            },
        },
        'C': {
            '@type': 'BufferView',
            '@index': 5,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 8),
                            '@hub': 2, '@slice': (0, 8)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 7),
                            '@hub': 2, '@slice': (8, 15)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (7, 8),
                      '@hub': 0, '@slice': (24, 80)},
                'bias': {'@type': 'array', '@index': 1, '@shape': (7,),
                         '@hub': 0, '@slice': (80, 87)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
            },
            'input_deltas': {
                '@type': 'BufferView',
                '@index': 4,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 8),
                            '@hub': 6, '@slice': (0, 8),
                            '@is_backward_only': True}
            },
            'output_deltas': {
                '@type': 'BufferView',
                '@index': 5,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 7),
                            '@hub': 6, '@slice': (8, 15),
                            '@is_backward_only': True}
            },
            'gradients': {
                '@type': 'BufferView',
                '@index': 6,
                'W': {'@type': 'array', '@index': 0, '@shape': (7, 8),
                      '@hub': 4, '@slice': (24, 80),
                      '@is_backward_only': True},
                'bias': {'@type': 'array', '@index': 1, '@shape': (7,),
                         '@hub': 4, '@slice': (80, 87),
                         '@is_backward_only': True}
            },
        },
        'D': {
            '@type': 'BufferView',
            '@index': 6,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 12),
                            '@hub': 2, '@slice': (3, 15)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 11),
                            '@hub': 3, '@slice': (0, 11)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                'W': {'@type': 'array', '@index': 0, '@shape': (11, 12),
                      '@hub': 0, '@slice': (87, 219)},
                'bias': {'@type': 'array', '@index': 1, '@shape': (11,),
                         '@hub': 0, '@slice': (219, 230)}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
            },
            'input_deltas': {
                '@type': 'BufferView',
                '@index': 4,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 12),
                            '@hub': 6, '@slice': (3, 15),
                            '@is_backward_only': True}
            },
            'output_deltas': {
                '@type': 'BufferView',
                '@index': 5,
                'default': {'@type': 'array', '@index': 0,
                            '@shape': ('T', 'B', 11),
                            '@hub': 7, '@slice': (0, 11),
                            '@is_backward_only': True}
            },
            'gradients': {
                '@type': 'BufferView',
                '@index': 6,
                'W': {'@type': 'array', '@index': 0, '@shape': (11, 12),
                      '@hub': 4, '@slice': (87, 219),
                      '@is_backward_only': True},
                'bias': {'@type': 'array', '@index': 1, '@shape': (11,),
                         '@hub': 4, '@slice': (219, 230),
                         '@is_backward_only': True}
            },
        }}
