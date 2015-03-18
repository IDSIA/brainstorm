#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import pytest

from brainstorm.structure.architecture import (
    instantiate_layers_from_architecture)


#             /--- A -- C--
# InputLayer -        /    \
#             \--- B ------- D

@pytest.fixture
def layers():
    arch = {
        'InputLayer': {
            '@type': 'DataLayer',
            'shape': 2,
            '@outgoing_connections': {'A', 'B'}
        },
        'A': {
            '@type': 'FeedForwardLayer',
            'shape': 3,
            '@outgoing_connections': {'C'}
        },
        'B': {
            '@type': 'FeedForwardLayer',
            'shape': 5,
            '@outgoing_connections': {'C', 'D'}
        },
        'C': {
            '@type': 'FeedForwardLayer',
            'shape': 7,
            '@outgoing_connections': {'D'}
        },
        'D': {
            '@type': 'FeedForwardLayer',
            'shape': 11,
            '@outgoing_connections': set()
        }
    }
    return instantiate_layers_from_architecture(arch)



joint_layout = {
    'sizes': (45, 0, 110),
    'layout': [
        ('InputLayer', {'layout': [
            ('parameters', {'layout': []}),
            ('inputs', {'layout': []}),
            ('outputs', {'slice': (2, 0, 14), 'layout': [
                ('input_data', {'slice': (2, 0, 4),   'shape': (4,)}),
                ('targets',    {'slice': (2, 10, 14), 'shape': (4,)})
            ]}),
            ('state', {'layout': []}),
        ]}),
        ('RnnLayer', {'layout': [
            ('parameters', {'slice': (0, 0, 50), 'layout': [
                ('W', {'slice': (0, 0, 20),  'shape': (4, 5)}),
                ('R', {'slice': (0, 20, 45), 'shape': (5, 5)}),
                ('b', {'slice': (0, 45, 50), 'shape': (5,  )})
            ]}),
            ('inputs', {'slice': (2, 0, 4), 'layout': [
                ('default', {'slice': (2, 0, 4), 'shape': (4,)})
            ]}),
            ('outputs', {'slice': (2, 14, 19), 'layout': [
                ('default', {'tslice': (2, 14, 19), 'shape': (5,)})
            ]}),
            ('state', {'slice': (2, 30, 35), 'layout': [
                ('Ha', {'slice': (2, 30, 35), 'shape': (5,)})
            ]}),
        ]}),
        ('OutLayer', {'layout': [
            ('parameters', {'slice': (0, 50, 110), 'layout': [
                ('W', {'slice': (0, 50, 100),  'shape': (5, 10)}),
                ('b', {'slice': (0, 100, 110), 'shape': (10,  )})
            ]}),
            ('inputs', {'slice': (2, 14, 19), 'layout': [
                ('default', {'slice': (2, 14, 19), 'shape': (5,)})
            ]}),
            ('outputs', {'slice': (2, 19, 29), 'layout': [
                ('default', {'slice': (2, 19, 29), 'shape': (10,)})
            ]}),
            ('state', {'slice': (2, 35, 45), 'layout': [
                ('Ha', {'slice': (2, 35, 55), 'shape': (10,)})
            ]})
        ]}),
        ('MseLayer', {'layout': [
            ('parameters', {'layout': []}),
            ('inputs', {'layout': [
                ('net_out', {'slice': (2, 19, 29), 'shape': (10,)}),
                ('targets', {'slice': (2, 10, 14), 'shape': (10,)}),
            ]}),
            ('outputs', {'slice': (2, 29, 30), 'layout': [
                ('default', {'slice': (2, 29, 30), 'shape': (1,)})
            ]}),
            ('state', {'layout': []}),
        ]})
    ]
}