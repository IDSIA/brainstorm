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
            '@type': 'InputLayer',
            'out_shape': 2,
            'sink_layers': {'A', 'B'}
        },
        'A': {
            '@type': 'FeedForwardLayer',
            'out_shape': 3,
            'sink_layers': {'C'}
        },
        'B': {
            '@type': 'FeedForwardLayer',
            'out_shape': 5,
            'sink_layers': {'C', 'D'}
        },
        'C': {
            '@type': 'FeedForwardLayer',
            'out_shape': 7,
            'sink_layers': {'D'}
        },
        'D': {
            '@type': 'FeedForwardLayer',
            'out_shape': 11,
            'sink_layers': set()
        }
    }
    return instantiate_layers_from_architecture(arch)
