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
