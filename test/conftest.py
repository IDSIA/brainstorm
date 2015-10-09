#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import pytest

from brainstorm.structure.architecture import \
    instantiate_layers_from_architecture


def pytest_addoption(parser):
    parser.addoption("--skipslow", action="store_true",
                     help="skip slow tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and item.config.getoption("--skipslow"):
        pytest.skip("skipped because of --skipslow option")


#        /--- A -- C--
# Input -        /    \
#        \--- B ------- D

@pytest.fixture
def layers():
    arch = {
        'Input': {
            '@type': 'Input',
            'out_shapes': {'default': ('T', 'B', 2)},
            '@outgoing_connections': ['A', 'B']
        },
        'A': {
            '@type': 'FullyConnected',
            'size': 3,
            '@outgoing_connections': ['C']
        },
        'B': {
            '@type': 'FullyConnected',
            'size': 5,
            '@outgoing_connections': ['C', 'D']
        },
        'C': {
            '@type': 'FullyConnected',
            'size': 7,
            '@outgoing_connections': ['D']
        },
        'D': {
            '@type': 'FullyConnected',
            'size': 11,
            '@outgoing_connections': []
        }
    }
    return instantiate_layers_from_architecture(arch)
