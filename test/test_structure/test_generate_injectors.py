#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.construction import (ConstructionLayer,
                                               ConstructionInjector)
from brainstorm.structure.architecture import (get_injector_description,
                                               generate_injectors)
from brainstorm.describable import create_from_description
from brainstorm.injectors.core import Injector


def test_get_layer_description():
    l1 = ConstructionLayer('layertype', 10, name='foo')
    i = ConstructionInjector('injector_type', 'my_targets', some_kwarg=12)
    l1 << i
    descr = get_injector_description(i)
    assert descr == {
        '@type': 'injector_type',
        'target_from': 'my_targets',
        'layer': 'foo',
        'some_kwarg': 12
    }


def test_generate_injectors():
    l1 = ConstructionLayer('layertype', 10, name='foo1')
    l2 = ConstructionLayer('layertype', 10, name='foo2')
    i1 = ConstructionInjector('injector_type_A', 'mse_targets', name='mse')
    i2 = ConstructionInjector('injector_type_B', 'cee_targets', name='cee')
    l1 << i1
    l1 >> l2 << i2

    injects1 = generate_injectors(l1)
    injects2 = generate_injectors(l2)
    assert injects1 == injects2
    assert injects1 == {
        'mse': {
            '@type': 'injector_type_A',
            'target_from': 'mse_targets',
            'layer': 'foo1',
        },
        'cee': {
            '@type': 'injector_type_B',
            'target_from': 'cee_targets',
            'layer': 'foo2',
        }
    }


def test_injectors_create_from_description():
    injector_descr = {
        'mse1': {
            '@type': 'MeanSquaredError',
            'target_from': 'mse_targets',
            'layer': 'foo1',
        },
        'mse2': {
            '@type': 'MeanSquaredError',
            'target_from': 'other_mse_targets',
            'layer': 'foo2',
        }}
    injectors = create_from_description(injector_descr)
    assert set(injectors.keys()) == {'mse1', 'mse2'}
    assert isinstance(injectors['mse1'], Injector)
    assert isinstance(injectors['mse2'], Injector)
    assert injectors['mse1'].__class__.__name__ == 'MeanSquaredError'
    assert injectors['mse2'].__class__.__name__ == 'MeanSquaredError'
    assert injectors['mse1'].layer == 'foo1'
    assert injectors['mse2'].layer == 'foo2'
    assert injectors['mse1'].target_from == 'mse_targets'
    assert injectors['mse2'].target_from == 'other_mse_targets'
