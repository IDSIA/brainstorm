#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
from structure.view_references import (
    get_regex_for_reference, get_key_to_references_mapping,
    resolve_references)


@pytest.mark.parametrize('ref', ['FeedForwardLayer',
                                 'OutputLayer_12',
                                 '_a12_34bcx'])
def test_get_regex_for_reference_non_starred(ref):
    expr = get_regex_for_reference(ref)
    assert expr.match(ref)


@pytest.mark.parametrize('ref', ['layer', 'layer12', 'layer_23'])
def test_get_regex_for_reference_star_at_end(ref):
    expr = get_regex_for_reference('layer*')
    assert expr.match(ref)
    assert not expr.match('something' + ref)


@pytest.mark.parametrize('ref', ['LstmLayer_12', '12', 'foo12'])
def test_get_regex_for_reference_star_at_beginning(ref):
    expr = get_regex_for_reference('*12')
    assert expr.match(ref)
    assert not expr.match(ref + 'something')


@pytest.mark.parametrize('ref', ['LeftLstmLayer', 'LeftFooLayer', 'LeftLayer'])
def test_get_regex_for_reference_star_in_mid(ref):
    expr = get_regex_for_reference('Left*Layer')
    assert expr.match(ref)
    assert not expr.match('something' + ref)
    assert not expr.match(ref + 'something')


def test_get_key_to_references_mapping():
    keys = ['InputLayer', 'ForwardLayer_1', 'ForwardLayer_2']
    references = ['InputLayer', 'Forward*', '*Layer_2', '*']
    key_to_refs = get_key_to_references_mapping(keys, references)
    assert key_to_refs == {
        'InputLayer': {'InputLayer', '*'},
        'ForwardLayer_1': {'Forward*', '*'},
        'ForwardLayer_2': {'Forward*', '*Layer_2', '*'}
    }


def test_get_key_to_references_mapping_default():
    keys = ['InputLayer', 'ForwardLayer_1', 'ForwardLayer_2']
    references = ['*Layer_2', 'default']
    key_to_refs = get_key_to_references_mapping(keys, references)
    assert key_to_refs == {
        'InputLayer': {'default'},
        'ForwardLayer_1': {'default'},
        'ForwardLayer_2': {'*Layer_2'}
    }


def test_get_key_to_references_mapping_raises_non_matching_ref():
    keys = ['InputLayer', 'ForwardLayer_1', 'ForwardLayer_2']
    references = ['*Layer_2', 'Lstm*']
    with pytest.raises(AssertionError):
        key_to_refs = get_key_to_references_mapping(keys, references)


def test_resolve_references1():
    refs = {'*_bias': 2, 'IX': 1, 'default': 0}
    struct = {'IX': None, 'OX': None, 'I_bias': None, 'O_bias': None}
    full_thing = resolve_references(struct, refs)
    assert full_thing == {'IX': {1}, 'OX': {0}, 'I_bias': {2}, 'O_bias': {2}}


def test_resolve_references2():
    refs = {'*_bias': 2, 'I_bias': 1, 'default': 0}
    keys = {'IX': None, 'OX': None, 'I_bias': None, 'O_bias': None}
    full_thing = resolve_references(keys, refs)
    assert full_thing == {'IX': {0}, 'OX': {0}, 'I_bias': {1, 2}, 'O_bias': {2}}


def test_resolve_references_parent_default():
    refs = {'FooLayer': {'HX': 0}, 'default': 1}
    keys = {'FooLayer': {'HX': None, 'H_bias': None},
            'BarLayer': {'HX': None, 'H_bias': None}}
    full_thing = resolve_references(keys, refs)
    assert full_thing == {
        'FooLayer': {'HX': {0}, 'H_bias': {1}},
        'BarLayer': {'HX': {1}, 'H_bias': {1}}
    }


def test_resolve_referencese():
    refs = {'LstmLayer*': {'IX': 1},
            '*Layer*': {'*_bias': 2},
            '*_1': {'I_bias': [4, 5]},
            'ForwardLayer': {'H_bias': 3, 'default': 6},
            '*_2': 7,
            'default': 0}

    keys = {
        'LstmLayer_1': {'IX': None, 'OX': None, 'I_bias': None, 'O_bias': None},
        'LstmLayer_2': {'IX': None, 'OX': None, 'I_bias': None, 'O_bias': None},
        'ForwardLayer': {'HX': None, 'H_bias': None},
        'FooLayer': {'bar': None, 'bar_bias': None},
    }
    full_thing = resolve_references(keys, refs)
    assert full_thing == {
        'LstmLayer_1': {'IX':{1}, 'OX': {0}, 'I_bias': {2, 4, 5}, 'O_bias':{2}},
        'LstmLayer_2': {'IX':{1, 7}, 'OX': {7}, 'I_bias': {2, 7}, 'O_bias':{2, 7}},
        'ForwardLayer': {'HX':{6}, 'H_bias':{2, 3}},
        'FooLayer': {'bar':{0}, 'bar_bias':{2}}
    }