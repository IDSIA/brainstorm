#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import json

import numpy as np
import pytest

import brainstorm as bs
from brainstorm.describable import (Describable, create_from_description,
                                    get_description)
from brainstorm.handlers.numpy_handler import NumpyHandler
from brainstorm.optional import has_pycuda
# ######################### get_all_undescribed ###############################
from brainstorm.structure.architecture import generate_architecture


def test_get_all_undescribed_on_empty_describable():
    class Foo1(Describable):
        pass

    assert Foo1.__get_all_undescribed__() == {}


def test_get_all_undescribed_on_describable_with_set():
    class Foo2(Describable):
        __undescribed__ = {'a', 'b'}

    assert Foo2.__get_all_undescribed__() == {'a': None, 'b': None}


def test_get_all_undescribed_on_describable_with_dict():
    class Foo3(Describable):
        __undescribed__ = {'a': 12, 'b': 23}

    assert Foo3.__get_all_undescribed__() == {'a': 12, 'b': 23}


def test_get_all_undescribed_on_inherited_describable():
    class Foo4(Describable):
        __undescribed__ = {'a'}

    class Bar1(Foo4):
        __undescribed__ = {'b': 23}

    assert Bar1.__get_all_undescribed__() == {'a': None, 'b': 23}


def test_get_all_undescribed_on_inherited_describable_overrides():
    class Foo5(Describable):
        __undescribed__ = {'a': 10}

    class Bar2(Foo5):
        __undescribed__ = {'a': 11}

    assert Bar2.__get_all_undescribed__() == {'a': 11}


# ##################### get_all_default_values ##############################

def test_get_all_default_values_on_empty_describable():
    class Foo6(Describable):
        pass

    assert Foo6.__get_all_default_values__() == {}


def test_get_all_default_values_on_simple_describable():
    class Foo7(Describable):
        __default_values__ = {'a': 10, 'b': 42}

    assert Foo7.__get_all_default_values__() == {'a': 10, 'b': 42}


def test_get_all_default_values_on_inherited_describable():
    class Foo8(Describable):
        __default_values__ = {'a': 10}

    class Bar3(Foo8):
        __default_values__ = {'b': 42}

    assert Bar3.__get_all_default_values__() == {'a': 10, 'b': 42}


def test_get_all_default_values_on_inherited_describable_overrides():
    class Foo9(Describable):
        __default_values__ = {'a': 10}

    class Bar4(Foo9):
        __default_values__ = {'a': 42}

    assert Bar4.__get_all_default_values__() == {'a': 42}


# ##################### __describe__ ##############################

def test_describe_on_empty_describable():
    class Foo10(Describable):
        pass

    assert Foo10().__describe__() == {'@type': 'Foo10'}


def test_describe_on_describable_with_attributes():
    class Foo11(Describable):
        def __init__(self):
            self.a = 10
            self.b = 'bar'

    assert Foo11().__describe__() == {'@type': 'Foo11', 'a': 10, 'b': 'bar'}


def test_describe_on_describable_with_ignored_attributes():
    class Foo12(Describable):
        __undescribed__ = {'a'}

        def __init__(self):
            self.a = 10
            self.b = 'bar'

    assert Foo12().__describe__() == {'@type': 'Foo12', 'b': 'bar'}


def test_describe_on_describable_with_default_attributes():
    class Foo13(Describable):
        __default_values__ = {'a': 10, 'b': 12}

        def __init__(self):
            self.a = 10
            self.b = 'bar'

    assert Foo13().__describe__() == {'@type': 'Foo13', 'b': 'bar'}


# ##################### __new_from_description__ ##############################

def test_new_from_description_on_empty_describable():
    class Foo14(Describable):
        pass

    f = Foo14.__new_from_description__({'@type': 'Foo14'})
    assert isinstance(f, Foo14)


def test_new_from_description_on_describable_with_attributes():
    class Foo15(Describable):
        pass

    f = Foo15.__new_from_description__({'@type': 'Foo15', 'a': 10, 'b': 20})
    assert isinstance(f, Foo15)
    assert f.a == 10
    assert f.b == 20


def test_new_from_description_with_nested_describable_attributes():
    class Foo16(Describable):
        pass

    f = Foo16.__new_from_description__({'@type': 'Foo16',
                                        'a': {'@type': 'Foo16'}})
    assert isinstance(f, Foo16)
    assert isinstance(f.a, Foo16)


def test_new_from_description_on_describable_with_undescribed():
    class Foo17(Describable):
        __undescribed__ = {'a': 23}

    f = Foo17.__new_from_description__({'@type': 'Foo17'})
    assert isinstance(f, Foo17)
    assert f.a == 23


def test_new_from_description_on_describable_with_default_value():
    class Foo18(Describable):
        __default_values__ = {'a': 23}

    f = Foo18.__new_from_description__({'@type': 'Foo18'})
    assert isinstance(f, Foo18)
    assert f.a == 23


def test_new_from_description_calls_init_from_description():
    init_called = []

    class Foo19(Describable):
        def __init_from_description__(self, description):
            init_called.append(description)

    description = {'@type': 'Foo19'}
    f = Foo19.__new_from_description__(description)
    assert isinstance(f, Foo19)
    assert init_called == [description]


# ##################### get_description ##############################

def test_get_description_basic_types():
    assert get_description(13) == 13
    assert get_description(0xff) == 0xff
    assert get_description(23.5) == 23.5
    assert get_description(1.3e-7) == 1.3e-7
    assert get_description(True) is True
    assert get_description(False) is False
    assert get_description('foo') == 'foo'
    assert get_description(None) is None


def test_get_description_list():
    assert get_description([]) == []
    assert get_description([1, 2, 3]) == [1, 2, 3]
    assert get_description([1, 'b']) == [1, 'b']


def test_get_description_dict():
    assert get_description({}) == {}
    assert get_description({'a': 1, 'b': 'bar'}) == {'a': 1, 'b': 'bar'}


def test_get_description_ndarray():
    assert get_description(np.array(1.7)) == 1.7
    assert get_description(np.array([[1], [2], [3]])) == [[1], [2], [3]]
    assert get_description(np.array([1, 2, 3])) == [1, 2, 3]


def test_get_description_describable():
    class Foo20(Describable):
        pass
    f = Foo20()
    assert get_description(f) == {'@type': 'Foo20'}


def test_get_description_for_undescribable_raises():
    with pytest.raises(TypeError) as excinfo:
        get_description(pytest)

    assert excinfo.value.args[0].find("module") > -1


def test_get_description_with_undescribable_in_list_raises():
    with pytest.raises(TypeError) as excinfo:
        get_description([1, 2, pytest, 4])

    assert excinfo.value.args[0].find("module") > -1
    assert excinfo.value.args[0].find("[2]") > -1


def test_get_description_with_undescribable_in_dict_raises():
    with pytest.raises(TypeError) as excinfo:
        get_description({'foo': pytest})

    assert excinfo.value.args[0].find("module") > -1
    assert excinfo.value.args[0].find("[foo]") > -1


def test_get_description_with_undescribable_in_describable_raises():
    class Foo21(Describable):
        pass
    f = Foo21()
    f.mymod = pytest
    with pytest.raises(TypeError) as excinfo:
        get_description(f)

    assert excinfo.value.args[0].find("module") > -1
    assert excinfo.value.args[0].find("[Foo21.mymod]") > -1


# ##################### create_from_description ##############################

def test_create_from_description():
    assert create_from_description(13) == 13
    assert create_from_description(0xff) == 0xff
    assert create_from_description(23.5) == 23.5
    assert create_from_description(1.3e-7) == 1.3e-7
    assert create_from_description(True) is True
    assert create_from_description(False) is False
    assert create_from_description('foo') == 'foo'
    assert create_from_description(None) is None


def test_create_from_description_list():
    assert create_from_description([]) == []
    assert create_from_description([1, 2, 3]) == [1, 2, 3]
    assert create_from_description([1, 'b']) == [1, 'b']


def test_create_from_description_dict():
    assert create_from_description({}) == {}
    assert create_from_description({'a': 1, 'b': 'ar'}) == {'a': 1, 'b': 'ar'}


def test_create_from_description_describable():
    class Foo22(Describable):
        pass
    f = create_from_description({'@type': 'Foo22'})
    assert isinstance(f, Foo22)


def test_create_from_description_with_undescribable_raises():
    with pytest.raises(TypeError) as excinfo:
        create_from_description({'@type': 'Unknown66'})

    assert excinfo.value.args[0].find("Unknown66") > -1


def test_create_from_description_with_invalid_description_raises():
    with pytest.raises(TypeError) as excinfo:
        create_from_description(pytest)

    assert excinfo.value.args[0].find('module') > -1


# ################# test describing handler ###################################

@pytest.mark.skipif(has_pycuda is False, reason='requires pycuda and skcuda')
def test_describe_pycuda_handler():
    from brainstorm.handlers.pycuda_handler import PyCudaHandler
    pch = PyCudaHandler()
    d = get_description(pch)
    assert d == {'@type': 'PyCudaHandler'}
    pch2 = create_from_description(d)
    assert isinstance(pch2, PyCudaHandler)


def test_describe_numpy_handler():
    nh = NumpyHandler(np.float32)
    d = get_description(nh)
    assert d == {'@type': 'NumpyHandler', 'dtype': 'float32'}
    nh2 = create_from_description(d)
    assert isinstance(nh2, NumpyHandler)
    assert nh2.dtype == np.float32

# ################# test describing a Network #################################

arch = generate_architecture(
    bs.layers.Input(out_shapes={'default': ('T', 'B', 7)}) >>
    bs.layers.FullyConnected(3) >>
    bs.layers.Loss())


def test_describe_network():
    net = bs.Network.from_architecture(arch)
    net.initialize(1)
    assert get_description(net) == {
        '@type': 'Network',
        'architecture': json.loads(json.dumps(arch)),
        'handler': {'@type': 'NumpyHandler', 'dtype': 'float32'},
        'initializers': {'default': 1},
        'weight_modifiers': {},
        'gradient_modifiers': {},
        'output_name': None
    }


def test_get_network_from_description():
    net = bs.Network.from_architecture(arch)
    net.initialize(1)
    d = get_description(net)
    net2 = create_from_description(d)
    assert isinstance(net2, bs.Network)
    assert net2.layers.keys() == net.layers.keys()
    assert isinstance(net2.handler, NumpyHandler)
    assert net2.handler.dtype == np.float32
    assert np.all(net2.buffer.parameters == net.buffer.parameters)


# ################# test describing a Trainer #################################

def test_describe_trainer():
    tr = bs.Trainer(bs.training.SgdStepper(learning_rate=0.7), verbose=False)
    tr.add_hook(bs.hooks.StopAfterEpoch(23))
    tr.add_hook(bs.hooks.StopOnNan())

    d = get_description(tr)
    assert d == {
        '@type': 'Trainer',
        'verbose': False,
        'hooks': {
            'StopAfterEpoch': {
                '@type': 'StopAfterEpoch',
                'max_epochs': 23,
                'priority': 1},
            'StopOnNan': {
                '@type': 'StopOnNan',
                'check_parameters': True,
                'check_training_loss': True,
                'logs_to_check': [],
                'priority': 2}},
        'stepper': {
            '@type': 'SgdStepper',
            'learning_rate': 0.7},
        'train_scorers': []
    }


def test_recreate_trainer_from_description():
    tr = bs.Trainer(bs.training.SgdStepper(learning_rate=0.7), verbose=False)
    tr.add_hook(bs.hooks.StopAfterEpoch(23))
    tr.add_hook(bs.hooks.StopOnNan())

    d = get_description(tr)

    tr2 = create_from_description(d)
    assert isinstance(tr2, bs.Trainer)
    assert tr2.verbose is False
    assert list(tr2.hooks.keys()) == ['StopAfterEpoch', 'StopOnNan']
    assert tr2.hooks['StopAfterEpoch'].max_epochs == 23
    assert isinstance(tr2.stepper, bs.training.SgdStepper)
    assert tr2.stepper.learning_rate == 0.7
