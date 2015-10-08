#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import pickle
from copy import copy

import pytest

from brainstorm.describable import (Describable, create_from_description,
                                    get_description)
from brainstorm.randomness import RandomState, Seedable, global_rnd


# ##################### RandomState ###########################################

@pytest.fixture
def rnd():
    return RandomState(1)


def test_randomstate_constructor_without_arg():
    rnd1 = RandomState()
    rnd2 = RandomState()
    assert rnd1.get_seed() != rnd2.get_seed()


def test_randomstate_constructor_with_seed():
    rnd1 = RandomState(2)
    assert rnd1.get_seed() == 2


def test_randomstate_set_seed(rnd):
    rnd.set_seed(23)
    assert rnd.get_seed() == 23


def test_randomstate_randint_randomness(rnd):
    a = rnd.randint(10000)
    b = rnd.randint(10000)
    assert a != b


def test_randomstate_seeded_randint_deterministic(rnd):
    rnd.set_seed(1)
    a = rnd.randint(10000)
    rnd.set_seed(1)
    b = rnd.randint(10000)
    assert a == b


def test_randomstate_reset_randint_deterministic(rnd):
    a = rnd.randint(10000)
    rnd.reset()
    b = rnd.randint(10000)
    assert a == b


def test_randomstate_get_new_random_state_randomness(rnd):
    rnd1 = rnd.create_random_state()
    rnd2 = rnd.create_random_state()
    assert rnd1.get_seed() != rnd2.get_seed


def test_randomstate_seeded_get_new_random_state_deterministic(rnd):
    rnd1 = rnd.create_random_state()
    rnd.reset()
    rnd2 = rnd.create_random_state()
    assert rnd1.get_seed() == rnd2.get_seed()


# ################## global_rnd ###############################################

def test_global_rnd_exists():
    assert isinstance(global_rnd, RandomState)


# ################## Seedable #################################################

def test_seedable_constructor_without_seed():
    seedable1 = Seedable()
    seedable2 = Seedable()
    assert seedable1.rnd.get_seed() != seedable2.rnd.get_seed()


def test_seedable_constructor_with_seed():
    seedable = Seedable(1)
    assert seedable.rnd.get_seed() == 1


def test_seedable_description_does_not_include_rnd1():
    class Foo0(Seedable):
        pass

    assert get_description(Foo0()) == {'@type': 'Foo0'}


def test_seedable_description_does_not_include_rnd():
    class Foo1(Seedable, Describable):
        pass

    assert get_description(Foo1()) == {'@type': 'Foo1'}


def test_seedable_initializes_from_description1():
    class Foo2(Seedable, Describable):
        pass

    f = create_from_description({'@type': 'Foo2'})
    assert hasattr(f, 'rnd')
    assert isinstance(f.rnd, RandomState)
    f.rnd.randint(100)  # assert no throw


def test_seedable_initializes_from_description2():
    class Foo3(Seedable, Describable):
        def __init_from_description__(self, description):
            super(Foo3, self).__init_from_description__(description)

    f = create_from_description({'@type': 'Foo3'})
    assert hasattr(f, 'rnd')
    assert isinstance(f.rnd, RandomState)
    f.rnd.randint(100)  # assert no throw


def test_random_state_copyable():
    r = RandomState(127)
    _ = r.randint(10)
    r2 = copy(r)
    assert r.get_seed() == r2.get_seed()
    assert r.generate_seed() == r2.generate_seed()


def test_random_state_pickleable():
    r = RandomState(127)
    _ = r.randint(10)
    s = pickle.dumps(r)
    r2 = pickle.loads(s)
    assert r.get_seed() == r2.get_seed()
    assert r.generate_seed() == r2.generate_seed()
