#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest

from brainstorm.randomness import RandomState, global_rnd


@pytest.fixture
def rnd():
    return RandomState(1)


def test_constructor_without_arg():
    rnd1 = RandomState()
    rnd2 = RandomState()
    assert rnd1.get_seed() != rnd2.get_seed()


def test_constructor_with_seed():
    rnd1 = RandomState(2)
    rnd2 = RandomState(2)
    assert rnd1.get_seed() == rnd2.get_seed()


def test_set_seed(rnd):
    rnd.set_seed(1)
    assert rnd.get_seed() == 1


def test_randint_randomness(rnd):
    a = rnd.randint(10000)
    b = rnd.randint(10000)
    assert a != b


def test_seeded_randint_deterministic(rnd):
    rnd.set_seed(1)
    a = rnd.randint(10000)
    rnd.set_seed(1)
    b = rnd.randint(10000)
    assert a == b


def test_reset_randint_deterministic(rnd):
    a = rnd.randint(10000)
    rnd.reset()
    b = rnd.randint(10000)
    assert a == b


def test_get_new_random_state_randomness(rnd):
    rnd1 = rnd.get_new_random_state()
    rnd2 = rnd.get_new_random_state()

    assert rnd1.get_seed() != rnd2.get_seed


def test_seeded_get_new_random_state_deterministic(rnd):
    rnd.set_seed(1)
    rnd1 = rnd.get_new_random_state()
    rnd.set_seed(1)
    rnd2 = rnd.get_new_random_state()
    assert rnd1.get_seed() == rnd2.get_seed()


def test_get_item_randomness(rnd):
    rnd1 = rnd['A']
    rnd2 = rnd['A']
    rnd1.randint(1000)  != rnd2.randint(1000)


def test_seeded_get_item_deterministic(rnd):
    rnd.set_seed(1)
    rnd1 = rnd['A']
    rnd2 = rnd['A']
    assert rnd1.get_seed() == rnd2.get_seed()


def test_seeded_get_item_deterministic2(rnd):
    rnd.set_seed(1)
    rnd1 = rnd['A']
    rnd2 = rnd['A']
    assert rnd1 == rnd2


def test_get_item_independent_of_previous_usage(rnd):
    rnd.set_seed(1)
    rnd1 = rnd['A']
    rnd.set_seed(1)
    rnd.randint(1000)
    rnd2 = rnd['A']
    assert rnd1 != rnd2
    assert rnd1.get_seed() == rnd2.get_seed()


def test_get_item_different_names(rnd):
    rnd1 = rnd['A']
    rnd2 = rnd['B']
    assert rnd1 != rnd2


# ################## global_rnd ###############################################

def test_global_rnd_randomness():
    assert global_rnd.randint(1000) != global_rnd.randint(1000)


def test_seeded_global_rnd_deterministic():
    global_rnd.set_seed(1)
    a = global_rnd.randint(1000)
    global_rnd.set_seed(1)
    b = global_rnd.randint(1000)
    assert a == b
