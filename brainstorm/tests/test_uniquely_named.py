#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.construction import UniquelyNamed


def test_basename():
    n = UniquelyNamed('my_basename')
    assert n.name == 'my_basename'


def test_merging_scopes_no_conflict():
    n1 = UniquelyNamed('A')
    n2 = UniquelyNamed('B')
    n1.merge_scopes(n2)
    assert n1.name == 'A'
    assert n2.name == 'B'


def test_multimerging_no_conflict():
    n1 = UniquelyNamed('A')
    n2 = UniquelyNamed('B')
    n1.merge_scopes(n2)
    n2.merge_scopes(n1)
    n2.merge_scopes(n1)
    n1.merge_scopes(n2)
    assert n1.name == 'A'
    assert n2.name == 'B'


def test_merging_scopes():
    n1 = UniquelyNamed('my_basename')
    n2 = UniquelyNamed('my_basename')
    n1.merge_scopes(n2)
    assert n1.name == 'my_basename_1'
    assert n2.name == 'my_basename_2'


def test_merging_scopes_symmetric():
    n1 = UniquelyNamed('my_basename')
    n2 = UniquelyNamed('my_basename')
    n2.merge_scopes(n1)
    assert n1.name == 'my_basename_1'
    assert n2.name == 'my_basename_2'


def test_merging_scopes_transitiv():
    n1 = UniquelyNamed('my_basename')
    n2 = UniquelyNamed('my_basename')
    n3 = UniquelyNamed('my_basename')
    n1.merge_scopes(n2)
    n2.merge_scopes(n3)
    assert n1.name == 'my_basename_1'
    assert n2.name == 'my_basename_2'
    assert n3.name == 'my_basename_3'


def test_merging_scopes_transitiv2():
    n1 = UniquelyNamed('my_basename')
    n2 = UniquelyNamed('my_basename')
    n3 = UniquelyNamed('my_basename')
    n4 = UniquelyNamed('my_basename')
    n1.merge_scopes(n2)
    n3.merge_scopes(n4)
    n2.merge_scopes(n4)
    assert n1.name == 'my_basename_1'
    assert n2.name == 'my_basename_2'
    assert n3.name == 'my_basename_3'
    assert n4.name == 'my_basename_4'


def test_sneaky_name_collision():
    n1 = UniquelyNamed('A_2')
    n2 = UniquelyNamed('A')
    n3 = UniquelyNamed('A')
    n4 = UniquelyNamed('A')
    n1.merge_scopes(n2)
    n2.merge_scopes(n3)
    n3.merge_scopes(n4)

    assert n1.name == 'A_2'
    assert n2.name == 'A_1'
    assert n3.name == 'A_3'
    assert n4.name == 'A_4'


def test_no_sneaky_name_collision():
    n0 = UniquelyNamed('A_2')
    n1 = UniquelyNamed('A_2')
    n2 = UniquelyNamed('A')
    n3 = UniquelyNamed('A')
    n4 = UniquelyNamed('A')
    n0.merge_scopes(n1)
    n1.merge_scopes(n2)
    n2.merge_scopes(n3)
    n3.merge_scopes(n4)

    assert n0.name == 'A_2_1'
    assert n1.name == 'A_2_2'
    assert n2.name == 'A_1'
    assert n3.name == 'A_2'
    assert n4.name == 'A_3'


def test_separate_scopes():
    n0 = UniquelyNamed('A')
    n1 = UniquelyNamed('A')
    n2 = UniquelyNamed('A')
    n3 = UniquelyNamed('A')

    n1.merge_scopes(n0)
    n2.merge_scopes(n3)
    assert n0.name == 'A_1'
    assert n1.name == 'A_2'
    assert n2.name == 'A_1'
    assert n3.name == 'A_2'
