#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals


class UniquelyNamed(object):
    """
    An object that maintains a scope of names and ensures that its name is
    unique within that scope by appending an appropriate index.

    If there are no collisions then its name is the same as the given basename.
    If there are multiple objects with the same name in the scope, then
    its name is the basename + _index where index is a number given according
    to the order in which the objects where created.
    """

    global_counter = 0

    def __init__(self, basename):
        self._basename = basename
        self.scope = {basename: [self]}
        self.creation_id = UniquelyNamed.global_counter
        UniquelyNamed.global_counter += 1

    def merge_scopes(self, other):
        new_scope = self.scope
        for name, scoped_names in other.scope.items():
            if name not in self.scope:
                new_scope[name] = []
            new_scope[name] = sorted(set(self.scope[name] + scoped_names),
                                     key=lambda x: x.creation_id)
        for n in new_scope:
            for sn in new_scope[n]:
                sn.scope = new_scope

    @property
    def name(self):
        if len(self.scope[self._basename]) == 1:
            return self._basename

        i = 1
        for un in self.scope[self._basename]:
            name = "{}_{}".format(self._basename, i)
            # see if this derived name is already taken
            # increase the index if need be
            while name in self.scope and len(self.scope[name]) == 1:
                i += 1
                name = "{}_{}".format(self._basename, i)
            if un is self:
                return name
            i += 1
