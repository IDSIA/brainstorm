#!/usr/bin/env python
# coding=utf-8
"""
This module provides the core functionality for describing objects.

Description
-----------
In brainstorm most objects can be converted into a so called description using
the :func:`get_description` function.
A description is a JSON-serializable data structure that contains all `static`
information to re-create that object using the :func:`create_from_description`
function. It does not, however, contain any `dynamic` information.
This means that an object created from a description is in the same state as
if it had been freshly instantiated, without any later modifications of its
internal state.

The descriptions for the basic types ``int``, ``float``, ``bool``, ``str``,
``list`` and ``dict`` are these values themselves. Other objects need to
inherit from :class:`Describable` and their description is always a ``dict``
containing the ``'@type': 'ClassName'`` key possibly along with other
properties.
"""
from __future__ import division, print_function, unicode_literals

from copy import deepcopy

import numpy as np
import six

from brainstorm.utils import get_inheritors


class Describable(object):
    """
    Base class for all objects that can be described and initialized from a
    description.

    Derived classes can specify the :attr:`__undescribed__` field to
    prevent
    certain attributes from being described. This field can be either a set of
    attribute names, or a dictionary mapping attribute names to their
    initialization value (used when a new object is created from the
    description).

    Derived classes can also specify an :attr:`__default_values__` dict. This
    dict allows for omitting certain attributes from the description if their
    value is equal to that default value.
    """
    __undescribed__ = {}
    """
    Set or dict of attributes that should not be part of the description.
    If specified as a dict, then these attributes are initialized to the
    specified values.
    """

    __default_values__ = {}
    """
    Dict of attributes with their corresponding default values.
    They will only be part of the description if their value differs from their
    default value
    """

    def __describe__(self):
        """
        Returns a description of this object. That is a dictionary
        containing the name of the class as ``@type`` and all members of the
        class. This description is json-serializable.

        If a sub-class of Describable contains non-describable members, it has
        to override this method to specify how it should be described.

        Returns:
            dict: Description of this object
        """
        description = {}
        ignorelist = self.__get_all_undescribed__()
        defaultlist = self.__get_all_default_values__()
        for member, value in self.__dict__.items():
            if member in ignorelist:
                continue
            if member in defaultlist and defaultlist[member] == value:
                continue
            try:
                description[member] = get_description(value)
            except TypeError as err:
                err.args = (err.args[0] + "[{}.{}]".format(
                    self.__class__.__name__, member),)
                raise

        description['@type'] = self.__class__.__name__
        return description

    @classmethod
    def __new_from_description__(cls, description):
        """
        Creates a new object from a given description.

        If a sub-class of Describable contains non-describable fields, it has
        to override this method to specify how they should be initialized from
        their description.

        Args:
            description (dict):
                description of this object

        Returns:
            A new instance of this class according to the description.
        """
        assert cls.__name__ == description['@type'], \
            "Description for '{}' has wrong type '{}'".format(
                cls.__name__, description['@type'])
        instance = cls.__new__(cls)

        for member, init_val in cls.__get_all_undescribed__().items():
            instance.__dict__[member] = deepcopy(init_val)

        for member, default_val in cls.__get_all_default_values__().items():
            instance.__dict__[member] = deepcopy(default_val)

        for member, descr in description.items():
            if member == '@type':
                continue
            instance.__dict__[member] = create_from_description(descr)

        cls.__init_from_description__(instance, description)
        return instance

    def __init_from_description__(self, description):
        """
        Subclasses can override this to provide additional initialization when
        created from a description.

        This method will be called AFTER the object has already been created
        from the description.

        Args:
            description (dict):
                the description of this object
        """
        pass

    @classmethod
    def __get_all_undescribed__(cls):
        ignore = {}
        for c_ignore in _traverse_ancestor_attrs(cls, '__undescribed__'):
            if isinstance(c_ignore, dict):
                ignore.update(c_ignore)
            elif isinstance(c_ignore, set):
                ignore.update({k: None for k in c_ignore})
        return ignore

    @classmethod
    def __get_all_default_values__(cls):
        default = {}
        for c_default in _traverse_ancestor_attrs(cls, '__default_values__'):
            if isinstance(c_default, dict):
                default.update(c_default)
        return default


def get_description(this):
    """
    Create a JSON-serializable description of this object.

    This description can be used to create a new instance of this object by
    calling :func:`create_from_description`.

    Args:
        this (Describable):
            An object to be described. Must either be a basic datatype
            or inherit from Describable.
    Returns:
        A JSON-serializable description of this object.
    """
    if isinstance(this, Describable):
        return this.__describe__()

    elif isinstance(this, (list, tuple)):
        result = []
        try:
            for i, v in enumerate(this):
                result.append(get_description(v))
        except TypeError as err:
            err.args = (err.args[0] + "[{}]".format(i),)
            raise
        return result
    elif isinstance(this, np.ndarray):
        return this.tolist()
    elif isinstance(this, dict):
        result = {}
        try:
            for k in this:
                result[k] = get_description(this[k])
        except TypeError as err:
            err.args = (err.args[0] + "[{}]".format(k),)
            raise
        return result
    elif (isinstance(this, (bool, float, type(None))) or
          isinstance(this, six.integer_types) or
          isinstance(this, six.string_types)):
        return this
    else:
        raise TypeError('Type: "{}" is not describable'.format(type(this)))


def create_from_description(description):
    """
    Instantiate a new object from a description.

    Args:
        description (dict):
            A description of the object.

    Returns:
        A new instance of the described object
    """
    if isinstance(description, dict):
        if '@type' in description:
            name = description['@type']
            for describable in get_inheritors(Describable):
                if describable.__name__ == name:
                    return describable.__new_from_description__(description)
            raise TypeError('No describable class "{}" found!'.format(name))
        else:
            return {k: create_from_description(v)
                    for k, v in description.items()}
    elif (isinstance(description, (bool, float, type(None))) or
          isinstance(description, six.integer_types) or
          isinstance(description, six.string_types)):
        return description
    elif isinstance(description, list):
        return [create_from_description(d) for d in description]

    raise TypeError('Invalid description of type {}'.format(type(description)))


def _traverse_ancestor_attrs(cls, attr_name):
    for c in reversed(cls.__mro__):
        if hasattr(c, attr_name):
            yield getattr(c, attr_name)
