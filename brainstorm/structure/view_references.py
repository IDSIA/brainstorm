#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import re
from brainstorm.utils import is_valid_layer_name


def get_regex_for_reference(reference):
    """
    Return a corresponding regex for refs like: 'FooLayer', 'I*_bias',
    or even '*layer*'.
    """
    if is_valid_layer_name(reference):
        return re.compile('^' + reference + '$')

    assert is_valid_layer_name(reference.replace('*', '_')), \
        "{} is not a valid layer reference.".format(reference)

    return re.compile('^' + reference.replace('*', '[_a-zA-Z0-9]*') + '$')


def get_key_to_references_mapping(keys, references):
    """
    Create a mapping that maps keys to their matching references.
    The 'default' reference matches only if no other reference does.
    The 'fallback' reference is ignored.

    :param keys: List of keys that the references are referring to.
    :type keys: iterable[str]
    :param references: List of references. Can be starred.
    :type references: iterable[str]
    :return: Dictionary mapping keys to sets of matching references.
    :rtype: dict[str, set[str]]
    """
    key_to_reference = {key: set() for key in keys}

    for ref in references:
        if ref in {'default', 'fallback'}:
            continue

        expr = get_regex_for_reference(ref)
        matching_keys = [key for key in key_to_reference if expr.match(key)]
        assert matching_keys, \
            "{} does not match any keys. Possible keys are: {}".format(
                ref, sorted(key_to_reference.keys()))

        for key in matching_keys:
            key_to_reference[key].add(ref)

    if 'default' in references:
        for key, refs in key_to_reference.items():
            if not refs:
                refs.add('default')

    return key_to_reference


def empty_dict_from(structure):
    if isinstance(structure, dict):
        return {k: empty_dict_from(v) for k, v in structure.items()}
    else:
        return set()


def append_to_all_leaves(structure, value):
    if isinstance(structure, dict):
        for k, v in structure.items():
            append_to_all_leaves(v, value)
    else:
        assert isinstance(structure, set)
        if isinstance(value, (list, set)):
            structure.update(value)
        else:
            structure.add(value)


def update_recursively(structure, values):
    if isinstance(structure, dict):
        assert isinstance(values, dict)
        for k in structure:
            update_recursively(structure[k], values[k])
    else:
        assert isinstance(structure, set) and isinstance(values, set)
        structure.update(values)


def resolve_references_recursively(structure, references):
    layer_to_references = get_key_to_references_mapping(structure, references)

    resolved = empty_dict_from(structure)
    for key, refs in layer_to_references.items():
        if isinstance(structure, dict):
            for ref in refs:
                if isinstance(references[ref], dict):
                    res = resolve_references_recursively(structure[key], references[ref])
                    update_recursively(resolved[key], res)
                else:
                    append_to_all_leaves(resolved[key], references[ref])
        else:
            resolved[key].update([references[ref] for ref in refs])




    return resolved