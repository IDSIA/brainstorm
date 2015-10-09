#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import re
from collections import namedtuple
from copy import deepcopy

from brainstorm.structure.buffer_views import BufferView
from brainstorm.utils import NetworkValidationError, is_valid_layer_name


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

    Args:
        keys (iterable[str]):
            List of keys that the references are referring to.

        references (iterable[str]):
            List of references. Can contain star wildcards.

    Returns:
        dict[str, set[str]]:
            Dictionary mapping keys to sets of matching references.
    """
    key_to_reference = {key: set() for key in keys}

    for ref in references:
        if ref in {'default', 'fallback'}:
            continue

        expr = get_regex_for_reference(ref)
        matching_keys = [key for key in key_to_reference if expr.match(key)]
        if not matching_keys:
            raise NetworkValidationError(
                "{} does not match any keys. Possible keys are: {}".format(
                    ref, sorted(key_to_reference.keys())))

        for key in matching_keys:
            key_to_reference[key].add(ref)

    if 'default' in references:
        for key, refs in key_to_reference.items():
            if not refs:
                refs.add('default')

    return key_to_reference


Node = namedtuple('Node', ['content', 'defaults', 'fallback'])


def empty_dict_from(structure):
    """
    Create a nested dict where all the leaves are Nodes from a given
    nested dict.

    Args:
        structure (dict):
            The nested dict structure to mimic
    Returns:
        dict:
            nested dictionary that mimics the structure
    """
    if isinstance(structure, (dict, BufferView)):
        return {k: empty_dict_from(v) for k, v in structure.items()}
    else:
        return Node(set(), set(), set())


def add_or_update(s, v):
    if isinstance(v, set):
        s.update(v)
    elif isinstance(v, (list, tuple)):
        s.add(tuple(v))
    else:
        s.add(v)


def append_to_all_leaves(structure, content, default, fallback):
    """
    Traverse the given structure and append or extend all the leaves (have to
    be Nodes) with the given value.
    """
    if isinstance(structure, dict):
        for k, v in structure.items():
            append_to_all_leaves(v, content, default, fallback)
    else:
        assert isinstance(structure, Node)
        add_or_update(structure.content, content)
        if default is not None:
            add_or_update(structure.defaults, default)
        if fallback is not None:
            add_or_update(structure.fallback, fallback)


def apply_references_recursively(resolved, references, parent_default,
                                 parent_fallback):
    if isinstance(resolved, dict) and isinstance(references, dict):
        current_default = references.get('default', parent_default)
        current_fallback = references.get('fallback', parent_fallback)
        layer_to_references = get_key_to_references_mapping(resolved,
                                                            references)
        for key, refs in layer_to_references.items():
            for ref in refs:
                apply_references_recursively(resolved[key], references[ref],
                                             current_default, current_fallback)
            if not refs:
                append_to_all_leaves(resolved[key], set(), current_default,
                                     current_fallback)
    else:
        append_to_all_leaves(resolved, references, parent_default,
                             parent_fallback)


def evaluate_defaults(structure):
    if isinstance(structure, dict):
        return {k: evaluate_defaults(v) for k, v in structure.items()}
    else:
        assert isinstance(structure, Node)
        if not structure.content:
            assert len(structure.defaults) <= 1
            return set(structure.defaults)
        else:
            return set(structure.content)


def get_fallbacks(structure):
    if isinstance(structure, dict):
        return {k: get_fallbacks(v) for k, v in structure.items()}
    else:
        assert isinstance(structure, Node)
        return structure.fallback


def resolve_references(structure, references):
    resolved = empty_dict_from(structure)
    apply_references_recursively(resolved, references, None, None)
    return evaluate_defaults(resolved), get_fallbacks(resolved)


def prune_view_references(view_refs):
    """Removes all empty entries and branches from view_reference tree"""
    pruned_refs = {}
    for layer_name, views in view_refs.items():
        layer_refs = {}
        if views is None:
            continue
        for view_name, setofthings in views.items():
            if setofthings:
                layer_refs[view_name] = setofthings

        if layer_refs:
            pruned_refs[layer_name] = layer_refs
    return pruned_refs


def copy_modifier_for(modifier, layer_name, view_name):
    copied_mod = deepcopy(modifier)
    copied_mod.layer_name = layer_name
    copied_mod.view_name = view_name
    return copied_mod


def order_and_copy_modifiers(view_refs):
    list_view_refs = {}
    for layer_name, views in view_refs.items():
        layer_refs = {}
        for view_name, setofmodifiers in views.items():
            l = []
            for mod in setofmodifiers:
                if isinstance(mod, (list, tuple)):
                    l.extend([copy_modifier_for(m, layer_name, view_name)
                              for m in mod])
                else:
                    l.append(copy_modifier_for(mod, layer_name, view_name))
            layer_refs[view_name] = l
        list_view_refs[layer_name] = layer_refs
    return list_view_refs
