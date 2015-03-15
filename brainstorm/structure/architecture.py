#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from copy import copy
from six import string_types

from brainstorm.utils import (InvalidArchitectureError,
                              is_valid_layer_name)
from brainstorm.layers.python_layers import get_layer_class_from_typename


def get_layer_description(layer):
    description = {
        '@type': layer.layer_type,
        'shape': layer.shape,
        '@outgoing_connections': {l.name for l in layer.outgoing}
    }
    if layer.layer_kwargs:
        description.update(layer.layer_kwargs)
    return description


def get_injector_description(injector):
    description = {
        '@type': injector.injector_type,
        'layer': injector.layer.name
    }
    if injector.target_from is not None:
        description['target_from'] = injector.target_from
    if injector.injector_kwargs:
        description.update(injector.injector_kwargs)
    return description


def generate_architecture(some_layer):
    layers = some_layer.collect_connected_layers()
    arch = {layer.name: get_layer_description(layer) for layer in layers}
    return arch


def generate_injectors(some_layer):
    layers = some_layer.collect_connected_layers()
    injects = {inj.name: get_injector_description(inj) for layer in layers
               for inj in layer.injectors}
    return injects


def collect_all_outgoing_connections(layer):
    if isinstance(layer['@outgoing_connections'], (list, set, tuple)):
        outgoing = [sink_name.partition('.')[0]
                    for sink_name in layer['@outgoing_connections']]
    else:  # dict
        outgoing = []
        for source_name, out_con in layer['@outgoing_connections'].items():
            outgoing.extend([sink_name.partition('.')[0]
                             for sink_name in out_con])
    return set(outgoing)


def validate_architecture(architecture):
    # schema
    for name, layer in architecture.items():
        if not isinstance(name, string_types):
            raise InvalidArchitectureError('Non-string name {}'.format(name))
        if '@type' not in layer:
            raise InvalidArchitectureError(
                'Missing @type for "{}"'.format(name))
        if not isinstance(layer['@type'], string_types):
            raise InvalidArchitectureError('Invalid @type for "{}": {}'.format(
                name, type(layer['@type'])))

        if '@outgoing_connections' in layer and not isinstance(
                layer['@outgoing_connections'], (set, list, tuple, dict)):
            raise InvalidArchitectureError(
                'Invalid @outgoing_connections for "{}"'.format(name))

    # layer naming
    for name in architecture:
        if not is_valid_layer_name(name):
            raise InvalidArchitectureError(
                "Invalid layer name: '{}'".format(name))

    # all outgoing connections are present
    for layer in architecture.values():
        outgoing = collect_all_outgoing_connections(layer)
        outgoing.difference_update(architecture)
        if outgoing:
            raise InvalidArchitectureError('Could not find sink layer(s) "{}"'
                                           .format(outgoing))

    # has at least one DataLayer
    data_layers_by_type = {n for n, l in architecture.items()
                           if l['@type'] == 'DataLayer'}
    if len(data_layers_by_type) == 0:
        raise InvalidArchitectureError('No DataLayers found!')

    # no sources for DataLayers
    for name, layer in architecture.items():
        dcon = data_layers_by_type.intersection(layer['@outgoing_connections'])
        if len(dcon) > 0:
            raise InvalidArchitectureError(
                'DataLayers can not have incoming connections! '
                'But {} connects to {}'.format(name, dcon.pop()))

    # TODO: check if connected
    # TODO: check for cycles
    return True


def get_canonical_layer_order(architecture):
    """
    Takes a dictionary representation of an architecture and sorts it
    by (topological depth, name) and returns this canonical layer order as a
    list of names.
    """
    layer_order = []
    already_ordered_layers = set()
    while True:
        remaining_layers = [l for l in architecture.keys()
                            if l not in already_ordered_layers]
        new_layers = [
            n for n in remaining_layers
            if collect_all_outgoing_connections(architecture[n]) <= already_ordered_layers]

        if not new_layers:
            break
        layer_order += sorted(new_layers, reverse=True)
        already_ordered_layers = set(layer_order)

    remaining_layers = set(architecture.keys()) - already_ordered_layers

    assert not remaining_layers, "couldn't reach %s" % remaining_layers
    return list(reversed(layer_order))


def get_kwargs(layer):
    kwarg_ignore = {'@type', 'shape', '@outgoing_connections'}
    return {k: copy(v) for k, v in layer.items() if k not in kwarg_ignore}


def get_source_layers(layer_name, architecture):
    return [n for n, l in architecture.items()
            if layer_name in l['@outgoing_connections']]


def combine_input_shapes(shapes):
    """
    Concatenate the given sizes on the outermost feature dimension.
    Check that the other dimensions match.
    :param shapes: list of size-tuples or integers
    :type shapes: list[tuple[int]] or list[int]
    :return: tuple[int]
    """
    if not shapes:
        return 0,
    tupled_shapes = [ensure_tuple_or_none(s) for s in shapes]
    dimensions = [len(s) for s in tupled_shapes]
    if min(dimensions) != max(dimensions):
        raise ValueError('Dimensionality mismatch. {}'.format(tupled_shapes))
    fixed_feature_shape = tupled_shapes[0][1:]
    if not all([s[1:] == fixed_feature_shape for s in tupled_shapes]):
        raise ValueError('Feature size mismatch. {}'.format(tupled_shapes))
    summed_shape = sum(s[0] for s in tupled_shapes)
    return (summed_shape,) + fixed_feature_shape


def ensure_tuple_or_none(a):
    if a is None:
        return a
    elif isinstance(a, tuple):
        return a
    elif isinstance(a, list):
        return tuple(a)
    else:
        return a,


def instantiate_layers_from_architecture(architecture):
    validate_architecture(architecture)
    layers = OrderedDict()
    for layer_name in get_canonical_layer_order(architecture):
        layer = architecture[layer_name]
        LayerClass = get_layer_class_from_typename(layer['@type'])
        shape = ensure_tuple_or_none(layer.get('shape'))
        sources = get_source_layers(layer_name, architecture)
        in_shape = combine_input_shapes([layers[l_name].shape
                                         for l_name in sources])
        layers[layer_name] = LayerClass(shape, in_shape, layer['@outgoing_connections'],
                                        sources, get_kwargs(layer))
    return layers
