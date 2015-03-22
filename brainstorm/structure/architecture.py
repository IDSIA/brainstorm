#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict, namedtuple
from copy import copy
from six import string_types
from brainstorm.structure.construction import ConstructionWrapper

from brainstorm.utils import (InvalidArchitectureError,
                              is_valid_layer_name)
from brainstorm.layers.base_layer import get_layer_class_from_typename


ConnectionBase = namedtuple('ConnectionBase',
                            ['start_layer', 'output_name',
                             'end_layer', 'input_name'])


class Connection(ConnectionBase):
    pass


def get_layer_description(layer_details):

    outgoing = {}
    for outp, inp, end_layer in layer_details.outgoing:
        if outp in outgoing:
            outgoing[outp].add("{}.{}".format(end_layer.name, inp))
        else:
            outgoing[outp] = {"{}.{}".format(end_layer.name, inp)}

    description = {
        '@type': layer_details.layer_type,
        '@outgoing_connections': outgoing
    }
    if layer_details.layer_kwargs:
        description.update(layer_details.layer_kwargs)
    return description


def generate_architecture(some_layer):
    if isinstance(some_layer, ConstructionWrapper):
        some_layer = some_layer.layer
    layers = some_layer.collect_connected_layers()
    arch = {layer.name: get_layer_description(layer) for layer in layers}
    return arch


def parse_connection(connection_string):
    end_layer, _, input_name = connection_string.partition('.')
    return end_layer, input_name


def collect_all_outgoing_connections(layer, layer_name):
    outgoing = []
    if isinstance(layer['@outgoing_connections'], (list, set, tuple)):
        for con_str in layer['@outgoing_connections']:
            end_layer, input_name = parse_connection(con_str)
            if not input_name:
                input_name = 'default'
            outgoing.append(Connection(layer_name, 'default',
                                       end_layer, input_name))
    else:  # dict
        for out_name, out_con in layer['@outgoing_connections'].items():
            for con_str in out_con:
                end_layer, input_name = parse_connection(con_str)
                if not input_name:
                    input_name = 'default'
                outgoing.append(Connection(layer_name, out_name,
                                           end_layer, input_name))
    return outgoing


def collect_all_connections(architecture):
    all_connections = []
    for layer_name, layer in architecture.items():
        all_connections.extend(collect_all_outgoing_connections(layer,
                                                                layer_name))
    return sorted(all_connections)


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
    connections = collect_all_connections(architecture)
    end_layers = {c.end_layer for c in connections}
    undefined_end_layers = end_layers.difference(architecture)
    if undefined_end_layers:
        raise InvalidArchitectureError(
            'Could not find end layer(s) "{}"'.format(undefined_end_layers))

    # has exactly one InputLayer and its called InputLayer
    if "InputLayer" not in architecture or \
            architecture['InputLayer']['@type'] != 'InputLayer':
        raise InvalidArchitectureError(
            'Needs exactly one InputLayer that is called "InputLayer"')

    # no connections to InputLayer
    if 'InputLayer' in end_layers:
        raise InvalidArchitectureError(
            'DataLayers can not have incoming connections!')

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
    connections = collect_all_connections(architecture)

    while True:
        remaining_layers = [l for l in architecture.keys()
                            if l not in already_ordered_layers]
        new_layers = []
        for layer_name in remaining_layers:
            outgoing_layers = {c.end_layer for c in connections
                               if c.start_layer == layer_name}
            if outgoing_layers <= already_ordered_layers:
                new_layers.append(layer_name)

        if not new_layers:
            break
        layer_order += sorted(new_layers, reverse=True)
        already_ordered_layers = set(layer_order)

    remaining_layers = set(architecture.keys()) - already_ordered_layers

    assert not remaining_layers, "couldn't reach %s" % remaining_layers
    return list(reversed(layer_order))


def get_kwargs(layer):
    kwarg_ignore = {'@type', '@outgoing_connections'}
    return {k: copy(v) for k, v in layer.items() if k not in kwarg_ignore}


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
    connections = collect_all_connections(architecture)
    for layer_name in get_canonical_layer_order(architecture):
        layer = architecture[layer_name]
        LayerClass = get_layer_class_from_typename(layer['@type'])
        incoming = {c for c in connections if c.end_layer == layer_name}
        outgoing = {c for c in connections if c.start_layer == layer_name}

        input_names = {c.input_name for c in incoming}
        in_shapes = {}
        for input_name in input_names:
            in_shapes[input_name] = combine_input_shapes(
                [layers[c.start_layer].out_shapes[c.output_name]
                 for c in incoming if c.input_name == input_name])

        layers[layer_name] = LayerClass(in_shapes, incoming, outgoing,
                                        **get_kwargs(layer))
    return layers

