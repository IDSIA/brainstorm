#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import itertools

import numpy as np
from brainstorm.utils import (NetworkValidationError, flatten,
                              convert_to_nested_indices, sort_by_index_key)


def create_layout(layers):
    forced_orders = get_forced_orders(layers)
    connections = get_connections(layers)
    m_cons = merge_connections(connections, forced_orders)

    # make a layout stub
    layout = create_layout_stub(layers)
    all_sinks = set(list(zip(*connections))[1])
    all_sources = list()
    for s in list(gather_array_nodes(layout)):
        if s in all_sinks:
            continue
        for fo in forced_orders:
            if s in all_sources:
                break
            elif s in fo:
                all_sources.extend(fo)
                break
        else:
            all_sources.append(s)
    # group them to hubs
    hubs = group_into_hubs(all_sources, m_cons, layout)

    # determine order for each hub
    ordered_sources_by_btype = get_source_order_by_btype(hubs, connections)

    all_sinks = sorted(list(zip(*connections))[1])
    buffer_sizes = [0, 0, 0]
    for btype, sources in enumerate(ordered_sources_by_btype):

        c_table = set_up_connection_table(sources, all_sinks, connections)
        sizes = [get_feature_size(get_by_path(s, layout)['@shape'])
                 for s in sources]
        indexes = np.cumsum([0] + sizes)

        for source_name, start, stop in zip(sources, indexes, indexes[1:]):
            source_layout = get_by_path(source_name, layout)
            source_layout['@slice'] = (start, stop)

        for i, sink_name in enumerate(all_sinks):
            if np.sum(c_table[:, i]) == 0:
                continue  # this sink is not connected
            start = indexes[np.argmax(c_table[:, i])]
            stop = indexes[c_table.shape[0] - np.argmax(c_table[::-1, i])]

            sink_layout = get_by_path(sink_name, layout)
            sink_layout['@slice'] = (start, stop)

        buffer_sizes[btype] = indexes[-1]
    return buffer_sizes, layout


def get_source_order_by_btype(hubs, connections):
    ordered_sources_by_btype = [[], [], []]
    for sources, sinks, btype in hubs:
        connection_table = set_up_connection_table(sources, sinks, connections)
        flat_sources = list(flatten(sources))
        source_index_structure = convert_to_nested_indices(sources)
        perm = permute_rows(connection_table, source_index_structure)
        final_sources = [flat_sources[i] for i in perm]
        ordered_sources_by_btype[btype].extend(final_sources)
    return ordered_sources_by_btype


def get_forced_orders(layers):
    forced_orders = [get_parameter_order(n, l) for n, l in layers.items()]
    forced_orders += [get_internal_order(n, l) for n, l in layers.items()]
    forced_orders = list(filter(None, forced_orders))
    # ensure no overlap
    for fo in forced_orders:
        for other in forced_orders:
            if fo is other:
                continue
            intersect = set(fo) & set(other)
            assert not intersect, "Forced orders may not overlap! but {} " \
                                  "appear(s) in multiple.".format(intersect)
    return forced_orders


def create_layout_stub(layers):
    root = {'@type': 'BufferView'}
    for i, (layer_name, layer) in enumerate(layers.items()):
        root[layer_name] = get_layout_stub_for_layer(layer)
        root[layer_name]['@type'] = 'BufferView'
        root[layer_name]['@index'] = i
    return root


def get_layout_stub_for_layer(layer):
    layout = {}

    layout['inputs'] = {
        k: {'@type': 'array',
            '@index': i,
            '@shape': layer.in_shapes[k],
            } for i, k in enumerate(layer.inputs)
    }
    layout['inputs']['@type'] = 'BufferView'
    layout['inputs']['@index'] = 0

    layout['outputs'] = {
        k: {'@type': 'array',
            '@index': i,
            '@shape': layer.out_shapes[k],
            } for i, k in enumerate(layer.out_shapes)
    }
    layout['outputs']['@type'] = 'BufferView'
    layout['outputs']['@index'] = 1

    layout['parameters'] = {
        k: add_array_type(v)
        for k, v in layer.get_parameter_structure().items()
    }
    layout['parameters']['@type'] = 'BufferView'
    layout['parameters']['@index'] = 2

    layout['internals'] = {
        k: add_array_type(v)
        for k, v in layer.get_internal_structure().items()
    }
    layout['internals']['@type'] = 'BufferView'
    layout['internals']['@index'] = 3

    return layout


def add_array_type(d):
    d['@type'] = 'array'
    return d


def create_path(layer_name, category, substructure):
    return "{}.{}.{}".format(layer_name, category, substructure)


def get_by_path(path, layout):
    current_node = layout
    for p in path.split('.'):
        try:
            current_node = current_node[p]
        except KeyError:
            raise KeyError('Path "{}" could not be resolved. Key "{}" missing.'
                           .format(path, p))
    return current_node


def gather_array_nodes(layout):
    for k, v in sorted(layout.items(), key=sort_by_index_key):
        if k.startswith('@'):
            continue
        if isinstance(v, dict) and v['@type'] == 'BufferView':
            for sub_path in gather_array_nodes(v):
                yield k + '.' + sub_path
        elif isinstance(v, dict) and v['@type'] == 'array':
            yield k


def get_connections(layers):
    connections = []
    for layer_name, layer in layers.items():
        for con in layer.outgoing:
            start = create_path(con.start_layer, 'outputs', con.output_name)
            end = create_path(con.end_layer, 'inputs', con.input_name)
            connections.append((start, end))
    return sorted(connections)


def get_order(structure):
    return tuple(sorted(structure, key=lambda x: structure[x]['@index']))


def get_parameter_order(layer_name, layer):
    return tuple([create_path(layer_name, 'parameters', o)
                  for o in get_order(layer.get_parameter_structure())])


def get_internal_order(layer_name, layer):
    return tuple([create_path(layer_name, 'internals', o)
                  for o in get_order(layer.get_internal_structure())])


def merge_connections(connections, forced_orders):
    """
    Replace connection nodes with forced order lists if they are part of it.
    """
    merged_connections = []
    for start, stop in connections:
        for fo in forced_orders:
            if start in fo:
                start = fo
            if stop in fo:
                stop = fo
        merged_connections.append((start, stop))
    return merged_connections


def get_buffer_type(shape):
    if shape[0] == 'T':
        assert shape[1] == 'B', "Invalid shape: {}".format(shape)
        return 2
    if shape[0] == 'B':
        return 1
    else:
        return 0


def get_feature_size(shape):
    """Get the feature size of a shape-template."""
    buffer_type = get_buffer_type(shape)
    return int(np.prod(shape[buffer_type:]))


def group_into_hubs(remaining_sources, connections, layout):
    buffer_hubs = []
    while remaining_sources:
        node = remaining_sources[0]
        source_set, sink_set = get_forward_closure(node, connections)
        for s in source_set:
            remaining_sources.remove(s)
        # get buffer type for hub and assert its uniform
        btypes = [get_buffer_type(get_by_path(s, layout)['@shape'])
                  for s in flatten(source_set)]
        assert min(btypes) == max(btypes)
        btype = btypes[0]
        # get hub size
        buffer_hubs.append((sorted(source_set), sorted(sink_set), btype))
    return buffer_hubs


def get_forward_closure(node, connections):
    """
    For a given node return two sets of nodes such that:
      - the given node is in the source_set
      - the sink_set contains all the connection targets for nodes of the
        source_set
      - the source_set contains all the connection starts for nodes from the
        sink_set

    :param node: The node to start the forward closure from.
    :param connections: list of nodes
    :type connections: list
    :return: A tuple (source_set, sink_set) where source_set is set of
        nodes containing the initial node and all nodes connecting to nodes
        in the sink_set. And sink_set is a set of nodes containing all
        nodes receiving connections from any of the nodes from the source_set.
    :rtype: (set, set)
    """
    source_set = {node}
    sink_set = {end for start, end in connections if start in source_set}
    growing = True
    while growing:
        growing = False
        new_source_set = {start for start, end in connections
                          if end in sink_set}
        new_sink_set = {end for start, end in connections
                        if start in source_set}
        if len(new_source_set) > len(source_set) or\
                len(new_sink_set) > len(sink_set):
            growing = True
            source_set = new_source_set
            sink_set = new_sink_set
    return source_set, sink_set


def set_up_connection_table(sources, sinks, connections):
    """
    Construct a source/sink connection table from a list of connections.

    :type sources: list[object]
    :type sinks: list[object]
    :type connections: list[tuple[object, object]]
    :rtype: np.ndarray
    """
    # set up connection table
    connection_table = np.zeros((len(sources), len(sinks)))
    for start, stop in connections:
        if start in sources and stop in sinks:
            start_idx = sources.index(start)
            stop_idx = sinks.index(stop)
            connection_table[start_idx, stop_idx] = 1

    return connection_table


def permute_rows(connection_table, nested_indices):
    """
    Given a list of sources and a connection table, find a permutation of the
    sources, such that they can be connected to the sinks via a single buffer.
    :type connection_table: np.ndarray
    :rtype: list[int]
    """
    # systematically try all permutations until one satisfies the condition
    for perm in itertools.permutations(nested_indices):
        perm = list(flatten(perm))
        ct = np.atleast_2d(connection_table[perm])
        if can_be_connected_with_single_buffer(ct):
            return perm

    raise NetworkValidationError("Failed to lay out buffers. "
                                   "Please change connectivity.")


def can_be_connected_with_single_buffer(connection_table):
    """
    Check for a connection table if it represents a layout that can be realized
    by a single buffer. This is equivalent to checking if in every column of
    the table all the ones form a connected block.
    :type connection_table: np.ndarray
    :rtype: bool
    """
    padded = np.pad(connection_table, [(1, 1), (0, 0)], 'constant')
    return np.all(np.abs(np.diff(padded, axis=0)).sum(axis=0) <= 2)

