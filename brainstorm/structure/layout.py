#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import itertools
from collections import OrderedDict

import numpy as np

from brainstorm.structure.buffer_structure import BufferStructure
from brainstorm.utils import (NetworkValidationError, get_by_path,
                              convert_to_nested_indices, flatten,
                              get_normalized_path, sort_by_index_key)


class Hub(object):
    @staticmethod
    def create(source_set, sink_set, layout, connections):
        def ensure_uniform(l):
            assert min(l) == max(l)
            return l[0]

        sorted_sources = sorted(source_set)
        flat_sources = list(flatten(sorted_sources))
        nesting = convert_to_nested_indices(sorted_sources)

        # get buffer type for hub and assert its uniform
        structs = [BufferStructure.from_layout(get_by_path(layout, s))
                   for s in flat_sources]
        btype = ensure_uniform([s.buffer_type for s in structs])
        # max context size
        context_size = max([s.context_size for s in structs])

        hub = Hub(flat_sources, nesting, sorted(sink_set), btype, context_size)
        hub.setup(connections)
        hub.sizes = [structs[i].feature_size for i in hub.perm]
        hub.size = sum(hub.sizes)
        hub.is_backward_only = ensure_uniform([structs[i].is_backward_only
                                               for i in hub.perm])
        return hub

    def __init__(self, flat_sources, nesting, sinks, btype, context_size=0):
        self.flat_sources = flat_sources
        self.nesting = nesting
        self.sinks = sinks
        self.btype = btype
        self.context_size = context_size
        self.connection_table = []
        self.sizes = []
        self.size = -1
        self.perm = None

    def get_shape(self, time_size=1, batch_size=1):
        full_shape = (time_size + self.context_size,
                      batch_size,
                      self.size)
        return full_shape[2 - self.btype:]

    def setup(self, connections):
        self.set_up_connection_table(connections)
        self.permute_rows()

    def set_up_connection_table(self, connections):
        """
        Construct a source/sink connection table from a list of connections.
        Args:
            connections (list[tuple]):
                list of connections
        Returns:
            np.ndarray:
                connection table
        """
        # set up connection table
        self.connection_table = np.zeros((len(self.flat_sources),
                                          len(self.sinks)))
        for start, stop in connections:
            if start in self.flat_sources and stop in self.sinks:
                start_idx = self.flat_sources.index(start)
                stop_idx = self.sinks.index(stop)
                self.connection_table[start_idx, stop_idx] = 1

    def permute_rows(self):
        """
        Given a list of sources and a connection table, find a permutation of
        the sources, such that they can be connected to the sinks via a single
        buffer.
        """
        # systematically try all permutations until one satisfies the condition
        for perm in itertools.permutations(self.nesting):
            self.perm = list(flatten(perm))
            ct = np.atleast_2d(self.connection_table[self.perm])
            if Hub.can_be_connected_with_single_buffer(ct):
                self.connection_table = ct
                self.flat_sources = [self.flat_sources[i] for i in self.perm]
                return

        raise NetworkValidationError("Failed to lay out buffers. "
                                     "Please change connectivity.")

    @staticmethod
    def can_be_connected_with_single_buffer(connection_table):
        """
        Check for a connection table if it represents a layout that can be
        realized by a single buffer.

        This means checking if in every column of the table all the ones form a
        connected block.

        Args:
            connection_table (array_like):
                2d array of zeros and ones representing the connectivity
                between inputs and outputs of a hub.

        Returns:
            bool
        """
        padded = np.zeros((connection_table.shape[0] + 2,
                           connection_table.shape[1]))
        padded[1:-1, :] = connection_table
        return np.all(np.abs(np.diff(padded, axis=0)).sum(axis=0) <= 2)

    def get_indices(self):
        idxs = np.cumsum([0] + self.sizes)
        for source_name, start, stop in zip(self.flat_sources, idxs, idxs[1:]):
            yield source_name, (int(start), int(stop))

        for i, sink_name in enumerate(self.sinks):
            start = idxs[np.argmax(self.connection_table[:, i])]
            stop = idxs[self.connection_table.shape[0] -
                        np.argmax(self.connection_table[::-1, i])]
            yield sink_name, (int(start), int(stop))


def create_layout(layers):
    # gather connections and order-constraints
    forced_orders = get_forced_orders(layers)
    connections = get_connections(layers)

    # create a stub layout
    layout = create_layout_stub(layers)
    all_sources = get_all_sources(forced_orders, connections, layout)

    # group into hubs and lay them out
    hubs = group_into_hubs(all_sources, forced_orders, connections, layout)
    hubs = sorted(hubs, key=lambda x: (x.is_backward_only, x.btype))
    layout_hubs(hubs, layout)

    # add shape to parameters
    if '@slice' not in layout['parameters']:
        layout['parameters']['@slice'] = (0, 0)
        layout['parameters']['@hub'] = 0
    if '@slice' not in layout['gradients']:
        layout['gradients']['@slice'] = (0, 0)
        layout['gradients']['@hub'] = 0
    param_slice = layout['parameters']['@slice']
    layout['parameters']['@shape'] = (param_slice[1] - param_slice[0],)
    layout['gradients']['@shape'] = (param_slice[1] - param_slice[0],)

    return hubs, layout


def layout_hubs(hubs, layout):
    """
    Determine and fill in the @slice entries into the layout and return total
    buffer sizes.
    """
    for hub_nr, hub in enumerate(hubs):
        for buffer_name, _slice in hub.get_indices():
            buffer_layout = get_by_path(layout, buffer_name)
            buffer_layout['@slice'] = _slice
            buffer_layout['@hub'] = hub_nr


def get_all_sources(forced_orders, connections, layout):
    """Gather all sources while preserving order of the sources."""
    all_sinks = sorted(set(list(zip(*connections))[1])) if connections else []
    all_sources = list()
    for s in gather_array_nodes(layout):
        if s in all_sinks + ['parameters', 'gradients']:
            continue
        for fo in forced_orders:
            if s in set(flatten(all_sources)):
                break
            elif s in fo:
                all_sources.append(fo)
                break
        else:
            all_sources.append(s)

    return all_sources


def get_forced_orders(layers):
    forced_orders = [get_parameter_order(n, l) for n, l in layers.items()]
    forced_orders += [get_gradient_order(n, l) for n, l in layers.items()]
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
    root = {'@type': 'BufferView',
            'parameters': {
                '@type': 'array',
                '@index': 0},
            'gradients': {
                '@type': 'array',
                '@index': 1,
                '@is_backward_only': True}
            }
    for i, (layer_name, layer) in enumerate(layers.items(), start=2):
        root[layer_name] = get_layout_stub_for_layer(layer)
        root[layer_name]['@type'] = 'BufferView'
        root[layer_name]['@index'] = i
    return root


def get_layout_stub_for_layer(layer):
    layout = {}

    layout['inputs'] = {
        k: layer.in_shapes[k].to_json(i)
        for i, k in enumerate(sorted(layer.in_shapes))
    }
    layout['inputs']['@type'] = 'BufferView'
    layout['inputs']['@index'] = 0

    layout['outputs'] = {
        k: v.to_json(i) for i, (k, v) in enumerate(layer.out_shapes.items())
    }
    layout['outputs']['@type'] = 'BufferView'
    layout['outputs']['@index'] = 1

    parameters = layer.parameter_shapes
    assert isinstance(parameters, OrderedDict)
    layout['parameters'] = {
        k: v.to_json(i) for i, (k, v) in enumerate(parameters.items())
    }
    layout['parameters']['@type'] = 'BufferView'
    layout['parameters']['@index'] = 2

    internals = layer.internal_shapes
    assert isinstance(parameters, OrderedDict)

    layout['internals'] = {
        k: v.to_json(i) for i, (k, v) in enumerate(internals.items())
    }
    layout['internals']['@type'] = 'BufferView'
    layout['internals']['@index'] = 3

    layout['input_deltas'] = {
        k: layer.in_shapes[k].to_json(i)
        for i, k in enumerate(sorted(layer.in_shapes))
    }
    for k, v in layout['input_deltas'].items():
        v['@is_backward_only'] = True
    layout['input_deltas']['@type'] = 'BufferView'
    layout['input_deltas']['@index'] = 4

    layout['output_deltas'] = {
        k: v.to_json(i) for i, (k, v) in enumerate(layer.out_shapes.items())
    }
    for k, v in layout['output_deltas'].items():
        v['@is_backward_only'] = True
    layout['output_deltas']['@type'] = 'BufferView'
    layout['output_deltas']['@index'] = 5

    layout['gradients'] = {
        k: v.to_json(i) for i, (k, v) in enumerate(parameters.items())
    }
    for k, v in layout['gradients'].items():
        v['@is_backward_only'] = True
    layout['gradients']['@type'] = 'BufferView'
    layout['gradients']['@index'] = 6

    return layout


def gather_array_nodes(layout):
    for k, v in sorted(layout.items(), key=sort_by_index_key):
        if k.startswith('@'):
            continue
        if isinstance(v, dict) and v['@type'] == 'BufferView':
            for sub_path in gather_array_nodes(v):
                yield k + '.' + sub_path
        elif isinstance(v, dict) and v['@type'] == 'array':
            yield k


def get_backward_connection(start, stop, layer):
    start_layer, start_category, start_buffer = start.split('.', 2)
    stop_layer, stop_category, stop_buffer = stop.split('.', 2)
    back_buffer_name = {'parameters': 'gradients',
                        'inputs': 'input_deltas',
                        'outputs': 'output_deltas'}
    new_end = '.'.join([stop_layer, back_buffer_name[stop_category],
                        stop_buffer])

    if start_category == 'internals':
        dstart_buffer = 'd' + start_buffer
        if dstart_buffer not in layer.internal_shapes:
            raise KeyError('Missing delta buffer {} for the internal buffer {}'
                           '.'.format(dstart_buffer, start_buffer))
        new_start = '.'.join([start_layer, 'internals', dstart_buffer])
    else:
        new_start = '.'.join([start_layer, back_buffer_name[start_category],
                              start_buffer])

    return new_start, new_end


def get_connections(layers):
    connections = []
    for layer_name, layer in layers.items():
        for con in layer.outgoing:
            start = get_normalized_path(con.start_layer, 'outputs',
                                        con.output_name)
            end = get_normalized_path(con.end_layer, 'inputs', con.input_name)
            connections.append((start, end))

            bwd_con = get_backward_connection(start, end, layer)

            if bwd_con:
                connections.append(bwd_con)

    # add connections to implicit 'parameters', and 'gradients'-layer
    for layer_name, layer in layers.items():
        for param_name in layer.parameter_shapes:
            start = get_normalized_path(layer_name, 'parameters', param_name)
            end = 'parameters'
            connections.append((start, end))

            start = get_normalized_path(layer_name, 'gradients', param_name)
            end = 'gradients'
            connections.append((start, end))

    return sorted(connections)


def get_order(structure):
    return tuple(sorted(structure, key=lambda x: structure[x]['@index']))


def get_parameter_order(layer_name, layer):
    return tuple([get_normalized_path(layer_name, 'parameters', o)
                  for o in layer.parameter_shapes])


def get_gradient_order(layer_name, layer):
    return tuple([get_normalized_path(layer_name, 'gradients', o)
                  for o in layer.parameter_shapes])


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


def group_into_hubs(remaining_sources, forced_orders, connections, layout):
    m_cons = merge_connections(connections, forced_orders)
    hubs = []
    while remaining_sources:
        node = remaining_sources[0]
        source_set, sink_set = get_forward_closure(node, m_cons)
        for s in source_set:
            remaining_sources.remove(s)

        hubs.append(Hub.create(source_set, sink_set, layout, connections))

    return hubs


def get_forward_closure(node, connections):
    """
    For a given node return two sets of nodes such that:
      - the given node is in the source_set
      - the sink_set contains all the connection targets for nodes of the
        source_set
      - the source_set contains all the connection starts for nodes from the
        sink_set

    Args:
        node (str):
            The node to start the forward closure from.
        connections (list[(str, str)]):
            list of connections (start_node, end_node)

    Returns:
        (set, set):
            A tuple (source_set, sink_set) where source_set is set of
            nodes containing the initial node and all nodes connecting to nodes
            in the sink_set. And sink_set is a set of nodes containing all
            nodes receiving connections from any of the nodes from the
            source_set.
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
        if len(new_source_set) > len(source_set) or \
                len(new_sink_set) > len(sink_set):
            growing = True
            source_set = new_source_set
            sink_set = new_sink_set
    return source_set, sink_set
