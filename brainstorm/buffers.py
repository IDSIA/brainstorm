#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from collections import namedtuple, OrderedDict
import itertools
import numpy as np
from brainstorm.utils import InvalidArchitectureError

ParameterLayout = namedtuple('ParameterLayout', ['size', 'layout'])

InOutLayout = namedtuple('InOutLayout',
                         ['size', 'source_layout', 'sink_layout'])


def create_param_layout(layers):
    """
    Determine the total size and the layout for the parameter buffer.
    The layout is a dictionary mapping the layer names to slice objects.
    """
    bounds = np.cumsum([0] + [l.get_parameter_size() for l in layers.values()])
    total_size = bounds[-1]
    layout = OrderedDict([(name, slice(bounds[i], bounds[i+1]))
                          for i, name in enumerate(layers)])
    return ParameterLayout(total_size, layout)


def create_in_out_layout(layers):
    remaining_sources = list(layers.keys())
    buffer_hubs = []
    while remaining_sources:
        layer = remaining_sources[0]
        source_set, sink_set = get_forward_closure(layer, layers)
        for s in source_set:
            remaining_sources.remove(s)
        buffer_hubs.append(lay_out_buffer_hub(source_set, sink_set, layers))

    return buffer_hubs


def lay_out_buffer_hub(source_set, sink_set, layers):
    # Set up connection table
    source_list, sink_list, connection_table = set_up_connection_table(
        source_set, sink_set, layers)
    perm = permute_rows(connection_table)
    source_list = [source_list[i] for i in perm]
    connection_table = np.atleast_2d(connection_table[perm])

    # Source Layout
    source_bounds = np.cumsum([0] + [layers[n].out_size for n in source_list])
    total_size = source_bounds[-1]
    source_layout = OrderedDict([
        (name, slice(source_bounds[i], source_bounds[i+1]))
        for i, name in enumerate(source_list)])

    # Sink Layout
    sink_layout = OrderedDict()
    for i, n in enumerate(sink_list):
        connectivity = connection_table[:, i]

        start_idx = -1
        for j, c in enumerate(connectivity):
            if start_idx == -1 and c == 1:
                start_idx = source_bounds[j]
            if start_idx > -1 and c == 0:
                stop_idx = source_bounds[j]
                break
        else:
            stop_idx = source_bounds[-1]

        sink_layout[n] = slice(start_idx, stop_idx)
        # assert stop_idx - start_idx == layers[n].in_size

    return InOutLayout(total_size, source_layout, sink_layout)


def get_forward_closure(layer_name, layers):
    """
    For a given layer return two sets of layer names such that:
      - the given layer is in the source_set
      - the sink_set contains all the target layers of the source_set
      - the source_set contains all the source layers of the sink_set

    :param layer_name: The name of the layer to start the forward closure from.
    :type layer_name: unicode
    :param layers: dictionary of instantiated layers. They should have
        sink_layers and source_layers fields.
    :type layers: dict
    :return: A tuple (source_set, sink_set) where source_set is set of
        layer names containing the initial layer and all sources of all layers
        in the sink_set. And sink_set is a set of layer names containing all
        the targets for all the layers from the source_set.
    :rtype: (set, set)
    """
    source_set = {layer_name}
    sink_set = set(layers[layer_name].sink_layers)
    growing = True
    while growing:
        growing = False
        new_source_set = {s for l in sink_set
                          for s in layers[l].source_layers}
        new_sink_set = {t for l in source_set
                        for t in layers[l].sink_layers}
        if len(new_source_set) > len(source_set) or\
                len(new_sink_set) > len(sink_set):
            growing = True
            source_set = new_source_set
            sink_set = new_sink_set
    return source_set, sink_set


def set_up_connection_table(sources, sinks, layers):
    """
    Given a forward closure and the architecture constructs the
    connection table.

    :type sources: set[unicode]
    :type sinks: set[unicode]
    :type layers: dict
    :rtype: (list, list, np.ndarray)
    """
    # turn into sorted lists
    source_list = sorted([l for l in sources])
    sink_list = sorted([l for l in sinks])
    # set up connection table
    connection_table = np.zeros((len(source_list), len(sink_list)))
    for i, source in enumerate(source_list):
        for sink in layers[source].sink_layers:
            connection_table[i, sink_list.index(sink)] = 1

    return source_list, sink_list, connection_table


def permute_rows(connection_table):
    """
    Given a list of sources and a connection table, find a permutation of the
    sources, such that they can be connected to the sinks via a single buffer.
    @type connection_table: np.ndarray
    @rtype: list[int]
    """
    # systematically try all permutations until one satisfies the condition
    final_permutation = None
    for perm in itertools.permutations(range(connection_table.shape[0])):
        perm = list(perm)
        ct = np.atleast_2d(connection_table[perm])
        if can_be_connected_with_single_buffer(ct):
            final_permutation = perm
            break
    if final_permutation is None:
        raise InvalidArchitectureError("Failed to lay out buffers. "
                                       "Please change connectivity.")

    return final_permutation


def can_be_connected_with_single_buffer(connection_table):
    """
    Check for a connection table if it represents a layout that can be realized
    by a single buffer. This is equivalent to checking if in every column of
    the table all the ones form a connected block.
    @type connection_table: np.ndarray
    @rtype: bool
    """
    for i in range(connection_table.shape[1]):
        region_started = False
        region_stopped = False
        for j in range(connection_table.shape[0]):
            if not region_started and connection_table[j, i]:
                region_started = True
            elif region_started and not region_stopped and \
                    not connection_table[j, i]:
                region_stopped = True
            elif region_stopped and connection_table[j, i]:
                return False
    return True
