#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm import layers

__all__ = ['get_in_out_layers_for_classification', 'draw_network']


def get_in_out_layers_for_classification(in_shape, nr_classes,
                                         data_name='default',
                                         targets_name='targets',
                                         outlayer_name="Output",
                                         mask_name=None):
    if isinstance(in_shape, int):
        in_shape = (in_shape, )

    out_layer = layers.Classification(nr_classes, name=outlayer_name)

    if mask_name is None:
        inp_layer = layers.Input(out_shapes={data_name: ('T', 'B') + in_shape,
                                             targets_name: ('T', 'B', 1)})
        inp_layer - targets_name >> 'targets' - out_layer
        out_layer - 'loss' >> layers.Loss()
    else:
        inp_layer = layers.Input(out_shapes={data_name: ('T', 'B') + in_shape,
                                             targets_name: ('T', 'B', 1),
                                             mask_name: ('T', 'B', 1)})
        mask_layer = layers.Mask()
        inp_layer - targets_name >> 'targets' - out_layer
        out_layer - 'loss' >> mask_layer >> layers.Loss()
        inp_layer - mask_name >> 'mask' - mask_layer

    return inp_layer, out_layer


def draw_network(network, filename='network.png'):
    try:
        import pydot
        graph = pydot.Dot(graph_type='digraph')

        for k, v in network.architecture.items():
            for out_view, dest_list in v['@outgoing_connections'].items():
                for dest in dest_list:
                    edge = pydot.Edge(k, dest.split('.')[0])
                    graph.add_edge(edge)

        graph.write_png(filename)
        print('Network drawing saved as {}'.format(filename))

    except ImportError:
        print("pydot is required for drawing networks but was not found.")

