#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm import layers

__all__ = ['get_in_out_layers_for_classification', 'draw_network',
           'print_network_info']


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


def get_in_out_layers_for_regression(in_shape, nr_classes,
                                     data_name='default',
                                     targets_name='targets',
                                     outlayer_name="Output",
                                     mask_name=None):
    if isinstance(in_shape, int):
        in_shape = (in_shape, )

    fc_layer = layers.FullyConnected(nr_classes, name='fc_'+outlayer_name,
                                     activation_function='linear')
    out_layer = layers.SquaredDifference(name=outlayer_name)

    if mask_name is None:
        inp_layer = layers.Input(out_shapes={data_name: ('T', 'B') + in_shape,
                                             targets_name: ('T', 'B', 1)})
        inp_layer - targets_name >> 'inputs_2' - out_layer
        out_layer >> layers.Loss()
    else:
        inp_layer = layers.Input(out_shapes={data_name: ('T', 'B') + in_shape,
                                             targets_name: ('T', 'B', 1),
                                             mask_name: ('T', 'B', 1)})
        mask_layer = layers.Mask()
        inp_layer - targets_name >> 'inputs_2' - out_layer
        out_layer >> mask_layer >> layers.Loss()
        inp_layer - mask_name >> 'mask' - mask_layer

    fc_layer >> 'inputs_1' - out_layer

    return inp_layer, fc_layer, out_layer


def draw_network(network, filename='network.png'):

    try:
        import pygraphviz as pgv
        graph = pgv.AGraph(directed=True)
        for k, v in network.architecture.items():
                for out_view, dest_list in v['@outgoing_connections'].items():
                    for dest in dest_list:
                        graph.add_edge(k, dest.split('.')[0])

        graph.draw(filename, prog='dot')
        print('Network drawing saved as {}'.format(filename))
    except ImportError:
        print("pygraphviz is required for drawing networks but was not found.")


def print_network_info(network):
    print('=' * 30, "Network information", '=' * 30)
    print('total number of parameters: ', network.buffer.parameters.size)
    for layer in network.layers.values():
        print(layer.name)
        num_params = 0
        for view in network.buffer[layer.name].parameters.keys():
            view_size = network.buffer[layer.name].parameters[view].size
            view_shape = network.buffer[layer.name].parameters[view].shape
            print('\t', view, view_shape)
            num_params += view_size
        print('number of parameters:', num_params)
        print('input shapes:')
        for view in layer.in_shapes.keys():
            print('\t', view, layer.in_shapes[view].feature_shape, end='\t')
        print()
        print('output shapes:')
        for view in layer.out_shapes.keys():
            print('\t', view, layer.out_shapes[view].feature_shape, end='\t')
        print()
        print('-' * 80)
