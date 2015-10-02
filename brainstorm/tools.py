#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import h5py
from brainstorm import layers
from brainstorm.training.trainer import run_network
from brainstorm.utils import get_by_path
from brainstorm.scorers import (
    gather_losses_and_scores, aggregate_losses_and_scores)

__all__ = ['get_in_out_layers_for_classification', 'draw_network',
           'print_network_info', 'evaluate', 'save_features']


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

    fc_layer = layers.FullyConnected(nr_classes, name='fc_' + outlayer_name,
                                     activation='linear')
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


def evaluate(net, iter, scorers=(), out_name='', targets_name='targets',
             mask_name=None):
    iterator = iter(handler=net.handler)
    scores = {scorer.__name__: [] for scorer in scorers}
    for n in net.get_loss_values():
        scores[n] = []

    for _ in run_network(net, iterator):
        net.forward_pass()
        gather_losses_and_scores(
            net, scorers, scores, out_name=out_name,
            targets_name=targets_name, mask_name=mask_name)

    return aggregate_losses_and_scores(scores, net, scorers)


def save_features(net, iter, file_name, feat_name, verbose=True):
    iterator = iter(verbose=verbose, handler=net.handler)

    first_pass = True
    num_items = 0
    with h5py.File(file_name, 'w') as f:
        for _ in run_network(net, iterator):
            net.forward_pass()
            data = net.handler.get_numpy_copy(get_by_path(net.buffer,
                                                          feat_name))
            num_items += data.shape[1]
            if first_pass:
                ds = f.create_dataset(
                    feat_name, data.shape, data.dtype, chunks=data.shape,
                    maxshape=(data.shape[0], None) + data.shape[2:])
                ds[:] = data
                first_pass = False
            else:
                ds.resize(size=num_items, axis=1)
                ds[:, num_items - data.shape[1]:num_items, ...] = data

