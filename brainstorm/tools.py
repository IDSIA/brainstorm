#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm import layers
from brainstorm.training.trainer import run_network

__all__ = ['get_in_out_layers_for_classification', 'draw_network',
           'print_network_info', 'evaluate']


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


def _flatten_all_but_last(a):
    if a is None:
        return None
    return a.reshape(-1, a.shape[-1])

import numpy as np


def _weighted_average(errors):
    errors = np.array(errors)
    assert errors.ndim == 2 and errors.shape[1] == 2
    return np.sum(errors[:, 1]) * errors[:, 0] / np.sum(errors[:, 0])


def evaluate(net, iter, scorers=(), out_name='', targets_name='targets',
             mask_name=None, verbose=True):
    iterator = iter(verbose=verbose, handler=net.handler)
    losses = OrderedDict()
    for n in net.get_loss_values():
        losses[n] = []

    log = {scorer.__name__: [] for scorer in scorers}
    for _ in run_network(net, iterator):
        net.forward_pass()
        ls = net.get_loss_values()
        for name, loss in ls:
            losses[name].append(net._buffer_manager.batch_size, loss)

        for sc in scorers:
            name = sc.__name__
            predicted = net.get_output(sc.out_name) if sc.out_name\
                else net.get_output(out_name)
            true_labels = net.get_input(sc.targets_name) if sc.targets_name\
                else net.get_input(targets_name)
            mask = net.get_input(sc.mask_name) if sc.mask_name\
                else (net.get_input(mask_name) if mask_name else None)

            predicted = _flatten_all_but_last(predicted)
            true_labels = _flatten_all_but_last(true_labels)
            mask = _flatten_all_but_last(mask)
            weight = mask.sum() if mask else predicted.shape[0]

            log[name].append(weight, sc(true_labels, predicted, mask))

    results = OrderedDict()
    if len(losses) == 1:
        results['loss'] = _weighted_average(list(losses.values())[0])
    else:
        results['losses'] = OrderedDict()
        for name, loss in losses.items():
            results['losses'][name] = _weighted_average(loss)

    for sc in scorers:
        results[sc.__name__] = sc.aggregate(log[sc.__name__])

    return results
