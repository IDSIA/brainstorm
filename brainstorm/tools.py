#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import h5py
import six

from brainstorm import layers
from brainstorm.scorers import (aggregate_losses_and_scores,
                                gather_losses_and_scores)
from brainstorm.training.trainer import run_network
from brainstorm.utils import get_by_path, get_brainstorm_info

__all__ = ['get_in_out_layers_for_classification',
           'get_in_out_layers_for_regression', 'draw_network',
           'print_network_info', 'evaluate', 'extract_and_save']


def get_in_out_layers_for_classification(in_shape, nr_classes,
                                         data_name='default',
                                         targets_name='targets',
                                         fc_name=None,
                                         out_name="Output",
                                         mask_name=None):
    """Prepare input and output layers for building a multi-class classifier.

    This is a helper function for quickly building a typical multi-class
    classifier. It returns an ``Input`` layer and a ``FullyConnected`` layer
    which are already connected to other layers needed for this common case,

    The returned ``FullyConnected`` layer is already connected to the output
    layer (with the specified name) which is a ``SoftmaxCE`` layer.
    The targets are already connected to the SoftmaxCE layer as well.
    Finally, the ``loss`` output of the output layer is already connected to a
    ``Loss`` layer to make the network trainable.

    Example:
        >>> from brainstorm import tools, Network, layers
        >>> inp, out = tools.get_in_out_layers_for_classification(784, 10)
        >>> net = Network.from_layer(inp >> layers.FullyConnected(1000) >> out)
    Args:
        in_shape (int or tuple[int]): Shape of the input data.
        nr_classes (int): Number of possible classes.
        data_name (Optional[str]):
            Name of the input data which will be provided by a data iterator.
            Defaults to 'default'.
        targets_name (Optional[str]):
            Name of the ground-truth target data which will be provided by a
            data iterator. Defaults to 'targets'.
        fclayer_name (Optional[str]):
            Name for the fully connected layer which connects to the softmax
            layer. If unspecified, it is set to outlayer_name + '_FC'.
        outlayer_name (Optional[str]):
            Name for the output layer. Defaults to 'Output'.
        mask_name (Optional[str]):
            Name of the mask data which will be provided by a data iterator.
            Defaults to None.

            The mask is needed if error should be injected
            only at certain time steps (for sequential data).
    Returns:
        tuple: An ``Input`` and a ``FullyConnected`` layer.
    NOTE:
        This tool provides the output layer for `multi-class` classification,
        not `multi-label` classification.
    """
    if isinstance(in_shape, int):
        in_shape = (in_shape, )
    fc_name = out_name + '_FC' if fc_name is None else fc_name
    fc_layer = layers.FullyConnected(nr_classes, name=fc_name)
    out_layer = layers.SoftmaxCE(name=out_name)
    fc_layer >> out_layer

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

    return inp_layer, fc_layer


def get_in_out_layers_for_regression(in_shape, nr_outputs,
                                     data_name='default',
                                     targets_name='targets',
                                     outlayer_name="Output",
                                     mask_name=None):
    """Prepare input and output layers for building a multi-class classifier.

    This is a helper function for quickly building a typical network for a
    regression task with Mean Squared Loss. The output layer is a
    ``FullyConnected`` layer with linear activation.

    Example:
        >>> from brainstorm import tools, Network, layers
        >>> inp, out = tools.get_in_out_layers_for_regression(100, 10)
        >>> net = Network.from_layer(inp >> layers.FullyConnected(1000) >> out)
    Args:
        in_shape (int or tuple[int]): Shape of the input data.
        nr_classes (int): Number of outputs.
        data_name (Optional[str]):
            Name of the input data which will be provided by a data iterator.
            Defaults to 'default'.
        targets_name (Optional[str]):
            Name of the ground-truth target data which will be provided by a
            data iterator. Defaults to 'targets'.
        outlayer_name (Optional[str]):
            Name for the output layer.
        mask_name (Optional[str]):
            Name of the mask data which will be provided by a data iterator.
            Defaults to None.

            The mask is needed if error should be injected
            only at certain time steps (for sequential data).
    Returns:
        tuple: An input and an output layer.
    """
    if isinstance(in_shape, int):
        in_shape = (in_shape, )

    fc_layer = layers.FullyConnected(nr_outputs, name=outlayer_name,
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


def draw_network(network, file_name='network.png'):
    """Write a diagram for a network to a file.

    Args:
        network (brainstorm.structure.Network): Network to be drawn.
        file_name (Optional[str]): Defaults to 'network.png'.
    Note:
        This tool requires the pygraphviz library to be installed.
    Raises:
        ImportError: If pygraphviz can not be imported.
    """

    try:
        import pygraphviz as pgv
        graph = pgv.AGraph(directed=True)
        for k, v in network.architecture.items():
                for out_view, dest_list in v['@outgoing_connections'].items():
                    for dest in dest_list:
                        graph.add_edge(k, dest.split('.')[0])

        graph.draw(file_name, prog='dot')
        print('Network drawing saved as {}'.format(file_name))
    except ImportError:
        print("pygraphviz is required for drawing networks but was not found.")


def print_network_info(network):
    """Print detailed information about the network.

    This tools prints the input, output and parameter shapes for all the
    layers. It also prints the total number of parameters in each layer and
    in the full network.

    Args:
        network (brainstorm.structure.Network):
            A network for which the details are printed.
    """
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


def evaluate(network, iter, scorers=(), out_name='', targets_name='targets',
             mask_name=None):
    """Evaluate one or more scores for a network.

    This tool can be used to evaluate scores of a trained network on test
    data.

    Args:
        network (brainstorm.structure.Network): Network to be evaluated.
        iter (brainstorm.DataIterator): A data iterator which produces the
                                        data on which the scores are computed.
        scorers (tuple[brainstorm.scorers.Scorer]): A list or tuple of Scorers.
        out_name (Optional[str]): Name of the network output which is scored
                                  against the targets.
        targets_name (Optional[str]): Name of the targets data provided by the
                                      data iterator (``iter``).
        mask_name (Optional[str]): Name of the mask data  provided by the
                                   data iterator (``iter``).
    """
    iterator = iter(handler=network.handler)
    scores = {scorer.__name__: [] for scorer in scorers}
    for n in network.get_loss_values():
        scores[n] = []

    for _ in run_network(network, iterator):
        network.forward_pass()
        gather_losses_and_scores(
            network, scorers, scores, out_name=out_name,
            targets_name=targets_name, mask_name=mask_name)

    return aggregate_losses_and_scores(scores, network, scorers)


def extract_and_save(network, iter, buffer_names, file_name):
    """Save the desired buffer values of a network to an HDF5 file.

    In particular, this tool can be used to save the predictions of a
    network on a dataset.
    In general, any number of internal, input or output buffers of the network
    can be extracted.

    Examples:
        >>> getter = Minibatches(100, default=x_test)
        >>> extract_and_save(network,
        ...                  getter,
        ...                  ['Output.outputs.probabilities',
        ...                   'Hid1.internals.H'],
        ...                  'network_features.hdf5')
    Args:
        network (brainstorm.structure.Network): Network using which the features
                                            should be generated.
        iter (brainstorm.DataIterator): A data iterator which produces the
                                        data on which the features are
                                        computed.
        buffer_names (list[unicode]): Name of the buffer views to be saved (in
                                      dotted notation). See example.
        file_name (unicode): Name of the hdf5 file (including extension) in
                             which the features should be saved.
    """
    iterator = iter(handler=network.handler)
    if isinstance(buffer_names, six.string_types):
        buffer_names = [buffer_names]
    num_items = 0
    ds = []

    with h5py.File(file_name, 'w') as f:
        f.attrs.create('info', get_brainstorm_info())
        f.attrs.create('format', b'Buffers file v1.0')

        for _ in run_network(network, iterator, all_inputs=False):
            network.forward_pass()
            first_pass = False if len(ds) > 0 else True
            for num, buffer_name in enumerate(buffer_names):
                data = network.handler.get_numpy_copy(
                    get_by_path(network.buffer, buffer_name))
                if num == 0:
                    num_items += data.shape[1]
                if first_pass:
                    ds.append(f.create_dataset(
                        buffer_name, data.shape, data.dtype, chunks=data.shape,
                        maxshape=(data.shape[0], None) + data.shape[2:]))
                    ds[-1][:] = data
                else:
                    ds[num].resize(size=num_items, axis=1)
                    ds[num][:, num_items - data.shape[1]:num_items, ...] = data
