#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import json
from collections import OrderedDict

import h5py
import numpy as np

from brainstorm.describable import create_from_description, get_description
from brainstorm.handlers import default_handler
from brainstorm.initializers import ArrayInitializer, evaluate_initializer
from brainstorm.layers.loss_layer import LossLayerImpl
from brainstorm.randomness import Seedable
from brainstorm.structure.architecture import (
    generate_architecture, instantiate_layers_from_architecture)
from brainstorm.structure.buffer_views import BufferView
from brainstorm.structure.buffers import BufferManager
from brainstorm.structure.layout import create_layout
from brainstorm.structure.view_references import (order_and_copy_modifiers,
                                                  prune_view_references,
                                                  resolve_references)
from brainstorm.utils import NetworkValidationError, get_brainstorm_info
from brainstorm.value_modifiers import GradientModifier

__all__ = ['Network']


# ################################ Network ####################################

class Network(Seedable):
    __undescribed__ = {'layers', 'loss_layers', 'buffer', '_buffer_manager'}

    # -------------------------- Constructors ---------------------------------
    @classmethod
    def from_layer(cls, some_layer):
        """
        Create Network instance from a construction layer.

        Args:
            some_layer (brainstorm.construction.ConstructionWrapper):
                Some layer used to wire up an architecture with `>>`

        Returns:
            Network:
                A fully functional Network instance.
        """
        arch = generate_architecture(some_layer)
        return cls.from_architecture(arch)

    @classmethod
    def from_architecture(cls, architecture):
        """
        Create Network instance from given architecture.

        Args:
            architecture (dict):
                JSON serializable Architecture description.
        Returns:
            Network:
                A fully functional Network instance.
        """
        layers = instantiate_layers_from_architecture(architecture)
        hubs, layout = create_layout(layers)
        buffer_manager = BufferManager(layout, hubs)
        return cls(layers, buffer_manager, architecture)

    @classmethod
    def __new_from_description__(cls, description):
        net = Network.from_architecture(description['architecture'])
        net.set_handler(create_from_description(description['handler']))
        net.initialize(create_from_description(description['initializers']))
        net.set_gradient_modifiers(
            create_from_description(description['gradient_modifiers']))
        net.set_weight_modifiers(
            create_from_description(description['weight_modifiers']))
        net.output_name = description.get('output_name')
        return net

    @classmethod
    def from_hdf5(cls, filename):
        """
        Load network from HDF5 file.

        Args:
            filename (str):
                Name of the file that the network should be loaded from.

        Returns:
            Network:
                The loaded network.

        See Also:
            :meth:`.save_as_hdf5`
        """
        with h5py.File(filename, 'r') as f:
            description = json.loads(f['description'].value.decode())
            net = create_from_description(description)
            net.handler.set_from_numpy(net.buffer.parameters,
                                       f['parameters'].value)
        return net

    def __init__(self, layers, buffer_manager, architecture, seed=None,
                 handler=default_handler):
        super(Network, self).__init__(seed)
        self.layers = layers
        self.loss_layers = _get_loss_layers(layers)
        self._buffer_manager = buffer_manager
        self.buffer = self._buffer_manager.views
        self.architecture = architecture
        self.handler = None
        self.set_handler(handler)
        self.initializers = {}
        self.weight_modifiers = {}
        self.gradient_modifiers = {}
        self.output_name = None

    def get(self, buffer_path):
        """
        Get a numpy copy of the buffer corresponding to buffer_path.

        Examples:
            >>> parameters = net.get('parameters')
            >>> outputs = net.get('OutputLayer.outputs.probabilities')
            >>> forget_gates = net.get('Lstm.internals.Fb')

        Args:
            buffer_path (str):
                A dotted path to the buffer that should be copied and returned.

        Returns:
            numpy.ndarray:
                A numpy array copy of the specified buffer.

        Raises:
            KeyError:
                If no buffer is found for the given path.
        """
        b = self.buffer[buffer_path]
        if isinstance(b, BufferView):
            raise KeyError('buffer_path lead to a buffer, but a BufferView. '
                           'Try appending one of the following to your path: '
                           '{}'.format(', '.join(sorted(b.keys()))))
        return self.handler.get_numpy_copy(b)

    def get_input(self, input_name):
        """
        Get a numpy copy of one of the named inputs that are currently used.

        Args:
            input_name (str): The name of the input that should be retrieved.

        Returns:
            numpy.ndarray:
                A numpy array copy of the specified input.
        """
        return self.get('Input.outputs.' + input_name)

    # -------------------------- Setup Methods --------------------------------

    def initialize(self, default_or_init_dict=None, seed=None, **kwargs):
        """Initialize the weights of the network.

        Initialization can be specified in three equivalent ways:

            1. just a default initializer:

                >>> net.initialize(Gaussian())

                Note that this is equivalent to:

                >>> net.initialize(default=Gaussian())

            2. by passing a dictionary:

                >>> net.initialize({'RegularLayer': Uniform(),
                ...                 'LstmLayer': Gaussian()})

            3. by using keyword arguments:

                >>> net.initialize(RegularLayer=Uniform(),
                ...                LstmLayer=Uniform())

        All following explanations will be with regards to the dictionary style
        of initialization, because it is the most general one.

        Note:
            It is not recommended to combine 2. and 3. but if they are,
            then keyword arguments take precedence.

        Each initialization consists of a layer-pattern and that maps to an
        initializer or a weight-pattern dictionary.

        Layer patterns can take the following forms:

            1. ``{'layer_name': INIT_OR_SUBDICT}``
               Matches all the weights of the layer named layer_name
            2. ``{'layer_*': INIT_OR_SUBDICT}``
               Matches all layers with a name that starts with ``layer_``
               The wild-card ``*`` can appear at arbitrary positions and even
               multiple times in one path.

        There are two special layer patterns:

            3. ``{'default': INIT}``
               Matches all weights that are not matched by any other
               path-pattern
            4. ``{'fallback': INIT}``
               Set a fallback initializer for every weight. It will only be
               evaluated for the weights for which the regular initializer
               failed with an InitializationError.

               `This is useful for initializers that require a certain shape
               of weights and will not work otherwise. The fallback will then
               be used for all cases when that initializer failed.`

        The weight-pattern sub-dictionary follows the same form as the layer-
        pattern:

            1) ``{'layer_pattern': {'a': INIT_A, 'b': INIT_B}}``
            2) ``{'layer_pattern': {'a*': INIT}``
            3) ``{'layer_pattern': {'default': INIT}``
            4) ``{'layer_pattern': {'fallback': INIT}``


        An initializer can either be a scalar, something that converts to a
        numpy array of the correct shape or an :class:`Initializer` object.
        So for example:

        >>> net.initialize(default=0,
        ...                RnnLayer={'b': [1, 2, 3, 4, 5]},
        ...                ForwardLayer=bs.Gaussian())

        Note:
            Each view must match exactly one initialization and up to one
            fallback to be unambiguous. Otherwise the initialization will fail.

        You can specify a seed to make the initialization reproducible:

        >>> net.initialize({'default': bs.Gaussian()}, seed=1234)
        """
        init_refs = _update_references_with_dict(default_or_init_dict, kwargs)
        self.initializers = get_description(init_refs)
        all_parameters = {k: v.parameters
                          for k, v in self.buffer.items()
                          if isinstance(v, BufferView) and 'parameters' in v}
        _replace_lists_with_array_initializers(init_refs)
        initializers, fallback = resolve_references(all_parameters, init_refs)
        init_rnd = self.rnd.create_random_state(seed)
        for layer_name, views in sorted(all_parameters.items()):
            if views is None:
                continue
            for view_name, view in sorted(views.items()):
                init = initializers[layer_name][view_name]
                fb = fallback[layer_name][view_name]
                if len(init) > 1:
                    raise NetworkValidationError(
                        "Multiple initializers for {}.{}: {}".format(
                            layer_name, view_name, init))

                if len(init) == 0:
                    raise NetworkValidationError("No initializer for {}.{}".
                                                 format(layer_name, view_name))
                if len(fb) > 1:
                    raise NetworkValidationError(
                        "Multiple fallbacks for {}.{}: {}".format(
                            layer_name, view_name, fb))

                fb = fb.pop() if len(fb) else None
                self.handler.set_from_numpy(
                    view,
                    evaluate_initializer(init.pop(), view.shape, fb,
                                         seed=init_rnd.generate_seed()))

    def set_weight_modifiers(self, default_or_mod_dict=None, **kwargs):
        """
        Install
        :class:`ValueModifiers <brainstorm.value_modifiers.ValueModifier>` in
        the network to change the weights.

        They can be run manually using :meth:`.apply_weight_modifiers`,
        but they will also be called by the trainer after each weight update.

        Value modifiers can be set for specific weights in the same way
        initializers can, but there is no fallback.
        (see :meth:`.initialize` for details)


        A modifier can be a ValueModifiers object or a list of them.
        So for example:

        >>> net.set_weight_modifiers(
        ...    default=bs.ClipValues(-1, 1)
        ...    FullyConnectedLayer={'W': [bs.RescaleIncomingWeights(),
        ...                               bs.MaskValues(my_mask)]}
        ...    )

        Note:
            The order in which ValueModifiers appear in the list matters,
            because it is the same order in which they will be executed.
        """
        weight_mod_refs = _update_references_with_dict(default_or_mod_dict,
                                                       kwargs)
        all_parameters = {k: v.parameters
                          for k, v in self.buffer.items()
                          if k not in ['parameters', 'gradients'] and
                          'parameters' in v}
        weight_mods, fallback = resolve_references(all_parameters,
                                                   weight_mod_refs)

        assert not prune_view_references(fallback), \
            'fallback is not supported for weight modifiers'
        weight_mods = prune_view_references(weight_mods)
        self.weight_modifiers = order_and_copy_modifiers(weight_mods)
        # TODO: Check that all are ValueModifiers

    def set_gradient_modifiers(self, default_or_mod_dict=None, **kwargs):
        """
        Install
        :class:`ValueModifiers <brainstorm.value_modifiers.ValueModifier>` in
        the network to change the gradient.

        They can be run manually using :meth:`.apply_gradient_modifiers`, but
        they will also be called by the network after each backward pass.

        Gradient modifiers can be set for specific weights in the same way as
        initializers can, but there is no fallback.
        (see :meth:`.initialize` for details)

        A modifier can be a ValueModifiers object or a list of them.
        So for example:

        >>> net.set_gradient_modifiers(
        ...    default=bs.value_modifiers.ClipValues(-1, 1)
        ...    FullyConnectedLayer={'W': [bs.value_modifiers.ClipValues(),
        ...                               bs.value_modifiers.MaskValues(MASK)]}
        ...    )

        Note:
            The order in which ValueModifiers appear in the list matters,
            because it is the same order in which they will be executed.
        """
        gradient_mod_refs = _update_references_with_dict(default_or_mod_dict,
                                                         kwargs)
        all_parameters = {k: v.gradients
                          for k, v in self.buffer.items()
                          if k not in ['parameters', 'gradients'] and
                          'gradients' in v}
        gradient_mods, fallback = resolve_references(all_parameters,
                                                     gradient_mod_refs)

        assert not prune_view_references(fallback), \
            'fallback is not supported for gradient modifiers'
        gradient_mods = prune_view_references(gradient_mods)
        self.gradient_modifiers = order_and_copy_modifiers(gradient_mods)
        # TODO: Check that all are ValueModifiers or GradientModifiers

    def set_handler(self, new_handler):
        """
        Change the handler of this network.

        Examples:
            Use this to run a network on the GPU using the pycuda:

            >>> from brainstorm.handlers import PyCudaHandler
            >>> net.set_handler(PyCudaHandler())

        Args:
            new_handler (brainstorm.handlers.base_handler.Handler):
                The new handler this network should use.
        """
        self.handler = new_handler
        self._buffer_manager.set_handler(new_handler)
        self.buffer = self._buffer_manager.views
        for layer in self.layers.values():
            layer.set_handler(new_handler)

    # -------------------------- Running Methods ------------------------------

    def provide_external_data(self, data, all_inputs=True):
        """
        Provide the data for this network to perform its forward and backward
        passes on.

        Args:
            data (dict):
                A dictionary of the data that will be copied to the outputs of
                the Input layer.
            all_inputs (bool):
                If set to False this method will NOT check that all inputs are
                provided. Defaults to True.
        """
        time_size, batch_size = data[next(iter(data))].shape[: 2]
        self.buffer = self._buffer_manager.resize(time_size, batch_size)
        for name, buf in self.buffer.Input.outputs.items():
            if name not in data.keys() and all_inputs is False:
                continue
            if isinstance(data[name], self.handler.array_type):
                self.handler.copy_to(data[name], buf)
            else:
                # assert isinstance(data[name], np.ndarray)
                self.handler.set_from_numpy(buf, data[name])

    def forward_pass(self, training_pass=False, context=None):
        """
        Perform a forward pass on all the provided data.

        Note:
            All the input data to be used during this forward pass have to be
            passed to the network beforehand using
            :meth:`.provide_external_data`

        Args:
            training_pass (Optional[bool]):
                Indicates whether this forward pass belongs to training or not.
                This might change the behaviour of some layers.
            context (Optional[dict]):
                An optional network state as created by net.get_context().
                If provided the network will treat this as if it was the the
                state of the network at the t=-1. This is useful for continuing
                the computations of a recurrent neural network.
                Defaults to None.
        """
        if context is None:
            self._buffer_manager.clear_context()
        else:
            self._buffer_manager.apply_context(context)
        for layer_name, layer in list(self.layers.items())[1:]:
            layer.forward_pass(self.buffer[layer_name], training_pass)

    def backward_pass(self):
        """
        Perform a backward pass on all provided data and targets.

        Note:
            All the targets to be used during this backward pass have to be
            passed to the network beforehand using provide_external_data.
            Also this backward pass depends on the internal state produced by
            a forward pass. So you have to always run a forward_pass first.
        """
        self._buffer_manager.clear_backward_buffers()
        for layer_name, layer in reversed(list(self.layers.items())[1:]):
            layer.backward_pass(self.buffer[layer_name])
        self.apply_gradient_modifiers()

    def get_loss_values(self):
        """
        Get a dictionary of all the loss values that resulted from a
        forward pass.

        For simple networks with just one loss the dictionary has only
        a single entry called 'total_loss'.

        If there are multiple Loss layers the dictionary will also contain an
        entry for each Loss layer mapping its name to its loss, and the
        'total_loss' entry will contain the sum of all of them.

        Returns:
            dict[str, float]:
                A dictionary of all loss values that this network produced.
        """
        loss = 0.
        losses = OrderedDict()
        if len(self.loss_layers) == 1:
            losses['total_loss'] = float(self.get(self.loss_layers[0] +
                                                  '.outputs.loss'))
            return losses
        for loss_layer_name in self.loss_layers:
            l = float(self.get(loss_layer_name + '.outputs.loss'))
            losses[loss_layer_name] = l
            loss += l

        losses['total_loss'] = loss
        return losses

    def get_context(self):
        """
        Get the last timestep internal state of this network.
        (after a forward pass)
        This can be passed to the forward_pass method as context to continue
        a batch of sequences.

        Returns:
            dict:
                Internal state of this network at the last timestep.
        """
        return self._buffer_manager.get_context()

    def apply_weight_modifiers(self):
        for layer_name, views in self.weight_modifiers.items():
            for view_name, weight_mods in views.items():
                for wm in weight_mods:
                    wm.rnd.set_seed(self.rnd.generate_seed())
                    wm(self.handler,
                       self.buffer[layer_name].parameters[view_name])

    def apply_gradient_modifiers(self):
        for layer_name, views in self.gradient_modifiers.items():
            for view_name, gradient_mods in views.items():
                for gm in gradient_mods:
                    gm.rnd.set_seed(self.rnd.generate_seed())
                    if isinstance(gm, GradientModifier):
                        gm(self.handler,
                           self.buffer[layer_name].parameters[view_name],
                           self.buffer[layer_name].gradients[view_name])
                    else:
                        gm(self.handler,
                           self.buffer[layer_name].gradients[view_name])

    # -------------------------- Serialization --------------------------------

    def save_as_hdf5(self, filename, comment=''):
        """
        Save this network as an HDF5 file.
        The file will contain a description of this network and the parameters.

        Args:
            filename (str):
                Name of the file this network should be saved to.
                All directories have to exist already.

            comment (Optional[str]):
                An optional comment that will be saved inside the file.
        """
        with h5py.File(filename, 'w') as f:
            f.attrs.create('info', get_brainstorm_info())
            f.attrs.create('format', b'Network file v1.0')
            if comment:
                f.attrs.create('comment', comment.encode())
            description = get_description(self)
            f['description'] = json.dumps(description).encode()
            f.create_dataset(
                'parameters', compression='gzip',
                data=self.get('parameters'))


# ########################### Helper Methods ##################################

def _get_loss_layers(layers):
    return [name for name, l in layers.items() if isinstance(l, LossLayerImpl)]


def _update_references_with_dict(refs, ref_dict):
    if refs is None:
        references = dict()
    elif isinstance(refs, dict):
        references = refs
    else:
        references = {'default': refs}

    if set(references.keys()) & set(ref_dict.keys()):
        raise TypeError('Conflicting values for %s!' %
                        sorted(set(references.keys()) & set(ref_dict.keys())))

    references.update(ref_dict)

    return references


def _replace_lists_with_array_initializers(ref_dict):
    for key, value in ref_dict.items():
        if isinstance(value, dict):
            _replace_lists_with_array_initializers(value)
        elif isinstance(value, (list, np.ndarray)):
            ref_dict[key] = ArrayInitializer(value)
