#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict
import os

import h5py

import brainstorm as bs
from brainstorm.layers import FullyConnected
from brainstorm.data_iterators import Minibatches
from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (StructureTemplate,
                                                   BufferStructure)
from brainstorm.structure.construction import ConstructionWrapper

bs.global_rnd.set_seed(42)


# ------------------------------ Custom Layer ------------------------------- #

# This function will be used for wiring the layers
def Square(name=None):
    """Create a layer that squares its inputs elementwise"""
    return ConstructionWrapper.create(SquareLayerImpl, name=name)


# Layer implementations need to inherit from brainstorm.layers.base_layer.Layer
# And their class name needs to end with 'LayerImpl'
class SquareLayerImpl(Layer):
    # accept inputs in any format
    expected_inputs = {'default': StructureTemplate('...')}
    # no kwargs supported
    expected_kwargs = {}

    # For a custom layer we need to implement the following 3 methods:

    def setup(self, kwargs, in_shapes):
        # In this method we set up the buffer structure of the layer
        # we can use the kwargs passed to this layer (here we don't)
        # and the shapes of the inputs (an OrderedDict[str, BufferStructure])

        # This layer is elementwise so the output shapes should be the same as
        # the input shapes
        outputs = in_shapes
        parameters = OrderedDict()  # No parameters so this is empty
        internals = OrderedDict()   # Also no need for internal buffers
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        self.handler.mult_tt(inputs, inputs, outputs)
        self.handler.mult_st(0.5, outputs, outputs)

    def backward_pass(self, buffers):
        inputs = buffers.inputs.default
        output_deltas = buffers.output_deltas.default
        input_deltas = buffers.input_deltas.default
        self.handler.mult_add_tt(inputs, output_deltas, input_deltas)

# --------------------------- Testing the Layer ----------------------------- #
# Testing doesn't need to happen before every run. We recommend
# having a layer implementation + the tests in a separate file.

from brainstorm.tests.tools import get_test_configurations, run_layer_tests

for cfg in get_test_configurations():
    layer = SquareLayerImpl('Square',
                            {'default': BufferStructure('T', 'B', 3)},
                            set(), set())
    run_layer_tests(layer, cfg)


# ------------------------------ Demo Example ------------------------------- #

# ---------------------------- Set up Iterators ----------------------------- #

data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '../data')
data_file = os.path.join(data_dir, 'MNIST.hdf5')
ds = h5py.File(data_file, 'r')['normalized_split']
x_tr, y_tr = ds['training']['default'][:], ds['training']['targets'][:]
x_va, y_va = ds['validation']['default'][:], ds['validation']['targets'][:]

getter_tr = Minibatches(100, default=x_tr, targets=y_tr)
getter_va = Minibatches(100, default=x_va, targets=y_va)

# ----------------------------- Set up Network ------------------------------ #

inp, out = bs.tools.get_in_out_layers('classification', (28, 28, 1), 10)
network = bs.Network.from_layer(
    inp >>
    FullyConnected(500, name='Hid1', activation='linear') >>
    Square(name='MySquareLayer') >>
    out
)

network.initialize(bs.initializers.Gaussian(0.01))

# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.training.MomentumStepper(learning_rate=0.01,
                                                 momentum=0.9))
trainer.add_hook(bs.hooks.ProgressBar())
scorers = [bs.scorers.Accuracy(out_name='Output.outputs.predictions')]
trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers,
                                        name='validation'))
trainer.add_hook(bs.hooks.EarlyStopper('validation.Accuracy', patience=10, criterion='max'))
trainer.add_hook(bs.hooks.StopAfterEpoch(500))

# -------------------------------- Train ------------------------------------ #

trainer.train(network, getter_tr, valid_getter=getter_va)
print("Best validation set accuracy:", max(trainer.logs["validation"]["Accuracy"]))
