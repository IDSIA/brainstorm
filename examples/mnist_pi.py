#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import h5py

import brainstorm as bs
from brainstorm.handlers import PyCudaHandler
from brainstorm.data_iterators import Minibatches
import os

bs.global_rnd.set_seed(42)

# ---------------------------- Set up Iterators ----------------------------- #

data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '../data')
data_file = os.path.join(data_dir, 'MNIST.hdf5')
ds = h5py.File(data_file, 'r')['normalized_split']
x_tr, y_tr = ds['training']['default'][:], ds['training']['targets'][:]
x_va, y_va = ds['validation']['default'][:], ds['validation']['targets'][:]
x_te, y_te = ds['test']['default'][:], ds['test']['targets'][:]

getter_tr = Minibatches(100, default=x_tr, targets=y_tr)
getter_va = Minibatches(100, default=x_va, targets=y_va)
getter_te = Minibatches(100, default=x_te, targets=y_te)

# ----------------------------- Set up Network ------------------------------ #

inp, out = bs.tools.get_in_out_layers_for_classification(784, 10)
network = bs.Network.from_layer(
    inp >>
    bs.layers.Dropout(drop_prob=0.2) >>
    bs.layers.FullyConnected(1200, name='Hid1', activation='rel') >>
    bs.layers.Dropout(drop_prob=0.5) >>
    bs.layers.FullyConnected(1200, name='Hid2', activation='rel') >>
    bs.layers.Dropout(drop_prob=0.5) >>
    out
)

network.set_memory_handler(PyCudaHandler(init_cudnn=False))
network.initialize(bs.initializers.Gaussian(0.01))
network.set_weight_modifiers({"Output": bs.value_modifiers.ConstrainL2Norm(1)})

# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.training.MomentumStep(learning_rate=0.1, momentum=0.9),
                     double_buffering=False)
trainer.add_hook(bs.hooks.ProgressBar())
scorers = [bs.scorers.Accuracy(out_name='Output.probabilities')]
trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers,
                                        name='validation'))
trainer.add_hook(bs.hooks.SaveBestNetwork('validation.Accuracy',
                                          filename='mnist_pi_best.hdf5',
                                          name='best weights',
                                          criterion='max'))
trainer.add_hook(bs.hooks.StopAfterEpoch(500))

# -------------------------------- Train ------------------------------------ #

trainer.train(network, getter_tr, valid_getter=getter_va)
print("Best validation set accuracy:",
      max(trainer.logs["validation"]["Accuracy"]))
