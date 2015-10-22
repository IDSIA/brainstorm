#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import brainstorm as bs
from brainstorm.handlers import PyCudaHandler
from brainstorm.data_iterators import Minibatches, DataIterator
from brainstorm.initializers import Gaussian
from brainstorm.training.steppers import MomentumStepper
from brainstorm.hooks import MonitorLoss
from brainstorm.value_modifiers import ClipValues
from brainstorm.tools import get_in_out_layers
import numpy as np
import os
import sys
import math
import numpy as np
import sys
from brainstorm.randomness import Seedable
from brainstorm.utils import IteratorValidationError
from brainstorm.handlers._cpuop import _crop_images

if sys.version_info < (3,):
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

# ------------------------------ Get the data ------------------------------- #

def preprocess_data(datafilepath):
    datafile = open(datafilepath, 'r').read()
    data_new = datafile.replace(' ', '')
    data_new = data_new.replace('\\/', '')
    return data_new

def prepare_data(data_new, vocab_dict, sequence_length=100):
    # Print sample text from data
    print('')
    print('Sample text:\n', data_new[0:200])
    # Map text to dictionary number entry corresponding to symbol or character
    data = np.array([vocab_dict[char] for char in data_new], dtype=int)

    reverse_dict = dict([(v, k) for (k, v) in vocab_dict.items()])
    data_reverse = [reverse_dict[char] for char in data]  # for testing purposes

    # Initialize mask: All values are taken into account
    data_mask = np.ones(data.shape)
    # Assign targets: One step ahead prediction
    data_targets = data[1:]  # one step ahead

    # Resize data to correspond to size of targets
    # (also last entry of data would have no corresponding prediction)
    data = data[0:-1]  # make same size as targets (last data point cannot predict anything)
    data_mask = data_mask[0:-1]
    N_batch = data.shape[0]//sequence_length

    # Allocate reshaped data arrays:
    new_data = np.zeros((sequence_length, N_batch, 1), dtype=int)
    new_target = np.zeros((sequence_length, N_batch, 1), dtype=int)
    new_mask = np.zeros((sequence_length, N_batch, 1), dtype=int)

    # Reshape the data into sequences of sequence_length:
    for i in range(N_batch):
        for j in range(sequence_length):
            new_data[j, i, 0] = data[j+i*sequence_length]
            new_mask[j, i, 0] = data_mask[j+i*sequence_length]
            new_target[j, i, 0] = data_targets[j+i*sequence_length]
    data = new_data

    # To show that data is still in order: Reverse map to characters from numbers
    data_reverse = [reverse_dict[char] for char in np.squeeze(np.append(data[:, 0], data[:, 1]))]
    print('Reversely mapped sample text: \n', ''.join(data_reverse[0:200]))
    print('')
    data_mask = new_mask
    data_targets = new_target

    assert(data.shape == data_targets.shape == data_mask.shape)  # Data, mask and targets should have the same shape
    assert(len(data.shape) == 3)  # Data should be of the from T, B, F (Time, Batch, Features)

    return data, data_targets, data_mask

# ----------------------------- Set up Data ------------------------------ #
""" The setting up of data still needs to be simplified by simply importing from a hdf5 file """
train = '../data/ptb.char.train.txt'
valid = '../data/ptb.char.valid.txt'
test = '../data/ptb.char.test.txt'

train = preprocess_data(train)
valid = preprocess_data(valid)
test = preprocess_data(test)

vocab = np.unique(list(train)+list(valid)+list(test))
vocab_dict = dict(zip(list(vocab), range(0, len(vocab))))

print('Length of Dictionary:', len(vocab_dict))
print('Dictionary:', vocab_dict)
sequence_length = 100  # length of time sequence into which the symbols are rearranged

train_inputs, train_targets, train_mask = prepare_data(train, vocab_dict, sequence_length)
valid_inputs, valid_targets, valid_mask = prepare_data(valid, vocab_dict, sequence_length)
test_inputs, test_targets, test_mask = prepare_data(test, vocab_dict, sequence_length)

# ----------------------------- Set up Network ------------------------------ #
n_classes = len(vocab_dict)

inp, out = bs.tools.get_in_out_layers(task_type='classification', in_shape=n_classes, out_shape=n_classes, outlayer_name='out',
                                                   mask_name='mask')
inp >> bs.layers.Lstm(1000, name='cw_lstm_peep', activation='tanh') >> out
network = bs.Network.from_layer(out)
network.set_handler(PyCudaHandler())
network.initialize({"default": bs.initializers.Gaussian(0.1)}, seed=42)
network.set_gradient_modifiers({"cw_lstm_peep": bs.value_modifiers.ClipValues(low=-1., high=1)})

# ---------------------------- Set up Iterators ----------------------------- #
train_getter = bs.data_iterators.Minibatches(1, True, mask=train_mask,  # WITH OR WITHOUT SHUFFLING?
                              default=train_inputs, targets=train_targets)
valid_getter = bs.data_iterators.Minibatches(100, True, mask=valid_mask,
                              default=valid_inputs, targets=valid_targets)
test_getter = bs.data_iterators.Minibatches(100, True, mask=test_mask,
                             default=test_inputs, targets=test_targets)

vocab_dict_name = {'default': len(vocab_dict)}
train_getter = bs.data_iterators.OneHot(train_getter, vocab_dict_name)
valid_getter = bs.data_iterators.OneHot(valid_getter, vocab_dict_name)
test_getter = bs.data_iterators.OneHot(test_getter, vocab_dict_name)

# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.training.MomentumStepper(learning_rate=0.01, momentum=0.99))
trainer.add_hook(bs.hooks.ProgressBar())
scorers = [bs.scorers.Accuracy(out_name='out.outputs.probabilities')]
trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers,
                                        name='validation'))
trainer.add_hook(bs.hooks.SaveBestNetwork('validation.Accuracy',
                                          filename='penn_corpus_lstm_shuffle2.hdf5',
                                          name='best weights',
                                          criterion='max'))
trainer.add_hook(bs.hooks.StopAfterEpoch(500))

# -------------------------------- Train ------------------------------------ #
trainer.train(network, train_getter, valid_getter=valid_getter)
print("\nBest validation accuracy: ", max(trainer.logs["validation accuracy"]))