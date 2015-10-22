#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from six.moves.urllib.request import urlretrieve
import numpy as np
import zipfile
import h5py
import os


def convert_to_batches(serial_data, length, bs):
    assert serial_data.size % length == 0
    num_sequences = serial_data.size // length
    assert num_sequences % bs == 0
    num_batches = num_sequences // bs
    serial_data = serial_data.reshape((bs, num_batches * length))
    serial_data = np.vstack(np.hsplit(serial_data, num_batches)).T[:, :, None]
    return serial_data

batch_size = 100
# Batch size which will be used for training.
# Needed to maintain continuity of data across batches.
seq_len = 100
# Number of characters in each sub-sequence.
# Limits the number of time-steps that the gradient is back-propagated.
num_test_chars = 5000000
# Number of characters which will be used for testing.
# An equal number of characters will be used for validation.

bs_data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '.')
url = 'http://mattmahoney.net/dc/enwik8.zip'
hutter_file = os.path.join(bs_data_dir, 'enwik8.zip')
hdf_file = os.path.join(bs_data_dir, 'HutterPrize.hdf5')

print("Using data directory:", bs_data_dir)
if not os.path.exists(hutter_file):
    print("Downloading Hutter Prize data ...")
    urlretrieve(url, hutter_file)
    print("Done.")

print("Extracting Hutter Prize data ...")
raw_data = zipfile.ZipFile(hutter_file).read('enwik8')
print("Done.")

print("Preparing data for Brainstorm ...")
raw_data = np.fromstring(raw_data, dtype=np.uint8)
unique, data = np.unique(raw_data, return_inverse=True)

train_data = data[: -2 * num_test_chars]
train_targets = data[1: -2 * num_test_chars + 1]
valid_data = data[-2 * num_test_chars: -num_test_chars]
valid_targets = data[-2 * num_test_chars + 1: -num_test_chars + 1]
test_data = data[-num_test_chars:]
test_targets = np.append(data[-num_test_chars + 1:], [0])

# Convert to batches
train_data = convert_to_batches(train_data, seq_len, batch_size)
train_targets = convert_to_batches(train_targets, seq_len, batch_size)

valid_data = convert_to_batches(valid_data, seq_len, batch_size)
valid_targets = convert_to_batches(valid_targets, seq_len, batch_size)

test_data = convert_to_batches(test_data, seq_len, batch_size)
test_targets = convert_to_batches(test_targets, seq_len, batch_size)
print("Done.")

print("Creating Hutter Prize character-level HDF5 dataset ...")
f = h5py.File(hdf_file, 'w')
description = """
The Hutter Prize Wikipedia dataset, prepared for character-level language
modeling.

The data was obtained from the link:
http://mattmahoney.net/dc/enwik8.zip

Attributes
==========

description: This description.

unique: A 1-D array of unique characters (0-255 ASCII values) in the dataset.
The index of each character was used as the class ID for preparing the data.

Variants
========

split: Split into 'training', 'validation' and 'test' tests of size 90, 10 and
10 million characters respectively. Each sequence is {} characters long. The
dataset has been prepared expecting minibatches of {} sequences.
""".format(seq_len, batch_size)
f.attrs['description'] = description
f.attrs['unique'] = unique

variant = f.create_group('split')
group = variant.create_group('training')
group.create_dataset(name='default', data=train_data, compression='gzip')
group.create_dataset(name='targets', data=train_targets, compression='gzip')

group = variant.create_group('validation')
group.create_dataset(name='default', data=valid_data, compression='gzip')
group.create_dataset(name='targets', data=valid_targets, compression='gzip')

group = variant.create_group('test')
group.create_dataset(name='default', data=test_data, compression='gzip')
group.create_dataset(name='targets', data=test_targets, compression='gzip')

f.close()
print("Done.")
