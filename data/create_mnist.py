#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import gzip
import pickle
from six.moves.urllib.request import urlretrieve
import numpy as np
import h5py
import os
import sys

bs_data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '.')
url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
mnist_file = os.path.join(bs_data_dir, 'mnist.pkl.gz')
hdf_file = os.path.join(bs_data_dir, 'MNIST.hdf5')

print("Using data directory:", bs_data_dir)
if not os.path.exists(mnist_file):
    print("Downloading MNIST data ...")
    urlretrieve(url, mnist_file)
    print("Done.")

print("Extracting MNIST data ...")
with gzip.open(mnist_file, 'rb') as f:
    if sys.version_info < (3,):
        ds = pickle.load(f)
    else:
        ds = pickle.load(f, encoding='latin1')
print("Done.")

train_inputs, train_targets = \
    ds[0][0].reshape((1, 50000, 28, 28, 1)), ds[0][1].reshape((1, 50000, 1))
valid_inputs, valid_targets = \
    ds[1][0].reshape((1, 10000, 28, 28, 1)), ds[1][1].reshape((1, 10000, 1))
test_inputs, test_targets = \
    ds[2][0].reshape((1, 10000, 28, 28, 1)), ds[2][1].reshape((1, 10000, 1))

print("Creating HDF5 dataset ...")
f = h5py.File(hdf_file, 'w')
description = """
The MNIST handwritten digit database is a preprocessed subset of NIST's
Special Database 3 and Special Database 1. It consists of 70000 28x28 binary
images of handwritten digits (0-9), with approximately 7000 images per class.
There are 60000 training images and 10000 test images.

The dataset was obtained from the link:
http://deeplearning.net/data/mnist/mnist.pkl.gz
which hosts a normalized version of the data originally from:
http://yann.lecun.com/exdb/mnist/

Attributes
==========

description: This description.

Variants
========

normalized_full: Contains 'training' and 'test' sets. Image data has been
normalized by dividing all pixel values by 255.

normalized_split: Contains 'training' (first 50K out of the full training
set), 'validation' (remaining 10K out of the full training set) and
'test' sets. Image data has been normalized by dividing all pixel values by
255.

"""
f.attrs['description'] = description

variant = f.create_group('normalized_split')
group = variant.create_group('training')
group.create_dataset(name='default', data=train_inputs, compression='gzip')
group.create_dataset(name='targets', data=train_targets, compression='gzip')

group = variant.create_group('validation')
group.create_dataset(name='default', data=valid_inputs, compression='gzip')
group.create_dataset(name='targets', data=valid_targets, compression='gzip')

group = variant.create_group('test')
group.create_dataset(name='default', data=test_inputs, compression='gzip')
group.create_dataset(name='targets', data=test_targets, compression='gzip')


train_inputs = np.concatenate((train_inputs, valid_inputs), axis=1)
train_targets = np.concatenate((train_targets, valid_targets), axis=1)

variant = f.create_group('normalized_full')
group = variant.create_group('training')
group.create_dataset(name='default', data=train_inputs, compression='gzip')
group.create_dataset(name='targets', data=train_targets, compression='gzip')

group = variant.create_group('test')
group.create_dataset(name='default', data=test_inputs, compression='gzip')
group.create_dataset(name='targets', data=test_targets, compression='gzip')

f.close()
print("Done.")
