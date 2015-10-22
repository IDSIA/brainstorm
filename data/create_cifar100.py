#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from six.moves.urllib.request import urlretrieve
import numpy as np
import tarfile
import h5py
import os

bs_data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '.')
url = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
cifar100_file = os.path.join(bs_data_dir, 'cifar-100-binary.tar.gz')
hdf_file = os.path.join(bs_data_dir, 'CIFAR-100.hdf5')

print("Using data directory:", bs_data_dir)
if not os.path.exists(cifar100_file):
    print("Downloading CIFAR-100 data ...")
    urlretrieve(url, cifar100_file)
    print("Done.")

archive_paths = [
    'cifar-100-binary/train.bin',
    'cifar-100-binary/test.bin'
]

print("Extracting CIFAR-100 data ...")
with tarfile.open(cifar100_file) as f:
    res = []
    for fn in archive_paths:
        buf = f.extractfile(fn).read()
        tmp = np.fromstring(buf, dtype=np.uint8)
        tmp = tmp.reshape(-1, 1024 * 3 + 2)
        res.append(tmp)
print("Done.")

ds = np.concatenate(res)
nr_tr = 40000
x_tr = ds[:nr_tr, 2:].reshape((1, nr_tr, 3, 32, 32))
x_tr = x_tr.transpose([0, 1, 3, 4, 2])
y_tr = ds[:nr_tr, 1] .reshape((1, nr_tr, 1))

x_va = ds[40000: 50000, 2:].reshape((1, 10000, 3, 32, 32))
x_va = x_va.transpose([0, 1, 3, 4, 2])
y_va = ds[40000: 50000, 1].reshape((1, 10000, 1))

x_te = ds[50000:, 2:].reshape((1, 10000, 3, 32, 32))
x_te = x_te.transpose([0, 1, 3, 4, 2])
y_te = ds[50000:, 1].reshape((1, 10000, 1))

tr_mean = x_tr.squeeze().mean(axis=0)
tr_std = x_tr.squeeze().std(axis=0)
x_tr = (x_tr - tr_mean) / tr_std
x_va = (x_va - tr_mean) / tr_std
x_te = (x_te - tr_mean) / tr_std

print("Creating CIFAR-100 HDF5 dataset ...")
f = h5py.File(hdf_file, 'w')
description = """
The CIFAR-100 dataset is a labeled subset of the 80 million tiny images
dataset. It consists of 60000 32x32 colour images in 100 classes, with 600
images per class. There are 50000 training images and 10000 test images.

The dataset was obtained from the link:
http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz

Attributes
==========

description: This description.

mean: The pixel-wise mean of the first 40000 training set images.

std: The pixel-wise standard deviation of the first 40000 training set images.

std:

Variants
========

normalized_full: Contains 'training' and 'test' sets. Image data has been
normalized to have zero mean and unit standard deviation.

normalized_split: Contains 'training' (first 40K out of the full training
set), 'validation' (remaining 10K out of the full training set) and
'test' sets. Image data has been normalized to have zero mean and unit
standard deviation.

"""
f.attrs['description'] = description
f.attrs['mean'] = tr_mean
f.attrs['std'] = tr_std

variant = f.create_group('normalized_split')
group = variant.create_group('training')
group.create_dataset(name='default', data=x_tr, compression='gzip')
group.create_dataset(name='targets', data=y_tr, compression='gzip')

group = variant.create_group('validation')
group.create_dataset(name='default', data=x_va, compression='gzip')
group.create_dataset(name='targets', data=y_va, compression='gzip')

group = variant.create_group('test')
group.create_dataset(name='default', data=x_te, compression='gzip')
group.create_dataset(name='targets', data=y_te, compression='gzip')

nr_tr = 50000
x_tr = ds[:nr_tr, 2:].reshape((1, nr_tr, 3, 32, 32)).transpose([0, 1, 3, 4, 2])
x_tr = (x_tr - tr_mean) / tr_std
y_tr = ds[:nr_tr, 1].reshape((1, nr_tr, 1))

variant = f.create_group('normalized_full')
group = variant.create_group('training')
group.create_dataset(name='default', data=x_tr, compression='gzip')
group.create_dataset(name='targets', data=y_tr, compression='gzip')

group = variant.create_group('test')
group.create_dataset(name='default', data=x_te, compression='gzip')
group.create_dataset(name='targets', data=y_te, compression='gzip')

f.close()
print("Done.")
