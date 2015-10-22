==========
Brainstorm
==========

Brainstorm is a library designed to make working with neural networks fast, flexible and fun.

For help in using Brainstorm, please use the [mailing list](https://groups.google.com/forum/#!forum/mailstorm).

Installation
------------

Note: These instructions are for Ubuntu 14.04.

Pre-requisites: You need the Ubuntu packages `python-dev`, `libhdf5-dev` as well as the Python package `cython` before you install. 

To install:

* Clone the repository: `git clone git@github.com:Qwlouse/brainstorm.git`
* To install for development: `cd brainstorm; pip install -e .`. This links the installed library directly to the brainstorm directory, so that changes made to the library can be directly used.
* To install for usage: `cd brainstorm; python setup.py install`

If you'd like to use an NVIDIA GPU, make sure you have CUDA installed, then:

* Get latest PyCUDA: `pip install -U git+https://github.com/inducer/pycuda#egg=pycuda`
* Get latest scikit-cuda: `pip install -U git+https://github.com/lebedov/scikit-cuda#egg=scikit-cuda`

If you'd like to use convolutional/pooling layers on the GPU, these are provided through NVIDIA cuDNN which you should install from https://developer.nvidia.com/cudnn

Then install the latest Python wrappers for cuDNN (2.0b2 as of this writing): `pip install cudnn-python-wrappers==2.0b2`

Brainstorm uses the HDF5 file format to store datasets, networks etc. It is recommended to use a single location to store all datasets prepared and used by Brainstorm, which can be specified by setting the environment variable `BRAINSTORM_DATA_DIR` in your .bashrc file.

