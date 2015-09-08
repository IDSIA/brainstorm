==========
Brainstorm
==========

Brainstorm is a library designed to make working with neural networks fast, flexible and fun.

Installation
------------

Note: These instructions are for Ubuntu 14.04.

Pre-requisites: You need the Ubuntu packages `python-dev`, `libhdf5-dev` as well as the Python package `cython` before you install. 

To install:

* Clone the repository: `git clone git@github.com:Qwlouse/brainstorm.git`
* Install: `cd brainstorm; pip install -e .`

If you'd like to use an NVIDIA GPU, make sure you have CUDA installed, then:

* Get latest PyCUDA: `pip install git+ssh://git@github.com/inducer/pycuda#egg=pycuda`
* Get latest scikit-cuda: `pip install git+ssh://git@github.com/lebedov/scikit-cuda#egg=scikit-cuda`

If you'd like to use convolutional/pooling layers on the GPU, these are provided through NVIDIA cuDNN which you should install from https://developer.nvidia.com/cudnn
Then install the latest Python wrappers for cuDNN (2.0b2 as of this writing): `pip install cudnn-python-wrappers==2.0b2`

Brainstorm uses the HDF5 file format to store datasets, networks etc. It is recommended to use a single location to store all datasets prepared and used by Brainstorm, which can be specified by setting the environment variable `BRAINSTORM_DATA_DIR` in your .bashrc file.

