==========
Brainstorm
==========

Brainstorm is a library designed to make working with neural networks fast, flexible and fun.

Installation
------------

Note: These instructions are for Ubuntu 14.04.

Pre-requisites:

You need the Ubuntu packages `python-dev`, `libhdf5-dev` as well as the Python package `cython` before you install. Then:

* Clone the repository: `git clone git@github.com:Qwlouse/brainstorm.git`
* Install: `cd brainstorm; pip install -e .`

If you'd like to use an NVIDIA GPU, make sure you have CUDA installed, then:

* Get latest PyCUDA: `pip install git+ssh://git@github.com/inducer/pycuda#egg=pycuda`
* Get latest scikit-cuda: `pip install git+ssh://git@github.com/lebedov/scikit-cuda#egg=scikit-cuda`

If you'd like to use convolutional/pooling layers on the GPU, these are provided through NVIDIA cuDNN which you should install from https://developer.nvidia.com/cudnn
