############
Installation
############

Common Install Notes
====================

A basic requirement to use Brainstorm is Numpy, and we recommend that you make sure that you have a fast `BLAS <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_ installation which Numpy will use.
OpenBLAS is excellent, and can be installed on a Debian family system easily: ``sudo apt-get install libopenblas-dev``.

Brainstorm provide a ``PyCudaHandler`` which can be used to accelerate neural networks using Nvidia GPUs.
In order to use it, you need to have CUDA 7.0 or later already installed and setup from https://developer.nvidia.com/cuda-downloads

Installation variants
=====================

When installing from PyPI or GitHub, you can specify the following installation variants to additionally install optional dependencies:

all
pycuda
tests
live_viz
draw_net

******
Ubuntu
******

Install prerequisites:

.. code-block:: bash

    sudo apt-get install python-dev libhdf5-dev libopenblas-dev

Install the latest stable release from PyPI, including all optional dependencies:

.. code-block:: bash

    pip install brainstorm[all]

which will additionally install pycuda, scikit-cuda, pygraphviz and bokeh.

To install the latest master branch, you can do:

.. code-block:: bash

    pip install git+git@github.com:IDSIA/brainstorm.git#egg=brainstorm[all]

****
OS X
****

Instructions coming soon.

*******
Windows
*******

Instructions coming soon.

******
Docker
******

Builds of Brainstorm are available as `Docker <https://www.docker.com>`_
images: `Brainstorm Docker (CPU) <https://hub.docker.com/r/kaixhin/brainstorm/>`_ or
`Brainstorm Docker (CUDA) <https://hub.docker.com/r/kaixhin/cuda-brainstorm/>`_. These
are updated on a weekly basis with bleeding-edge builds of Brainstorm.
Examples of running bash in a Docker container are as follows:

.. code-block:: bash

    sudo docker run -it kaixhin/brainstorm
    sudo nvidia-docker run -it kaixhin/cuda-brainstorm:7.0

For a guide to Docker, see the `official docs <https://docs.docker.com>`_.
CUDA support requires `NVIDIA Docker <https://github.com/NVIDIA/nvidia-docker>`_.
For more details on how to use the Brainstorm Docker images,
consult the `source project <https://github.com/Kaixhin/dockerfiles>`_.
