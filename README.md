Brainstorm
==========

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square)](http://brainstorm.readthedocs.org/en/latest)
[![PyPi Version](https://img.shields.io/pypi/v/brainstorm.svg?style=flat-square)](https://pypi.python.org/pypi/brainstorm)
[![MIT license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](http://choosealicense.com/licenses/mit)
![Python Versions](https://img.shields.io/pypi/pyversions/brainstorm.svg?style=flat-square)

Brainstorm makes working with neural networks fast, flexible and fun.

Combining lessons from previous projects with new design elements, and written entirely in Python, Brainstorm has been designed to work on multiple platforms with multiple computing backends.


Getting Started
---------------
A good point to start is the brief [walkthrough](https://brainstorm.readthedocs.org/en/latest/walkthrough.html) of the ``cifar10_cnn.py`` example.  
More documentation is in progress, and hosted on [ReadTheDocs](https://brainstorm.readthedocs.org/en/latest/).
If you wish, you can also run the data preparation scripts (``data`` directory) and look at some basic examples (``examples`` directory).

Status
------
Brainstorm is under active development and is currently in beta. 

The currently available feature set includes recurrent (simple, LSTM, Clockwork), 2D convolution/pooling, Highway and batch normalization layers. API documentation is fairly complete and we are currently working on tutorials and usage guides.

Brainstorm abstracts computations via *handlers* with a consistent API. Currently, two handlers are provided: `NumpyHandler` for computations on the CPU (through Numpy/Cython) and `PyCudaHandler` for the GPU (through PyCUDA and scikit-cuda).

Installation
------------
Here are some quick instructions for installing the latest master branch on Ubuntu.

```bash
# Install pre-requisites
sudo apt-get update
sudo apt-get install python-dev libhdf5-dev git python-pip
# Get brainstorm
git clone https://github.com/IDSIA/brainstorm
# Install
cd brainstorm
[sudo] pip install -r requirements.txt
[sudo] python setup.py install
# Build local documentation (optional)
sudo apt-get install python-sphinx
make docs
# Install visualization dependencies (optional)
sudo apt-get install graphviz libgraphviz-dev pkg-config
[sudo] pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
```
To use your CUDA installation with brainstorm:
```bash
$ [sudo] pip install -r pycuda_requirements.txt
```
Set location for storing datasets:
```bash
echo "export BRAINSTORM_DATA_DIR=/home/my_data_dir/" >> ~/.bashrc
```

Help and Support
----------------

If you have any suggestions or questions, please post to the [Google group](https://groups.google.com/forum/#!forum/mailstorm).

If you encounter any errors or problems, please let us know by opening an issue.

License
-------

MIT License. Please see the LICENSE file.

Acknowledgements
----------------

Klaus Greff and Rupesh Srivastava would like to thank Jürgen Schmidhuber for his continuous supervision and encouragement.
Funding from EU projects NASCENCE (FP7-ICT-317662) and WAY (FP7-ICT-288551) was instrumental during the development of this project.
We also thank Nvidia Corporation for their donation of GPUs.
