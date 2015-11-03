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

If you have the pre-requisite packages `python-dev libhdf5-dev libopenblas-dev`, quick installation is simply: `pip install brainstorm`.
If you have CUDA set up already, `pip install brainstorm[all]` will install all optional dependencies.
See [Installation](https://brainstorm.readthedocs.org/en/latest/installation.html) instructions in the documentation for information about pre-requisites, GPU support, and various platforms.

Brainstorm is under active development and is currently in beta. 

Documentation is in progress, and hosted on [ReadTheDocs](https://brainstorm.readthedocs.org/en/latest/).
A good point to start is the brief [walkthrough](https://brainstorm.readthedocs.org/en/latest/walkthrough.html) of the ``cifar10_cnn.py`` example.

If you wish, you can also run the data preparation scripts (``data`` directory) and look at some basic examples (``examples`` directory).

Help and Support
----------------

If you have any suggestions or questions, please post to the [Google group](https://groups.google.com/forum/#!forum/mailstorm).

If you encounter any errors or problems, please let us know by opening an issue.

License
-------

MIT License. Please see the LICENSE file.

Acknowledgements
----------------

[Klaus Greff](http://people.idsia.ch/~greff/) and [Rupesh Srivastava](http://people.idsia.ch/~rupesh/) would like to thank [Jürgen Schmidhuber](http://people.idsia.ch/~juergen/) for his continuous supervision and encouragement.
Funding from EU projects NASCENCE (FP7-ICT-317662) and WAY (FP7-ICT-288551) was instrumental during the development of this project.
We also thank Nvidia Corporation for their donation of GPUs.
