==========
Brainstorm
==========

Brainstorm makes working with neural networks fast, flexible and fun.

Combining lessons from previous projects with new design elements, and written entirely in Python, Brainstorm has been designed to work on multiple platforms with multiple computing backends.

If you wish, you can dive into data preparation (``data`` directory) and look at some basic examples (``examples`` directory).
Documentation is in progress, and is available [here](https://brainstorm.readthedocs.org/en/latest/).

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
sudo apt-get install python-dev libhdf5-dev
# Get brainstorm
git clone git@github.com:IDSIA/brainstorm.git
# Install
cd brainstorm
pip install -r requirements.txt
python setup.py install
# Build local documentation (optional)
make docs
```
To use your CUDA installation with brainstorm:
```bash
$ pip install -r pycuda_requirements.txt
```
Set location for storing datasets:
```bash
echo "export BRAINSTORM_DATA_DIR=/home/my_data_dir/" >> ~/.bashrc
```

Help and Support
----------------

If you have any suggestions or questions, please post to the [google group](https://groups.google.com/forum/#!forum/mailstorm).

If you encounter any errors or problems, please let us know by opening an issue.

License
-------

MIT License. Please see the LICENSE file.

Acknowledgements
----------------

Klaus Greff and Rupesh Srivastava would like to thank Jürgen Schmidhuber for his continuous supervision and encouragement.
Funding from EU projects NASCENCE (FP7-ICT-317662) and WAY (FP7-ICT-288551) was instrumental during the development of this project.
We also thank Nvidia Corporation for their donation of GPUs.
