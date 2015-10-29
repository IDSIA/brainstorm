Examples
========

These examples have been included to quickly present some basic features of Brainstorm.
Prior to running an example, you should prepare the required dataset using the corresponding data preparation script in the ``data`` directory.
For each example, you can uncomment one line (indicated in the code) to make the training faster using your GPU. 

mnist_pi
--------

This example trains a simple neural network with 2 fully connected hidden layers and dropout on the MNIST dataset.


cifar10_cnn
-----------

This example trains a simple convolutional neural network on the CIFAR-10 dataset. 
Initialization is one of the most crucial aspects of training neural networks. This example shows how Brainstorm let us flexibly specify the initialization for different layers of the network.


hutter_lstm
-----------

This example trains a simple character-level LSTM recurrent neural network on the Hutter Prize dataset. It also shows a simple use-case of the ``create_net_from_spec`` tool which is useful for quickly building networks.


custom_layer
------------

This example shows how one can write a self-contained custom layer in a single file using Brainstorm.
It also shows how Brainstorm tests can be run on the layer to make sure that the implementation is correct.
Finally, it uses the newly created custom layer in a simple example like any other layer.
