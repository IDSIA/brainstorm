.. :changelog:

History
-------

0.5 (2015-12-01)
++++++++++++++++
Changed Behaviour
"""""""""""""""""
* examples now run on CPU by default
* added ``brainstorm.tools.shuffle_data`` and ``brainstorm.tools.split`` to help with data preparation
* ``SigmoidCE`` and ``SquaredDifference`` layers now outputs a loss for each dimension instead of summing over features.
* ``SquaredDifference`` layer does no longer scale by one half.
* Added a ``SquaredLoss`` layer that computes half the squared difference and
  has an interface that is compatible with the ``SigmoidCE`` and ``SigmoidCE`` layers.
* Output `probabilities` renamed to `predictions` in ``SigmoidCE`` and ``SigmoidCE`` layers.

New Features
""""""""""""
* added a `use_conv` option to ``brainstorm.tools.create_net_from_spec``
* added `criterion` option to ``brainstorm.hooks.EarlyStopper`` hook
* added ``brainstorm.tools.get_network_info`` function that returns information
  about the network as a string
* added ``brainstorm.tools.extract`` function that applies a network to some
  data and saves a set of requested buffers.
* ``brainstorm.layers.mask`` layer now supports masking individual features
* added ``brainstorm.hooks.StopAfterThresholdReached`` hook

Improvements
""""""""""""
* EarlyStopper now works for any timescale and interval
* Recurrent, Lstm, Clockwork, and ClockworkLstm layers now accept inputs of
  arbitrary shape by implicitly flattening them.
* several fixes to make building the docs easier
* some performance improvements of NumpyHandler operations ``binarize_t`` and ``index_m_by_v``
* sped up tests
* several improvements to installation scripts

Bugfixes
""""""""
* fixed `sqrt` operation for ``PyCudaHandler``. This should fix problems with BatchNormalization on GPU.
* fixed a bug for task_type='regression' in ``brainstorm.tools.get_in_out_layers``
  and ``brainstorm.tools.create_net_from_spec``
* removed defunct name argument from input layer
* fixed a crash when applying ``brainstorm.hooks.SaveBestNetwork`` to `rolling_training` loss
* various minor fixes of the ``brainstorm.hooks.BokehVisualizer``
* fixed a problem with ``sum_t`` operation in ``brainstorm.handlers.PyCudaHandler``
* fixed a blocksize problem in convolutional and pooling operations in ``brainstorm.handlers.PyCudaHandler``


0.5b0 (2015-10-25)
++++++++++++++++++
* First release on PyPI.
