#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import numpy as np
import pytest

from brainstorm import Network
from brainstorm.data_iterators import Undivided
from brainstorm.initializers import Gaussian
from brainstorm.layers import SoftmaxCE, Input, Lstm, Recurrent, FullyConnected
from brainstorm.training.utils import run_network

from brainstorm.tests.helpers import HANDLER

np.random.seed(1234)

NO_CON = set()


def simple_recurrent_net():
    inp = Input(out_shapes={'default': ('T', 'B', 2)})
    net = Network.from_layer(inp >> Recurrent(3, name='out'))
    return net


def lstm_net():
    inp = Input(out_shapes={'default': ('T', 'B', 2)})
    net = Network.from_layer(inp >> Lstm(3, name='out'))
    return net


layers_to_test_with_context = [
    simple_recurrent_net,
    lstm_net
]

ids = [f.__name__ for f in layers_to_test_with_context]


@pytest.fixture(params=layers_to_test_with_context, ids=ids)
def net_with_context(request):
    net = request.param()
    return net


def test_context_slice_allows_continuing_forward_pass(net_with_context):
    net = net_with_context
    net.set_handler(HANDLER)
    net.initialize(Gaussian(0.1), seed=1234)
    all_data = np.random.randn(4, 1, 2)

    # First do a pass on all the data
    net.provide_external_data({'default': all_data})
    net.forward_pass()
    final_context = [HANDLER.get_numpy_copy(x) if x is not None else None
                     for x in net.get_context()]
    final_outputs = HANDLER.get_numpy_copy(net.buffer.out.outputs.default)

    # Pass only part of data
    data_a = all_data[:2]
    net.provide_external_data({'default': data_a})
    net.forward_pass()

    # Pass rest of data with context
    data_b = all_data[2:]
    net.provide_external_data({'default': data_b})
    net.forward_pass(context=net.get_context())
    context = [HANDLER.get_numpy_copy(x) if x is not None else None
               for x in net.get_context()]
    outputs = HANDLER.get_numpy_copy(net.buffer.out.outputs.default)

    # Check if outputs are the same as final_outputs
    success = np.allclose(outputs[:-1], final_outputs[2:-1])
    if not success:
        print("Outputs:\n", outputs[:-1])
        print("Should match:\n", final_outputs[2:-1])
        raise AssertionError("Outputs did not match.")

    # Check if context is same as final_context
    assert len(context) == len(final_context), "Context list sizes mismatch!"

    for (x, y) in zip(context, final_context):
        if x is None:
            assert y is None
        else:
            # print("Context:\n", x)
            # print("Should match:\n", y)
            assert np.allclose(x, y)


inp = Input(out_shapes={'default': ('T', 'B', 4),
                        'targets': ('T', 'B', 1)})
hid = FullyConnected(2, name="Hid")
out = SoftmaxCE(name='Output')
(inp - 'targets' >> 'targets' - out)
simple_net = Network.from_layer(inp >> hid >> out)


def test_forward_pass_with_missing_data():
    it = Undivided(default=np.random.randn(3, 2, 4))(simple_net.handler)

    with pytest.raises(KeyError):
        for _ in run_network(simple_net, it):
            pass

    for _ in run_network(simple_net, it, all_inputs=False):
        pass
