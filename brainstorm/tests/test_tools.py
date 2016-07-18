#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import numpy as np
import pytest

from brainstorm import Network
from brainstorm.handlers import NumpyHandler
from brainstorm.tools import extract
from brainstorm.data_iterators import Minibatches 
from brainstorm.initializers import Gaussian
from brainstorm.layers import Input,FullyConnected

def get_simple_net():
    inp = Input(out_shapes={'default': ('T', 'B', 4)})
    out = FullyConnected(2, name='Output', activation='tanh')
    simple_net = Network.from_layer(inp >> out)
    return simple_net

def get_simple_net_with_mask():
    inp = Input(out_shapes={'default': ('T', 'B', 4),'mask': ('T','B',1)})
    out = FullyConnected(2, name='Output', activation='tanh')
    simple_net = Network.from_layer(inp >> out)
    return simple_net

def test_extract():
    net = get_simple_net()
    net.initialize(Gaussian(0.1))
    batch_size = 6
    input_data = np.random.rand(10,6,4).astype(np.float32)

    # compute expected result
    layer_W = net.buffer['Output']['parameters']['W']
    layer_bias = net.buffer['Output']['parameters']['bias']

    expected_result = np.tanh(np.dot(input_data,layer_W.T) + layer_bias)

    # run extract
    data_iterator = Minibatches(batch_size,default=input_data)

    extracted_data = extract(net,data_iterator,['Output.outputs.default'])

    assert expected_result.shape == extracted_data['Output.outputs.default'].shape
    assert np.allclose(expected_result,extracted_data['Output.outputs.default'])

def test_extract_with_mask():
    net = get_simple_net_with_mask()
#     pytest.set_trace()  
    net.initialize(Gaussian(0.1))
    batch_size = 6
    input_data = np.random.rand(10,6,4).astype(np.float32)
    # set some mask
    input_mask = np.zeros((10,6,1),dtype=np.float32)
    input_mask[0:5,0,0] = 1
    input_mask[0:3,0,0] = 1
    input_mask[0:8,0,0] = 1
    input_mask[0:7,0,0] = 1
    input_mask[0:2,0,0] = 1
    input_mask[0:3,0,0] = 1

    # compute expected result WITHOUT mask
    layer_W = net.buffer['Output']['parameters']['W']
    layer_bias = net.buffer['Output']['parameters']['bias']

    expected_result = np.tanh(np.dot(input_data,layer_W.T) + layer_bias)

    # run extract
    data_iterator = Minibatches(batch_size,default=input_data,mask=input_mask)

    extracted_data = extract(net,data_iterator,['Output.outputs.default'])

#     pytest.set_trace()  
    assert expected_result.shape == extracted_data['Output.outputs.default'].shape
    # where the mask is 0, we don't care for the result
    assert np.allclose(expected_result[input_mask.astype(bool)],extracted_data['Output.outputs.default'][input_mask.astype(bool)])




