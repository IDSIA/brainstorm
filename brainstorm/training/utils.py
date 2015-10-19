#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals


def run_network(net, iterator, all_inputs=True):
    for i, data in enumerate(iterator):
        net.provide_external_data(data, all_inputs=all_inputs)
        yield i
