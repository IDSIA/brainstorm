#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import threading


def run_network_double_buffer(net, iterator, all_inputs=True):
    def run_it(it):
        try:
            net.provide_external_data(next(it), all_inputs=all_inputs)
        except StopIteration:
            run_it.stop = True

    run_it.stop = False

    run_it(iterator)
    i = 0
    while not run_it.stop:
        t = threading.Thread(target=run_it, args=(iterator,))
        t.start()
        yield i
        t.join()
        i += 1


def run_network(net, iterator, all_inputs=True):
    for i, data in enumerate(iterator):
        net.provide_external_data(data, all_inputs=all_inputs)
        yield i
