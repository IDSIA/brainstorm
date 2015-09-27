#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import threading
import numpy as np


def run_network_double_buffer(net, iterator):
    def run_it(it):
        try:
            net.provide_external_data(next(it))
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


def run_network(net, iterator):
    for i, data in enumerate(iterator):
        net.provide_external_data(data)
        yield i


def _flatten_all_but_last(a):
    if a is None:
        return None
    return a.reshape(-1, a.shape[-1])


def _weighted_average(errors):
    errors = np.array(errors)
    assert errors.ndim == 2 and errors.shape[1] == 2
    return np.sum(errors[:, 1] * errors[:, 0] / np.sum(errors[:, 0]))


def gather_losses_and_scores(net, scorers, scores, out_name='',
                             targets_name='targets', mask_name=''):
    ls = net.get_loss_values()
    for name, loss in ls.items():
        scores[name].append((net._buffer_manager.batch_size, loss))

    for sc in scorers:
        name = sc.__name__
        predicted = net.get_output(sc.out_name) if sc.out_name\
            else net.get_output(out_name)
        true_labels = net.get_input(sc.targets_name) if sc.targets_name\
            else net.get_input(targets_name)
        mask = net.get_input(sc.mask_name) if sc.mask_name\
            else (net.get_input(mask_name) if mask_name else None)

        predicted = _flatten_all_but_last(predicted)
        true_labels = _flatten_all_but_last(true_labels)
        mask = _flatten_all_but_last(mask)
        weight = mask.sum() if mask else predicted.shape[0]

        scores[name].append((weight, sc(true_labels, predicted, mask)))


def aggregate_losses_and_scores(scores, net, scorers):
    results = OrderedDict()
    for name in net.get_loss_values():
        results[name] = _weighted_average(scores[name])
    for sc in scorers:
        results[sc.__name__] = sc.aggregate(scores[sc.__name__])
    return results
