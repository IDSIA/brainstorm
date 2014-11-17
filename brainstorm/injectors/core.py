#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.describable import Describable
from brainstorm.targets import (FramewiseTargets, LabelingTargets,
                                SequencewiseTargets)


class Injector(Describable):
    __default_values__ = {'target_from': 'default_target'}

    def __init__(self, layer, target_from='layer_default'):
        self.layer = layer
        self.target_from = target_from

    @staticmethod
    def _framewise(outputs, targets):
        raise NotImplementedError()

    @staticmethod
    def _framewise_binarizing(outputs, targets):
        raise NotImplementedError()

    @staticmethod
    def _labelling(outputs, targets):
        raise NotImplementedError()

    @staticmethod
    def _labelling_binarizing(outputs, targets):
        raise NotImplementedError()

    @staticmethod
    def _sequencewise(outputs, targets):
        raise NotImplementedError()

    @staticmethod
    def _sequencewise_binarizing(outputs, targets):
        raise NotImplementedError()

    # noinspection PyCallingNonCallable
    def __call__(self, output, target):
        implementations = {
            (FramewiseTargets, False): self._framewise,
            (FramewiseTargets, True): self._framewise_binarizing,
            (LabelingTargets, False): self._labelling,
            (LabelingTargets, True): self._labelling_binarizing,
            (SequencewiseTargets, False): self._framewise,
            (SequencewiseTargets, True): self._sequencewise_binarizing
        }
        binarizing = target.binarize_to is not None
        return implementations[(target.__class__, binarizing)](output, target)
