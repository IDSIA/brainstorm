#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


class Targets(object):
    """
    Baseclass for all targets objects. A targets object holds all the desired
    outputs and the corresponding mask (if any).
    """
    def __init__(self, binarize_to, mask, sequence_lengths=None):
        self.binarize_to = binarize_to
        self.mask = None
        self.data = None
        self.sequence_lengths = sequence_lengths
        if mask is not None:
            assert len(mask.shape) == 3 and mask.shape[2] == 1, \
                "Mask has to be 3D with the last dimension of size 1 " \
                "(not {})".format(mask.shape)
            self.mask = mask
            if self.sequence_lengths is None:
                self.sequence_lengths = get_sequence_lengths(mask)

    @property
    def shape(self):
        if self.binarize_to:
            feature_dim = self.binarize_to
        else:
            feature_dim = self.data.shape[2]
        if self.mask is not None:
            time_dim, sample_dim = self.mask.shape[:2]
        else:
            time_dim, sample_dim = self.data.shape[:2]
        return time_dim, sample_dim, feature_dim

    def __getitem__(self, item):
        raise NotImplementedError()

    def trim(self, new_length):
        assert new_length >= self.sequence_lengths.max()
        if self.mask is not None:
            self.mask = self.mask[:new_length]


class FramewiseTargets(Targets):
    """
    Provide a target value for every point in time. Although some timesteps
    might be masked out.
    """
    def __init__(self, targets, mask=None, binarize_to=None,
                 sequence_lengths=None):
        super(FramewiseTargets, self).__init__(binarize_to, mask,
                                               sequence_lengths)
        assert len(targets.shape) == 3, "Targets have to be 3D " \
                                  "(but was {})".format(targets.shape)
        self.data = targets
        if binarize_to:
            assert targets.shape[2] == 1, \
                "If binarizing the last dim has to be 1 (not {})" \
                "".format(targets.shape[2])
            self.data = np.array(self.data, dtype=np.int)
        if self.mask is not None:
            assert self.mask.shape[:2] == self.data.shape[:2], \
                "First two dimensions of targets and mask have to match (but "\
                "{} != {})".format(self.mask.shape[:2], self.data.shape[:2])
        else:
            self.sequence_lengths = (np.ones(self.data.shape[1]) *
                                     self.data.shape[0])

    def __getitem__(self, item):
        time_slice, sample_slice = _get_simplified_time_and_sample_slices(item)
        data = self.data[time_slice, sample_slice, :]
        mask = self.mask[time_slice, sample_slice, :] \
            if self.mask is not None else None
        seq_lens = self.sequence_lengths[sample_slice]
        if time_slice.start is not None:
            if time_slice.start < 0:
                start = self.data.shape[0] + time_slice.start
            else:
                start = time_slice.start
            seq_lens -= start
        seq_lens = np.clip(seq_lens, 0, data.shape[0])

        return FramewiseTargets(data, mask, self.binarize_to, seq_lens)

    def trim(self, new_length):
        super(FramewiseTargets, self).trim(new_length)
        self.data = self.data[:new_length, :, :]


class LabelingTargets(Targets):
    """
    Provide a list of labels for the sequence. If a mask is given, then the
    resulting deltas will be masked.
    """
    def __init__(self, labels, mask=None, binarize_to=None,
                 sequence_lengths=None):
        super(LabelingTargets, self).__init__(binarize_to, mask,
                                              sequence_lengths)
        assert isinstance(labels, list)
        self.data = labels
        if self.mask is not None:
            assert self.mask.shape[1] == len(self.data), \
                "Number of label sequences must match the number of masks "\
                "(but {} != {})".format(len(self.data), self.mask.shape[1])
        else:
            self.sequence_lengths = np.zeros(len(self.data))

    @property
    def shape(self):
        time_dim = self.mask.shape[0] if self.mask is not None else None
        feature_dim = self.binarize_to or len(self.data[0][0])
        return time_dim, len(self.data), feature_dim

    def __getitem__(self, item):
        time_slice, sample_slice = _get_simplified_time_and_sample_slices(item)
        data = self.data[sample_slice]
        mask = self.mask[time_slice, sample_slice, :] \
            if self.mask is not None else None

        if mask is not None:
            seq_lens = self.sequence_lengths[sample_slice]
            if time_slice.start is not None:
                if time_slice.start < 0:
                    start = self.mask.shape[0] + time_slice.start
                else:
                    start = time_slice.start
                seq_lens -= start
            seq_lens = np.clip(seq_lens, 0, mask.shape[0])
        else:
            seq_lens = None

        return LabelingTargets(data, mask, self.binarize_to, seq_lens)


class SequencewiseTargets(Targets):
    """
    Provide one target per sequence. If no mask is given then only the last
    timestep will receive deltas. If a mask is given, then all the masked in
    timesteps will receive deltas.
    """
    def __init__(self, targets, mask=None, binarize_to=None,
                 sequence_lengths=None):
        super(SequencewiseTargets, self).__init__(binarize_to, mask,
                                                  sequence_lengths)
        assert len(targets.shape) == 3, "Targets have to be 3D " \
                                  "(but was {})".format(targets.shape)
        assert targets.shape[0] == 1, \
            "First dimension(sequence length) should be 1 for " \
            "SequenceWiseTargets (but was {})".format(targets.shape)
        assert binarize_to is None or targets.shape[2] == 1, \
            "If binarizing the last dim has to be 1 (not {})" \
            "".format(targets.shape[2])
        self.data = targets
        if self.mask is not None:
            assert self.mask.shape[1] == self.data.shape[1], \
                "The number of targets and the number of masks have to match "\
                "(but {} != {})".format(self.data.shape[1], self.mask.shape[1])
        else:
            self.sequence_lengths = np.zeros(len(self.data))

    def __getitem__(self, item):
        time_slice, sample_slice = _get_simplified_time_and_sample_slices(item)
        data = self.data[:, sample_slice, :]
        mask = self.mask[time_slice, sample_slice, :] \
            if self.mask is not None else None

        if mask is not None:
            seq_lens = self.sequence_lengths[sample_slice]
            if time_slice.start is not None:
                if time_slice.start < 0:
                    start = self.mask.shape[0] + time_slice.start
                else:
                    start = time_slice.start
                seq_lens -= start
            seq_lens = np.clip(seq_lens, 0, mask.shape[0])
        else:
            seq_lens = None

        return SequencewiseTargets(data, mask, self.binarize_to, seq_lens)


def get_sequence_lengths(mask):
    """
    Given a mask it returns a list of the lengths of all sequences. Note: this
    assumes, that the mask has only values 0 and 1. It returns for each
    sequence the last index such that the mask is 1 there.
    :param mask: mask of 0s and 1s with shape=(t, b, 1)
    :return: array of sequence lengths with shape=(b,)
    """
    return mask.shape[0] - mask[::-1, :, 0].argmax(axis=0)


def _get_simplified_time_and_sample_slices(item):
    if isinstance(item, tuple):
        assert len(item) == 2, "Feature indexing not supported!"
        return _get_simple_slice(item[0]), _get_simple_slice(item[1])
    elif isinstance(item, int):
        return _get_simple_slice(item), slice(None)
    else:
        raise ValueError('Unsupported indexing type {}'.format(type(item)))


def _get_simple_slice(item):
    if isinstance(item, slice):
        assert item.step is None or item.step == 1
        return item
    elif isinstance(item, (int, np.int64, np.int32, np.uint32, np.uint64)):
        return slice(item, item + 1)
    else:
        raise ValueError('Unsupported indexing type {}'.format(type(item)))
