#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function
import numpy as np
import nervanagpu
from nervanagpu import NervanaGPU
from brainstorm.handlers.base_handler import Handler
from brainstorm.randomness import global_rnd

class NervanaGPUHandler(Handler):
    __undescribed__ = {'context', 'dtype', 'EMPTY', 'rnd'}

    def __init__(self, seed=None):
        self.dtype = np.float32
        self.context = NervanaGPU(default_dtype=np.float32)
        self.EMPTY = self.context.empty((), dtype=self.dtype)
        if seed is None:
            seed = global_rnd.generate_seed()

        self.rnd = None

    array_type = nervanagpu.nervanagpu.GPUTensor

    def __init_from_description__(self, description):
        self.__init__()

    # ------------------------- Allocate new memory ------------------------- #

    def allocate(self, size):
        return self.context.empty(size, dtype=self.dtype)

    def ones(self, shape):
        a = self.context.ones(shape=shape, dtype=self.dtype)

    def zeros(self, shape):
        return self.context.zeros(shape=shape, dtype=self.dtype)

    # ---------------------------- Copy and Fill ---------------------------- #

    def copy_to(self, dest, src):
        """Copy data from src to dest (both must be GPUTensors)."""
        dest.fill(src)

    def create_from_numpy(self, arr):
        return self.context.array(arr)

    def fill(self, mem, val):
        mem.fill(val)

    def get_numpy_copy(self, mem):
        assert type(mem) == self.array_type
        return mem.get()

    def set_from_numpy(self, mem, arr):
        assert mem.shape == arr.shape, "Shape of destination ({}) != Shape " \
                                       "of source ({})".format(mem.shape,
                                                               arr.shape)
        mem.set(arr.astype(self.dtype))

    # ---------------------------- Debug helpers ---------------------------- #

    def is_fully_finite(self, a):
        return np.all(np.isfinite(a.get()))  # Slow version

    # ----------------------- Mathematical operations ----------------------- #


