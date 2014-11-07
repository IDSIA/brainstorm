#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from copy import deepcopy, copy
import numpy as np
from brainstorm.describable import Describable


class RandomState(np.random.RandomState):
    """
    An extension of the numpy RandomState that saves it's own seed
    and offers convenience methods to generate seeds and other RandomStates.
    """

    seed_range = (0, 1000000000)

    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(*RandomState.seed_range)
        super(RandomState, self).__init__(seed)
        self._seed = seed

    def seed(self, seed=None):
        """
        Set the seed of this RandomState.
        This method is kept for compatibility with the numpy RandomState. But
        for better readability you are encouraged to use the set_seed() method
        instead.

        :param seed: the seed to reseed this random state with.
        :type seed: int
        """
        super(RandomState, self).seed(seed)
        self._seed = seed

    def get_seed(self):
        """
        Return the seed of this RandomState.
        """
        return self._seed

    def set_seed(self, seed):
        """
        Set the seed of this RandomState.
        """
        self.seed(seed)

    def generate_seed(self):
        return self.randint(RandomState.seed_range)

    def get_new_random_state(self, seed=None):
        if seed is None:
            seed = self.generate_seed()
        return RandomState(seed)


class Seedable(Describable):
    """
    Baseclass for all objects that use randomness. It helps to make sure all the
    results are reproducible.
    It offers a self.rnd which is a HierarchicalRandomState.
    """
    __undescribed__ = {'rnd', 'seed'}

    def __init__(self, seed=None):
        self.rnd = RandomState(seed)
        self.seed = self.rnd.get_seed()

    def set_seed(self, seed):
        self.rnd.set_seed(seed)
        self.seed = seed

    def __init_from_description__(self, description):
        Seedable.__init__(self)


def reseeding_deepcopy(values, seed):
    r = deepcopy(values)
    if isinstance(r, (RandomState, Seedable)):
        r.set_seed(seed)
    return r


def reseeding_copy(values, seed):
    r = copy(values)
    if isinstance(r, (RandomState, Seedable)):
        r.set_seed(seed)
    return r

### used categories:
# - preprocessing
# - datasets
# - network
#   * initialize
#   * set_constraints
#   * set_regularizers
# - trainer
# - data_iterator
global_rnd = RandomState(np.random.randint(*SEED_RANGE))

set_global_seed = global_rnd.set_seed
