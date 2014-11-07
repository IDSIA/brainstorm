#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from copy import deepcopy, copy
import numpy as np
from brainstorm.describable import Describable

SEED_RANGE = (0, 1000000000)


def get_subseed_for_item(base_seed, item):
    hash_string = str(base_seed) + '$' + str(item)
    return hash(hash_string) % (SEED_RANGE[1] - SEED_RANGE[0]) + SEED_RANGE[0]


class HierarchicalRandomState(np.random.RandomState):
    """
    An extension of the numpy RandomState that saves it's own seed and allows to
    create sub-RandomStates like this:
    >> sub_rnd = rnd['sub1']

    The sub-RandomStates depend on the seed of their creator, but are otherwise
    as independent as possible. That means that the seed of the sub-RandomState
    is only dependent on the seed of the creator and on it's name, not on the
    random state of the creator.
    """
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(*SEED_RANGE)
        super(HierarchicalRandomState, self).__init__(seed)
        self._seed = seed
        self.categories = dict()

    def seed(self, seed=None):
        """
        This method is kept for compatibility with the numpy RandomState. But
        for better readability you are encouraged to use the set_seed() method
        instead.

        :param seed: the seed to reseed this random state with.
        :type seed: int
        """
        super(HierarchicalRandomState, self).seed(seed)
        self._seed = seed
        self.categories = dict()

    def get_seed(self):
        return self._seed

    def set_seed(self, seed):
        self.seed(seed)

    def generate_seed(self, seed_range=None):
        if seed_range is None:
            seed_range = SEED_RANGE
        return self.randint(*seed_range)

    def get_new_random_state(self, seed=None):
        if seed is None:
            seed = self.generate_seed()

        return HierarchicalRandomState(seed)

    def __getitem__(self, item):
        if item not in self.categories:
            seed = get_subseed_for_item(self._seed, item)
            self.categories[item] = HierarchicalRandomState(seed)
        return self.categories[item]


class Seedable(Describable):
    """
    Baseclass for all objects that use randomness. It helps to make sure all the
    results are reproducible.
    It offers a self.rnd which is a HierarchicalRandomState.
    """
    __undescribed__ = {'rnd', 'seed'}

    def __init__(self, seed=None, category=None):
        if category is None:
            self.rnd = HierarchicalRandomState(seed)
        else:
            self.rnd = global_rnd[category].get_new_random_state(seed)
        self.seed = self.rnd.get_seed()

    def set_seed(self, seed):
        self.rnd.set_seed(seed)
        self.seed = seed

    def __init_from_description__(self, description):
        Seedable.__init__(self)


def reseeding_deepcopy(values, seed):
    r = deepcopy(values)
    if isinstance(r, (HierarchicalRandomState, Seedable)):
        r.set_seed(seed)
    return r


def reseeding_copy(values, seed):
    r = copy(values)
    if isinstance(r, (HierarchicalRandomState, Seedable)):
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
global_rnd = HierarchicalRandomState(np.random.randint(*SEED_RANGE))

set_global_seed = global_rnd.set_seed
