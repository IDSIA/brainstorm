#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import re

PYTHON_IDENTIFIER = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")


def is_valid_python_identifier(identifier):
    return PYTHON_IDENTIFIER.match(identifier) is not None