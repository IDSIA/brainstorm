#!/usr/bin/env python
import os
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.test import test as TestCommand
from distutils.errors import CompileError
from warnings import warn

# Check if we are going to use Cython to compile the pyx extension files
try:
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

# Try to get __about__ information
try:
    from brainstorm import __about__
    about = __about__.__dict__
except ImportError:
    # installing - dependencies are not there yet
    ext_modules = []
    # Manually extract the __about__
    about = dict()
    exec(open("brainstorm/__about__.py").read(), about)


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


# Add includes for building extensions
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy as np
        self.include_dirs.append(np.get_include())

    def run(self):
        try:
            _build_ext.run(self)
        except CompileError:
            warn('Failed to build optional extension modules')

# Cythonize pyx if possible, else compile C
if use_cython:
    from Cython.Build import cythonize
    extensions = cythonize([Extension('brainstorm.handlers._cpuop',
                                      ['brainstorm/handlers/_cpuop.pyx'])])

else:
    extensions = [
        Extension(
            'brainstorm.handlers._cpuop', ['brainstorm/handlers/_cpuop.c'],
            extra_compile_args=['-w', '-Ofast']),
    ]


# Setup testing
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


doclink = """
Documentation
-------------

The full documentation is at http://brainstorm.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
live_viz = ['bokeh']
draw_net = ['pygraphviz']
tests = ['pytest', 'mock']
pycuda = ['pycuda>=2015.1.3', 'scikit-cuda>=0.5.1']
all_deps = live_viz + draw_net + tests + pycuda

setup(
    name='brainstorm',
    version=about['__version__'],
    description='Fast, flexible and fun neural networks.',
    long_description=doclink + '\n\n' + history,
    author=about['__author__'],
    author_email="mailstorm@googlemail.com",
    url=about['__url__'],
    packages=['brainstorm',
              'brainstorm.structure',
              'brainstorm.layers',
              'brainstorm.training',
              'brainstorm.handlers'],
    setup_requires=['cython', 'numpy>=1.8'],
    install_requires=['cython', 'h5py', 'mock', 'numpy>=1.8', 'six'],
    extras_require={
        'live_viz':  live_viz,
        'draw_net': draw_net,
        'test': tests,
        'pycuda': pycuda,
        'all': all_deps
    },
    tests_require=tests,
    cmdclass={'test': PyTest, 'build_ext': build_ext},
    license=about['__license__'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
    ],

    ext_modules=extensions,
)
