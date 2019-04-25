#!/usr/bin/env python

from setuptools import setup, Extension, find_packages
import numpy

log2_c_module = Extension(
    "log2/_core",
    sources=["log2/log2module.c"],
    include_dirs=[numpy.get_include()],
    define_macros=[],
)

exec(open('log2/version.py').read())

setup(name='log2',
    packages=find_packages(),
    version=__version__,
    description='Several functions for computing the log base 2 of a unsigned 32-bit length word',
    author='Matthieu Baumann',
    author_email='matthieu.baumann@astro.unistra.fr',
    license='BSD',
    url='https://github.com/bmatthieu3/log2',
    ext_modules=[log2_c_module],
    install_requires=[],
    provides=['log2'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)