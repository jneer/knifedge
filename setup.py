import os
from setuptools import setup


def readme():
    with open('README.md', 'r') as f:
        return f.read()

setup(
    name="knifedge",
    version="0.0.1",
    author="Jonas S. Neergaard-Nielsen",
    author_email="j@neer.dk",
    description="Tools for laser beam profiling",
    license="MIT",
    keywords="example documentation tutorial",
    url="https://github.com/jneer/knifedge",
    packages=['knifedge'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'attrs',
        'pyyaml'
    ],
    long_description=readme(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
)