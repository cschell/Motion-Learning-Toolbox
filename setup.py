#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="Motion Learning Toolbox",
    version="0.0.2",
    description="Motion Learning Toolbox",
    author="Christian Rack",
    author_email="mail@chrisrack.de",
    url="",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "quaternionic",
    ],
    packages=find_packages(),
)