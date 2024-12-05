#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='birdclef',
    version='1.0.0',
    description='Audio Classification App',
    author='ramyasri',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
)
