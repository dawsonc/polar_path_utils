#!/usr/bin/env python

from setuptools import setup, find_packages

requirements = [
    "numpy",
    "matplotlib",
]

dev_requirements = [
    "black",
    "mypy",
    "pytest",
    "flake8",
]

setup(
    name="polar_path_utils",
    version="0.0.0",
    description="Path-planning utilities for polar coordinates",
    author="Charles Dawson",
    author_email="charles.dwsn@gmail.com",
    url="https://github.com/dawsonc/polar_path_utils",
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    packages=find_packages(),
)
