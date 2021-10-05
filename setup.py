#!/usr/bin/env python3
from setuptools import setup, find_packages
import versioneer

setup(
    version=versioneer.get_version(),
    packages=find_packages(),
    cmdclass=versioneer.get_cmdclass(),
)
