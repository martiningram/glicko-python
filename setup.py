from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name="glicko-py",
    version=getenv("VERSION", "LOCAL"),
    description="Glicko written in python",
    packages=find_packages(),
)
