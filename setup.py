import os
from setuptools import setup, find_packages

setup(
    name="acres",
    version="0.1.0",
    author="Václav Volhejn",
    description=("ML-based barcode sharpening."),
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "tensorflow == 1.5.0"
    ]
)
