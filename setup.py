from glob import glob
import os
from setuptools import setup, find_packages

contrib = [
    'Markus Rempfler',
]

# setup.
setup(
    name='dl-utils',
    version='0.1',
    description='deep learning utilities',
    author=', '.join(contrib),
    license='BSD',
    packages=find_packages(exclude=[
        'tests',
    ]),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'keras==2.1.6',
        'opencv-python>=3.4'
    ],
    # I believe zipping makes the import rather slow.
    zip_safe=False)
