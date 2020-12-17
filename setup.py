from setuptools import setup, find_packages

contrib = [
    'Markus Rempfler',
    'Raphael Ortiz',
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
        'numpy', 'matplotlib', 'scipy', 'opencv-python>=3.4',
        'tensorflow>=2.3,<2.4', 'scikit-image>=0.13', 'scikit-learn', 'future',
        'tqdm', 'pytest', 'pyyaml', 'h5py', 'tensorflow-addons>=0.11.2',
        'numba'
    ],
    # I believe zipping makes the import rather slow.
    zip_safe=False)
