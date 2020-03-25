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
        'numpy', 'matplotlib', 'scipy', 'opencv-python>=3.4', 'tensorflow>=2',
        'scikit-image>=0.13', 'scikit-learn', 'future', 'tqdm', 'pytest',
        'pyyaml', 'h5py', 'tensorflow-addons>=0.6.0', 'pandas>=0.25.3'
    ],
    # I believe zipping makes the import rather slow.
    zip_safe=False)
