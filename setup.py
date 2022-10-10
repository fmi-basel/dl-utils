from setuptools import setup, find_packages

contrib = [
    'Markus Rempfler',
    'Raphael Ortiz',
]

# setup.
setup(
    name='dl-utils',
    version='0.2.0',
    description='deep learning utilities',
    author=', '.join(contrib),
    license='MIT',
    packages=find_packages(exclude=[
        'tests',
    ]),
    install_requires=[
        'numpy<=1.19.5',  # numpy 1.2 incomptabile with tensorflow 2.3, see https://stackoverflow.com/questions/66207609/notimplementederror-cannot-convert-a-symbolic-tensor-lstm-2-strided-slice0-t
        'scipy',
        'tensorflow>=2.3',
        'scikit-image>=0.13',
        'scikit-learn',
        'future',
        'tqdm',
        'pytest',
        'pyyaml',
        'h5py',
        'tensorflow-addons'
    ],
    zip_safe=False)
