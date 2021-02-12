from setuptools import setup, find_packages

contrib = [
    'Markus Rempfler',
    'Raphael Ortiz',
]

# setup.
setup(name='dl-utils',
      version='0.2.0',
      description='deep learning utilities',
      author=', '.join(contrib),
      license='MIT',
      packages=find_packages(exclude=[
          'tests',
      ]),
      install_requires=[
          'numpy', 'scipy', 'tensorflow>=2.3,<2.4', 'scikit-image>=0.13',
          'scikit-learn', 'future', 'tqdm', 'pytest', 'pyyaml', 'h5py',
          'tensorflow-addons==0.11.2'
      ],
      zip_safe=False)
