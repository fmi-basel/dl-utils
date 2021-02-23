[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4543782.svg)](https://doi.org/10.5281/zenodo.4543782)


# dl-utils

a random collection of utilities for using tensorflow/keras.


Dependencies can be installed with

```
pip install --user -e dl-utils/
```

## Testing

Running pytest with the default options will skip tests marked as "slow" such as extensive parametric model testing or or tests involving training loop. To include them, use the runslow option:

```
pytest --runslow
```

## Citation

```bibtex
@software{dlutils,
  author       = {Rempfler, Markus and
                  Ortiz, Raphael},
  title        = {dl-utils},
  month        = feb,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {0.2.0},
  doi          = {10.5281/zenodo.4543782},
  url          = {https://doi.org/10.5281/zenodo.4543782}
}
```
