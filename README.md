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
