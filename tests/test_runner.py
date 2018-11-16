from dlutils.prediction.runner import runner

import numpy as np


def test_runner(n_vals=100):
    '''test runner with functions
    '''

    def generator_fn(val):
        return val % 2, val

    def processor_fn(key, val):
        return key, val * 2

    container = {0: [], 1: []}

    def output_fn(key, val):
        container[key].append(val)

    runner(generator_fn, processor_fn, output_fn, range(n_vals))

    for key, vals in container.items():
        assert len(vals) == n_vals / 2

        if key == 0:
            np.testing.assert_array_equal(vals, np.arange(n_vals)[::2] * 2)
        elif key == 1:
            np.testing.assert_array_equal(vals, np.arange(n_vals)[1::2] * 2)
        else:
            assert False


def test_runner_from_generator(k_runs=5, n_vals=10):
    '''test runner with generator functions
    '''

    def generator_fn(val):
        for k in range(k_runs):
            yield val % 2, val

    def processor_fn(key, val):
        for k in range(k_runs):
            yield key, val * 2

    container = {0: [], 1: []}

    def output_fn(key, val):
        container[key].append(val)

    runner(generator_fn, processor_fn, output_fn, range(n_vals))

    for key, vals in container.items():
        assert len(vals) == n_vals / 2 * k_runs**2

        if key == 0:
            np.testing.assert_array_equal(
                np.unique(vals),
                np.arange(n_vals)[::2] * 2)
        elif key == 1:
            np.testing.assert_array_equal(
                np.unique(vals),
                np.arange(n_vals)[1::2] * 2)
        else:
            assert False


if __name__ == '__main__':
    test_runner(20)
