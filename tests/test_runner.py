from dlutils.prediction.runner import runner

import numpy as np


def test_runner(n_vals=100):
    '''
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


if __name__ == '__main__':
    test_runner(20)
