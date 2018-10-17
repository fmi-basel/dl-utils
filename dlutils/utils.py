def set_seeds(seed_val=42):
    '''fix seeds for reproducibility.

    '''
    from numpy.random import seed
    seed(seed_val)
    from tensorflow import set_random_seed
    set_random_seed(seed_val)
