def set_seeds(seed_val=42):
    '''fix seeds for reproducibility.

    '''
    from numpy.random import seed
    seed(seed_val)
    from tensorflow import set_random_seed
    set_random_seed(seed_val)


def get_zero_based_task_id(default_return=None):
    '''fetches the environment variable for this process' task id.

    Returns None if process is not run in an SGE environment.

    '''
    import os
    sge_id = os.environ.get('SGE_TASK_ID', None)
    if sge_id is None:
        return default_return

    return int(sge_id) - 1
