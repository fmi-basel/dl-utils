from sklearn.model_selection import train_test_split


def split(samples, ratio, stratify=None, seed=13):
    '''split images into training and validation.

    Parameters
    ----------
    samples : list of samples
        samples to be split for training and validation.
    ratio : float, [0, 1]
        relative validation split size.
    stratify : list of categorical
        categories to be used to stratify the split.
    seed : int, optional
        set seed to ensure a reproducible split.

    Returns
    -------
    training_samples, test_samples : list of samples
        samples split for training and validation.

    '''
    if not 0 <= ratio <= 1.:
        raise ValueError('Split ratio needs to be within [0, 1].')

    return train_test_split(
        samples, test_size=ratio, stratify=stratify, random_state=seed)

