'''Image data augmentation in tensorflow.

'''
import tensorflow as tf


def random_axis_flip(axis, flip_prob):
    '''reverses axis with probability threshold

    Parameters
    ----------
    axis : int
        axis index to be flipped.
    flip_prob : float
        probability of flipping.

    Returns
    -------
    flipper : func
        transformation function.

    '''

    def _flipper(input_dict):
        '''expects an input dictionary.
        '''
        draw_prob = tf.random.uniform(
            shape=[], minval=0, maxval=1, dtype=tf.float32)

        # NOTE the cell-var-from-loop warning is disabled as the lambdas
        # are executed immediately by tf.cond and thus, evaluation happens
        # when val is still the *current* val.
        return {
            key: tf.cond(
                draw_prob <= flip_prob,
                lambda: tf.reverse(val, [axis]),  # pylint: disable = W0640
                lambda: val)  # pylint: disable = W0640
            for key, val in input_dict.items()
        }

    return _flipper


def random_gaussian_noise(noise_mu, noise_sigma, keys):
    '''adds pixel-wise IID gaussian noise to the given tensors.

    The noise distributions follows N(0, sigma) where

       sigma~N(noise_mu, noise_sigma).

    sigma is drawn for each batch independently.
    If sigma <= 0, then no noise is added to the current run.


    Parameters
    ----------
    noise_mu : float
        mean of the distribution from which sigma is drawn.
    noise_sigma : float
        standard deviation of the distribution from which sigma is drawn.
    keys : list
        list of keys indicating to which entries in the input_dict the
        noise shall be added.

    Returns
    -------
    distorter : func
        transformation function.

    '''
    if not keys:
        raise ValueError(
            'keys cant be {}! Expected a non-empty list of dict keys.'.format(
                keys))

    def _distorter(input_dict):
        '''adds gaussian noise to the entries in input_dict that
        are indexed by keys.

        '''
        sigma = tf.maximum(
            0., tf.random.normal(shape=[], mean=noise_mu, stddev=noise_sigma))

        output_dict = {key: val for key, val in input_dict.items()}
        for key in keys:
            noise = tf.random.normal(
                shape=tf.shape(output_dict[key]), mean=0, stddev=sigma)
            output_dict[key] = output_dict[key] + noise

        return output_dict

    return _distorter


def random_gaussian_offset(offset_sigma, keys):
    '''draws a random offset from N(0, offset_sigma) and
    adds it to the given input[key].

     Parameters
    ----------
    offset_sigma : float
        standard deviation of the distribution from which sigma is drawn.
    keys : list
        list of keys indicating to which entries in the input_dict the
        noise shall be added.

    Returns
    -------
    distorter : func
        transformation function.

    '''

    def _distorter(input_dict):
        '''adds offset to the entries in input_dict that
        are indexed by keys.

        '''
        output_dict = {key: val for key, val in input_dict.items()}
        for key in keys:
            image = output_dict[key]
            offset = tf.random.normal(shape=[], mean=0, stddev=offset_sigma)
            output_dict[key] = image + offset
        return output_dict

    return _distorter
