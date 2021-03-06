from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range

from tensorflow.keras.callbacks import Callback

from scipy.ndimage.filters import gaussian_filter1d

from numpy import logspace, log10

import numpy as np
import tensorflow.keras.backend as K


class LRFinderCallback(Callback):
    def __init__(self, base_lr=1e-5, max_lr=10.0):
        '''
        This callback is intended to explore well-suited learning
        rates as suggested in

        L.N. Smith, Cyclical Learning Rates for Training Neural Networks,
        WACV 2017

        Parameters
        ----------
        base_lr : float
            minimum learning rate to start with.
        max_lr : float
            maximum learning rate.

        Notes
        -----
        This scheduler will increase the learning rate
        exponentially in the range [base_lr. max_lr] for each batch.
        It restarts at base_lr every epoch. Beware that optimizers with
        momentum would need a restart between each epoch for a reliable
        estimate.

        '''
        super(LRFinderCallback, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr

    def on_train_begin(self, logs=None):
        self.losses = []
        self.idx = 0
        self.lr = logspace(log10(self.base_lr), log10(self.max_lr),
                           self.params['steps'])
        K.set_value(self.model.optimizer.lr, self.current_lr)

    def on_train_end(self, logs=None):
        estimate = self.suggest_lr(sigma=3)
        print('Suggested learning rate: {:1.3e}'.format(estimate))

    @property
    def current_lr(self):
        return self.lr[self.idx]

    def update(self):
        self.idx = (self.idx + 1) % len(self.lr)
        K.set_value(self.model.optimizer.lr, self.current_lr)

    def on_epoch_begin(self, epoch, logs=None):
        '''add new row to losses and reset lr.
        '''
        self.losses.append([])
        self.idx = 0

    def on_batch_end(self, epoch, logs=None):
        # keep track of learning rate and loss
        self.losses[-1].append(logs['loss'])

        # and update learning rate.
        self.update()

    def suggest_lr(self, **kwargs):
        return suggest_lr(self.lr, np.asarray(self.losses).flatten(), **kwargs)


def suggest_lr(learning_rates, losses, sigma):
    '''suggests optimal learning rate from lr-sweep data.

    Parameters
    ----------
    learning_rates : ndarray or list of shape (n,)
        learning rates corresponding to losses.
    losses : ndarray of shape (n, )
        losses acquired over learning rate sweep.
    sigma : float
        smoothing sigma for derivative.

    '''
    assert losses.ndim == 1, \
        'only one-dimensional loss arrays are supported'

    assert len(losses) == len(learning_rates), \
        'losses and learning_rates need to have the same shape'

    diff_loss = gaussian_filter1d(losses,
                                  sigma=sigma,
                                  axis=0,
                                  order=1,
                                  mode='nearest')
    smooth_loss = gaussian_filter1d(losses,
                                    sigma=sigma,
                                    axis=0,
                                    mode='nearest')

    end = np.argmin(smooth_loss, axis=0) + 1
    if (end >= len(diff_loss)):
        end = len(diff_loss) - 1
    idx = np.argmin(diff_loss[:end], axis=0)
    return learning_rates[idx]


def lr_finder(model,
              dataset_generator,
              steps,
              base_lr,
              max_lr,
              reps=1,
              verbose=0,
              **kwargs):
    '''functional interface to LRFinderCallback.

    Returns
    -------
    lrf : LRFinderCallback
        callback object with lr/loss history.

    '''
    cb = LRFinderCallback(base_lr, max_lr)

    if reps > 1:
        reps = 1
        import warnings
        warnings.warn('reps > 1 in lr_finder is no longer supported.')

    model.fit_generator(dataset_generator,
                        steps_per_epoch=steps,
                        epochs=reps,
                        callbacks=[cb],
                        verbose=verbose,
                        **kwargs)

    return cb


if __name__ == '__main__':
    pass
