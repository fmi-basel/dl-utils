from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TerminateOnNaN

from dlutils.training.scheduler import CosineAnnealingSchedule

import os
from numpy import ceil
import logging


class ModelConfigSaver(Callback):
    '''saves the architecture of the model before training begins.

    '''
    def __init__(self, filename):
        '''
        '''
        self.filename = filename

    def on_train_begin(self, logs={}, **kwargs):
        '''
        '''
        logger = logging.getLogger(__name__)
        logger.debug('Writing model architecture as .yaml to {}'.format(
            self.filename))
        if not os.path.exists(os.path.dirname(self.filename)):
            dirname = os.path.dirname(self.filename)
            os.makedirs(dirname)
            logger.debug('Created necessary folders: {}'.format(dirname))
        with open(self.filename, 'w') as fout:
            fout.write(self.model.to_yaml())

    def on_train_end(self, logs={}, **kwargs):
        '''
        '''
        filename = self.filename.replace('architecture.yaml', 'latest.h5')
        self.model.save_weights(filename, overwrite=True)


def create_callbacks(outdir,
                     nth_checkpoint,
                     lr,
                     epochs,
                     lr_min=None,
                     n_restarts=None,
                     epoch_to_restart_growth=1.0,
                     restart_decay=1.0,
                     debug=False,
                     best_monitor='val_loss',
                     monitor_mode='auto',
                     csv_append=False):
    '''Add basic callbacks for training.

    - ModelConfigSaver for architecture and final weights.
    - ModelCheckpoint for every nth epoch.
    - ModelCheckpoint for best val score.
    - CSVLogger for all metrics.
    - Tensorboard for model graph.
    - Cosine annealing learning rate schedule.
    - TerminateOnNaN

    '''
    if lr_min is None:
        lr_min = 0.05 * lr
    if n_restarts is None:
        epochs_to_restart = epochs
    else:
        n_restarts_factor = sum(epoch_to_restart_growth**x
                                for x in range(n_restarts))

        epochs_to_restart = (epochs + 1) / n_restarts_factor
        if epochs_to_restart < 1:
            raise ValueError(
                'Initial epoch_to_restart ({}) < 1. Decrease n_restarts ({}) or epoch_to_restart_growth ({})'
                .format(epochs_to_restart, n_restarts,
                        epoch_to_restart_growth))

        epochs_to_restart = int(ceil(epochs_to_restart))

    callbacks = []
    callbacks.append(
        ModelConfigSaver(os.path.join(outdir, 'model_architecture.yaml'), ))
    callbacks.append(
        ModelCheckpoint(os.path.join(outdir, 'model_best.h5'),
                        save_best_only=True,
                        save_weights_only=True,
                        monitor=best_monitor,
                        mode=monitor_mode))
    if nth_checkpoint < epochs:
        callbacks.append(
            ModelCheckpoint(
                os.path.join(outdir, 'model_{epoch:04}.h5'),
                period=
                nth_checkpoint,  # TODO `period` argument is deprecated. Please use `save_freq` to specify the frequency in number of samples seen.
                save_weights_only=True))
    callbacks.append(
        CSVLogger(os.path.join(outdir, 'training.log'),
                  separator=',',
                  append=csv_append))
    callbacks.append(
        TensorBoard(os.path.join(outdir, 'tensorboard-logs'),
                    write_graph=True,
                    write_grads=False,
                    write_images=debug,
                    histogram_freq=1 if debug else 0))
    callbacks.append(
        LearningRateScheduler(
            CosineAnnealingSchedule(lr_max=lr,
                                    lr_min=lr_min,
                                    epoch_max=epochs_to_restart,
                                    epoch_max_growth=epoch_to_restart_growth,
                                    reset_decay=restart_decay)))
    callbacks.append(TerminateOnNaN())
    return callbacks
