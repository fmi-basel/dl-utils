from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.callbacks import ModelCheckpoint, \
        TensorBoard, CSVLogger, LearningRateScheduler, \
        Callback
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

    def on_train_begin(self, **kwargs):
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


def create_callbacks(outdir,
                     nth_checkpoint,
                     lr,
                     epochs,
                     lr_min=None,
                     n_restarts=None,
                     restart_decay=1.0):
    '''Add basic callbacks for training.

    - ModelCheckpoint for latest and every nth epoch.
    - CSVLogger for all metrics.
    - Tensorboard for model graph.
    - Cosine annealing learning rate schedule

    '''
    if lr_min is None:
        lr_min = 0.05 * lr
    if n_restarts is None:
        epochs_to_restart = epochs
    else:
        epochs_to_restart = int(ceil((epochs + 1) / n_restarts))

    callbacks = []
    callbacks.append(
        ModelConfigSaver(os.path.join(outdir, 'model_architecture.yaml')))
    callbacks.append(
        ModelCheckpoint(
            os.path.join(outdir, 'model_latest.h5'),
            period=1,
            save_weights_only=True))
    if nth_checkpoint < epochs:
        callbacks.append(
            ModelCheckpoint(
                os.path.join(outdir, 'model_{epoch:04}.h5'),
                period=nth_checkpoint,
                save_weights_only=True))
    callbacks.append(
        CSVLogger(
            os.path.join(outdir, 'training.log'), separator=',', append=False))
    callbacks.append(
        TensorBoard(
            os.path.join(outdir, 'tensorboard-logs'),
            write_graph=True,
            write_grads=False,
            write_images=False,
            histogram_freq=0))
    callbacks.append(
        LearningRateScheduler(
            CosineAnnealingSchedule(
                lr_max=lr,
                lr_min=lr_min,
                epoch_max=epochs_to_restart,
                reset_decay=restart_decay)))
    return callbacks
