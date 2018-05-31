from keras.callbacks import ModelCheckpoint, \
        TensorBoard, CSVLogger, LearningRateScheduler
from dlutils.training.scheduler import CosineAnnealingSchedule

import os
from numpy import ceil


def create_callbacks(outdir,
                     nth_checkpoint,
                     lr,
                     epochs,
                     lr_min=None,
                     n_restarts=None,
                     restart_decay=None):
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
        epochs_to_restart = int(ceil(epochs + 1 / n_restarts))

    return [
        ModelCheckpoint(os.path.join(outdir, 'model_latest.h5'), period=1),
        ModelCheckpoint(
            os.path.join(outdir, 'model_{epoch:04}.h5'),
            period=nth_checkpoint),
        CSVLogger(
            os.path.join(outdir, 'training.log'), separator=',', append=False),
        TensorBoard(
            os.path.join(outdir, 'tensorboard-logs'),
            write_graph=True,
            write_grads=False,
            write_images=False,
            histogram_freq=0),
        LearningRateScheduler(
            CosineAnnealingSchedule(
                lr_max=lr,
                lr_min=lr_min,
                epoch_max=epochs_to_restart,
                reset_decay=restart_decay))
    ]
