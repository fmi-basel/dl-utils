from keras.callbacks import ModelCheckpoint, \
        TensorBoard, CSVLogger, LearningRateScheduler
from dlutils.training.scheduler import CosineAnnealingSchedule

import os

def create_callbacks(outdir, nth_checkpoint, lr, epochs):
    '''Add basic callbacks for training.

    - ModelCheckpoint for latest and every nth epoch.
    - CSVLogger for all metrics.
    - Tensorboard for model graph.
    - Cosine annealing learning rate schedule

    '''

    return [
        ModelCheckpoint(os.path.join(outdir, 'model_latest.h5'),
                        period=1),
        ModelCheckpoint(os.path.join(outdir, 'model_{epoch:04}.h5'),
                        period=nth_checkpoint),
        CSVLogger(
            os.path.join(outdir, 'training.log'), separator=',', append=False),
        TensorBoard(
            os.path.join(outdir, 'tensorboard-logs'),
            write_graph=True,
            write_grads=False,
            write_images=True,
            histogram_freq=nth_checkpoint),
        LearningRateScheduler(
            CosineAnnealingSchedule(
                lr_max=lr, lr_min=0.05 * lr, epoch_max=epochs))
    ]
