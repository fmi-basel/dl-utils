import logging
import os

from dlutils.training.lr_finder import lr_finder
from dlutils.training.callbacks import create_callbacks

from keras.backend import clear_session


def estimate_learning_rate(model_constructor, dataset):
    '''estimate learning rate from a sweep of increasing learning rate.

    '''
    logger = logging.getLogger(__name__)

    logger.info('Starting learning rate estimation')
    model = model_constructor()
    lrf = lr_finder(
        model,
        dataset,
        steps=100,
        base_lr=1e-6,
        max_lr=1.0,
        verbose=1,
    )

    learning_rate = lrf.suggest_lr(sigma=5.)
    logger.debug('Estimated starting learning rate: {}'.format(learning_rate))
    clear_session()

    return learning_rate


def train(dataset,
          model_constructor,
          outdir,
          learning_rate=None,
          steps_per_epoch=None,
          n_restarts=None,
          **training_config):
    '''
    '''
    logger = logging.getLogger(__name__)

    if learning_rate is None:
        learning_rate = estimate_learning_rate(model_constructor,
                                               dataset['training'])
    logger.info('Using learning_rate={}'.format(learning_rate))

    # build model
    model = model_constructor()
    model.summary()
    outdir = os.path.join(outdir, model.name)

    # setup callbacks
    callbacks = create_callbacks(
        lr=learning_rate,
        outdir=outdir,
        nth_checkpoint=10000,  # dont checkpoint models too frequently.
        n_restarts=n_restarts,
        epochs=training_config['epochs'])

    if steps_per_epoch is None:
        steps_per_epoch = len(dataset['training'])

    model.fit_generator(
        dataset['training'],
        steps_per_epoch=steps_per_epoch,
        validation_data=dataset['validation'],
        validation_steps=training_config.get('validation_steps',
                                             len(dataset['validation'])),
        callbacks=callbacks,
        **training_config)
    return outdir
