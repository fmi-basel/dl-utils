from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from dataset import collect_handles

from dlutils.models.factory import construct_base_model, get_model_name
from dlutils.models.utils import add_fcn_output_layers
from dlutils.training.lr_finder import lr_finder
from dlutils.training.generator import TrainingGenerator
from dlutils.training.augmentations import ImageDataAugmentation
from dlutils.training.callbacks import create_callbacks

from keras.optimizers import Adam, SGD
from keras.backend import clear_session
from scipy.stats import norm as gaussian_dist
from scipy.stats import uniform as uniform_dist

import numpy as np
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] (%(name)s) [%(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S')

# reduce tf-clutter output.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO consider moving the basedir to dataset.
basedir = ''  # TODO fill out.


def construct_model(model_name, batch_size, input_shape, learning_rate,
                    pred_names, classes_per_output, optimizer, **model_params):
    '''

    TODO add basemodel constructor as function argument
    '''
    raise NotImplementedError('Adjust model construction for given task')
    model = construct_base_model(
        model_name, input_shape=input_shape, **model_params)
    model = add_fcn_output_layers(model, pred_names, classes_per_output)

    if 'sgd' in optimizer:
        optimizer = SGD(lr=learning_rate, momentum=0.9)
    elif 'adam' in optimizer:
        optimizer = Adam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            decay=0.0)
    model.compile(
        optimizer=optimizer,
        loss={
            'cell_pred': 'binary_crossentropy',
            'border_pred': 'mean_absolute_error'
        },
    )

    logger.info('Compiled model: {}'.format(model.name))
    return model


def get_zero_based_task_id():
    '''
    '''
    sge_id = os.environ.get('SGE_TASK_ID', None)
    if sge_id is None:
        return 0
    else:
        return int(sge_id) - 1


def get_base_config():
    '''
    '''
    config = dict(
        training=dict(
            epochs=200,
            initial_epoch=0,
            use_multiprocessing=False,
            workers=4,  # evil
            max_queue_size=32,
            verbose=2),
        optimizer=dict(name='adam'),
        generator=dict(samples_per_handle=1),
        augmentation=dict(
            intensity_shift=gaussian_dist(loc=0, scale=.3),
            intensity_scaling=gaussian_dist(loc=1, scale=0.1),
            intensity_swap=False,
            flip=True,
            rotation=uniform_dist(-10., 20),
            shear=gaussian_dist(loc=0, scale=10.),
            zoom=uniform_dist(0.95, 0.075)),
        model=dict(batch_size=8),
        lr_schedule=dict(lr_min=1e-7, restart_decay=1., n_restarts=1),
    )

    return config


def get_config(taskId):
    '''generate config based on the current task id.
    '''
    config = get_base_config()

    from sklearn.model_selection import ParameterGrid
    grid = ParameterGrid({
        'model/n_levels': [5],
        'model/dropout': [
            0.0,
            0.05,
        ],
        'model/cardinality': [1, 1.5]
    }, )

    # update base config with specific config.
    for key, val in list(grid[taskId].items()):

        # get the nested dict
        split_keys = key.split('/')
        if len(split_keys) == 1:
            config[key] = val
        else:
            handle = config
            for subkey in split_keys[:-1]:
                handle = handle[subkey]
            handle[split_keys[-1]] = val

    # update some of the config based on the particular setting.
    config['model_name'] = get_model_name(
        dropout_rate=config['model']['dropout'], **config['model'])

    return config


def process(taskId=None):
    '''
    '''
    logger = logging.getLogger(__name__)

    config = get_config(get_zero_based_task_id())

    # log config
    for key, val in list(config.items()):
        if isinstance(val, dict):
            logger.info('{}:'.format(key))
            for subkey, subval in list(val.items()):
                logger.info('  {:30}: {}'.format(subkey, subval))
        else:
            logger.info('{:30}: {}'.format(key, val))

    # TODO save full training config in folder.

    # TODO customize
    outdir = './{}/{}/{}'.format(
        config['base_model'], config['model_name'], config['optimizer']['name']
        + ('-R' if config['lr_schedule']['n_restarts'] > 1 else ''))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # TODO customize
    # TODO consider moving the output definition to dataset.
    input_shape = (300, 300, 3)
    pred_names = ['cell_pred', 'border_pred']
    classes_per_output = [1, 1]

    # collect image handles with training annotation.
    train_handles, val_handles = collect_handles(basedir)
    logger.info('Training images: {}'.format(len(train_handles)))
    logger.info('Validation images: {}'.format(len(val_handles)))
    for handle in val_handles:
        logger.debug('	{}'.format(handle['img_path']))

    train_generator = TrainingGenerator(
        train_handles,
        patch_size=input_shape[:-1],
        batch_size=config['model']['batch_size'],
        samples_per_handle=config['generator']['samples_per_handle'],
        buffer=True,
        seed=13)
    train_generator.augmentator = ImageDataAugmentation(
        **config['augmentation'])

    val_generator = TrainingGenerator(
        val_handles,
        patch_size=input_shape[:-1],
        batch_size=config['model']['batch_size'],
        buffer=True,
        samples_per_handle=1,
        seed=13)

    # TODO move this to a separate function.
    model = construct_model(
        config['base_model'],
        input_shape=input_shape,
        learning_rate=1e-4,
        pred_names=pred_names,
        classes_per_output=classes_per_output,
        optimizer=config['optimizer']['name'],
        **config['model'])

    lrf = lr_finder(
        model,
        train_generator,
        steps=100,
        base_lr=1e-6,
        max_lr=1.0,
        verbose=2,
        workers=config['training']['workers'],
        max_queue_size=config['training']['max_queue_size'],
        use_multiprocessing=True)
    suggested_lr = lrf.suggest_lr(sigma=5.)
    logger.info('Suggested learning rate: {}'.format(suggested_lr))

    clear_session()

    model = construct_model(
        config['base_model'],
        input_shape=input_shape,
        learning_rate=suggested_lr,
        pred_names=pred_names,
        classes_per_output=classes_per_output,
        optimizer=config['optimizer']['name'],
        **config['model'])

    callbacks = create_callbacks(
        lr=suggested_lr,
        outdir=outdir,
        nth_checkpoint=1000,  # dont checkpoint models
        epochs=config['training']['epochs'],
        **config['lr_schedule'])

    # TODO implement deterministic validation split
    np.random.seed(42)  # reset seed to improve reproducibility.
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        **config['training'])


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    logger.debug('Starting')
    logger.info('Host: {}'.format(os.environ.get('HOST')))
    logger.info('GPU cores: {}'.format(os.environ.get('CUDA_VISIBLE_DEVICES')))
    logger.info('Job/Task ID: {}/{}'.format(
        os.environ.get('JOB_ID'), os.environ.get('SGE_TASK_ID')))

    taskId = None
    if len(sys.argv) >= 2:
        if sys.argv[1] == '--show':
            for idx in range(100):
                print(get_config(idx)['model_name'])
            exit()
        else:
            taskId = int(sys.argv[1])

    try:
        process(taskId)
    except Exception as err:
        logger.error(str(err), exc_info=True)
