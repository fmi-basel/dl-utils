from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
from glob import glob

from simpledl.config import read_config, Config
from simpledl.dataset import prepare_dataset
from simpledl.training import train

from dlutils.utils import set_seeds
from dlutils.models.factory import construct_base_model
from dlutils.models.heads import add_fcn_output_layers

from keras.optimizers import Adam

# general setup
# logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] (%(name)s) [%(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S')

# reduce tf-clutter output.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# seeds
set_seeds()

# Define problem specific settings
SETTINGS = {
    'training': {
        'epochs': 100,
        'steps_per_epoch': None,
        'learning_rate': None,
        'use_multiprocessing': False,
        'workers': 4
    },
    'dataset': {
        'patch_size': (290, 290),
        #        'task_type': 'instance_segmentation',
        'task_type': 'binary_segmentation',
        'batch_size': 4,
    },
    'augmentation': None,
    'model': {
        'model_name': 'rxunet',
        'n_levels': 4,
        'width': 1,
        'cardinality': 8,
        'n_blocks': 2,
        'input_shape': (None, None, 1),
        'dropout': 0.,
    }
}


def collect_data(input_dir, target_dir, pattern='*.tif'):
    '''gathers data for training from two folders. Assumes that
    the matching pairs have the same name.

    '''
    input_paths = sorted(glob(os.path.join(input_dir, pattern)))

    path_pairs = [(input_path, target_path) for input_path, target_path in (
        (path, os.path.join(target_dir, os.path.basename(path)))
        for path in input_paths) if os.path.exists(target_path)]
    return path_pairs


def get_model(model_name, input_shape, task_type, **model_kwargs):
    '''returns a model constructor function.

    '''
    if task_type == 'instance_segmentation':
        loss = {
            'fg_pred': 'binary_crossentropy',
            'separator_pred': 'mean_absolute_error'
        }
        pred_names = list(loss.keys())
        classes_per_output = [1, 1]
    elif task_type == 'binary_segmentation':
        loss = {'fg_pred': 'binary_crossentropy'}
        pred_names = list(loss.keys())
        classes_per_output = [1]
    else:
        raise ValueError('Unknown task type: {}'.format(task_type))

    def model_constructor():
        '''
        '''
        # get base model.
        model = construct_base_model(
            model_name, input_shape=input_shape, **model_kwargs)

        # and add desired heads.
        model = add_fcn_output_layers(model, pred_names, classes_per_output)

        optimizer = Adam(
            lr=0.001,  # NOTE LR will be overwritten by the callback.
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            decay=0.0)
        model.compile(
            optimizer=optimizer,
            loss=loss,
        )
        logger = logging.getLogger(__name__)
        logger.info('Compiled model: {}'.format(model.name))
        return model

    return model_constructor


def parse():
    '''parse command line arguments.

    '''
    logger = logging.getLogger('parse')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        help='config to load [.yaml]',
        required=False,
        default=None)
    parser.add_argument(
        '--dataset_input', help='directory with input data', required=True)
    parser.add_argument(
        '--dataset_target',
        help='directory with target data (ground truth)',
        required=True)
    parser.add_argument(
        '--output', help='directory to save trained model', required=True)

    args = parser.parse_args()
    logger.debug('Parsed arguments:')
    logger.debug('  config=%s', args.config)
    logger.debug('  dataset=(%s, %s)', args.dataset_input, args.dataset_target)
    logger.debug('  output=%s', args.output)
    return args


def main():
    '''
    '''
    logger = logging.getLogger('main')

    try:
        args = parse()

        # load config and set preliminary path.
        config = Config(**SETTINGS) if args.config is None else read_config(
            args.config)
        config.path = os.path.join(args.output, 'config.yaml')

        with config:
            dataset = prepare_dataset(
                collect_data(args.dataset_input, args.dataset_target),
                augmentation_params=config['augmentation'],
                **config['dataset'])
            model_constructor = get_model(
                task_type=config['dataset']['task_type'], **config['model'])
            train(
                dataset,
                model_constructor,
                outdir=args.output,
                **config['training'])

    except Exception as err:
        logger.error('Error: %s', str(err), exc_info=True)
        return 1
    return 0


if __name__ == '__main__':
    main()
