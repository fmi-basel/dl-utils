'''utilites for generating tf.data.Dataset pipelines from tfrecords
generated, e.g. by .tfrec_utils
'''
from glob import glob
from math import ceil

import tensorflow as tf

from .tfrec_utils import count_samples_in_records
from .cropping import random_crop


def create_dataset(filename_pattern,
                   batch_size,
                   parser_fn,
                   augmentations=None,
                   shuffle_buffer=2500,
                   shuffle=True,
                   drop_remainder=True,
                   patch_size=None):
    '''create a tf.data.Dataset pipeline to stream training or validation data from
    from several tfrecords.

    Parameters
    ----------
    filename_pattern : string
        string matching all filenames that should be used in the dataset.
        For example: 'data/train/*tfrecord'
    batch_size : int
        Size of batches to be generated.
    parser_fn : function
        Function to parse a single example. See RecordParser.parse.
    augmentations : list of functions
        Augmentation functions to be added.
    shuffle_buffer : int
        Size in number of items of buffer for shuffling.
    shuffle : bool
        shuffle items (before batching). Note that items are read interleaved
        from different files.
    drop_remainder : bool
        drop incomplete batches.

    '''
    num_parallel_calls = tf.contrib.data.AUTOTUNE
    prefetch_buffer = tf.contrib.data.AUTOTUNE
    cycle_length = len(glob(filename_pattern))

    total_batches = count_samples_in_records(filename_pattern) // batch_size
    if total_batches <= 0:
        raise RuntimeError(
            'Could not find enough tfrecord datasets to fill one batch!')

    # place dataset pipeline on cpu.
    with tf.device('/cpu:0'):

        # collect files.
        filenames = tf.data.Dataset.list_files(filename_pattern)

        # read in interleave mode.
        # NOTE For older versions of tensorflow, this should work instead
        # of the parallel_interleave.
        # dataset = (filenames.interleave(
        #     lambda filename: tf.data.TFRecordDataset(filename),
        #     cycle_length=cycle_length,
        #     block_length=block_length)
        dataset = (filenames.apply(
            tf.data.experimental.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=cycle_length)).map(
                    lambda sample: parser_fn(sample),
                    num_parallel_calls=num_parallel_calls))

        if patch_size is not None:
            dataset = dataset.map(
                random_crop(patch_size),
                num_parallel_calls=num_parallel_calls)

        # apply image augmentations.
        if augmentations is not None:
            for augmentation_fn in augmentations:
                dataset = dataset.map(
                    augmentation_fn, num_parallel_calls=num_parallel_calls)

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.batch(
            batch_size, drop_remainder=drop_remainder).prefetch(
                buffer_size=prefetch_buffer)

    return dataset, total_batches


def create_dataset_for_training(filename_pattern,
                                batch_size,
                                parser_fn,
                                augmentations=None,
                                **kwargs):
    '''see create_dataset for args.

    '''
    params = dict(shuffle=True, shuffle_buffer=2500, drop_remainder=True)
    kwargs.update(params)
    dataset, total_batches = create_dataset(filename_pattern, batch_size,
                                            parser_fn, augmentations, **kwargs)
    return dataset.repeat(), total_batches


def create_dataset_for_validation(filename_pattern, batch_size, parser_fn):
    '''see create_dataset for args.

    '''
    dataset, total_batches = create_dataset(
        filename_pattern,
        batch_size,
        parser_fn,
        shuffle=True,
        shuffle_buffer=100,
        drop_remainder=True)
    return dataset.repeat(), total_batches


def create_linear_dataset(fname, batch_size, parser_fn):
    '''create linear iterator over a single dataset for testing.

    '''
    num_parallel_calls = tf.contrib.data.AUTOTUNE
    prefetch_buffer = tf.contrib.data.AUTOTUNE

    total_batches = int(ceil(count_samples_in_records(fname) / batch_size))

    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset(fname).map(
            lambda sample: parser_fn(sample),
            num_parallel_calls=num_parallel_calls)

        dataset = dataset.batch(
            batch_size, drop_remainder=False).prefetch(prefetch_buffer)
        return dataset.repeat(1), total_batches
