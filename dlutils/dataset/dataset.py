'''utilites for generating tf.data.Dataset pipelines from tfrecords
generated, e.g. by .tfrec_utils
'''
from glob import glob

import tensorflow as tf

from .cropping import random_crop


def parse_from_tfrecords(filename_pattern,
                         parser_fn,
                         cycle_length=None,
                         num_parallel_calls=tf.data.experimental.AUTOTUNE,
                         balance_records=False):
    '''create a tf.data.Dataset from several tfrecord files.

    Parameters
    ----------
    filename_pattern : string or list of strings.
        string(s) matching all filenames that should be used in the dataset.
        For example: 'data/train/*tfrecord'
    parser_fn : function
        Function to parse a single example. Responsible for converting the
        serialized string tensor into the actual tensors.
    cycle_length : int
        Cycle length for interleaved reading. Default: number of matched files.
    num_parallel_calls : int
        Number of parallel calls for parsing. Default: AUTOTUNE.
    balance_records : bool
        if True, balances the records by repeating them independently before 
        interleaving. Creates an infinite dataset.

    Returns
    -------
    dataset : tf.data.Dataset
        Dataset of parsed samples.

    '''
    if cycle_length is None:
        try:
            cycle_length = len(glob(filename_pattern))
        except TypeError:
            try:
                cycle_length = sum(
                    len(glob(pattern)) for pattern in filename_pattern)
            except:
                raise ValueError(
                    'Could not determine cycle_length from filename_pattern!')

    # collect files.
    filenames = tf.data.Dataset.list_files(filename_pattern)

    if balance_records:
        load_record_fun = lambda x: tf.data.TFRecordDataset(x).repeat()
    else:
        load_record_fun = tf.data.TFRecordDataset

    # read from all with interleave...
    dataset = filenames.interleave(
        load_record_fun,
        # ~dataset = filenames.interleave(tf.data.TFRecordDataset,
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls)
    # ...and parse the samples.
    dataset = dataset.map(parser_fn, num_parallel_calls=num_parallel_calls)
    return dataset


def create_dataset(filename_pattern,
                   batch_size,
                   parser_fn,
                   transforms=None,
                   shuffle_buffer=1000,
                   shuffle=True,
                   drop_remainder=True,
                   cache_after_parse=False,
                   patch_size=None,
                   balance_records=False):
    '''create a tf.data.Dataset pipeline to stream training or validation data from
    from several tfrecords.

    Parameters
    ----------
    filename_pattern : string or list of strings
        strings matching all filenames that should be used in the dataset.
        For example: 'data/train/*tfrecord'
    batch_size : int
        Size of batches to be generated.
    parser_fn : function
        Function to parse a single example. See RecordParser.parse.
    transforms : list of functions
        Transformation to be added. E.g. used for data augmentations
    shuffle_buffer : int
        Size in number of items of buffer for shuffling.
    shuffle : bool
        Shuffle items (before batching). Note that items are read interleaved
        from different files.
    drop_remainder : bool
        Drop incomplete batches.
    cache_after_parse : bool
        Cache samples after reading and parsing.
    patch_size : tuple or None
        Shape of patch to sample. Set to None if no patch sampling is desired.
    balance_records : bool
        iI True, balances the records by repeating them independently before 
        interleaving. Creates an inifinte dataset.

    Returns
    -------
    dataset : tf.data.Dataset
        Dataset of parsed samples.

    Notes
    -----
    * All functions provided in ```transforms``` are applied sequentially.
      This can be used, for example, for augmentations.

    * Patch sampling is applied before any custom transforms. If you need
      pre-sampling transformations, then you should pass the sampling
      function as part of ```transforms``` at the desired position and
      set ```patch_size=None```.

    '''
    num_parallel_calls = tf.data.experimental.AUTOTUNE
    prefetch_buffer = tf.data.experimental.AUTOTUNE

    with tf.device('/cpu:0'):  # place dataset pipeline on cpu.
        dataset = parse_from_tfrecords(filename_pattern,
                                       parser_fn=parser_fn,
                                       num_parallel_calls=num_parallel_calls,
                                       balance_records=balance_records)

        if cache_after_parse:
            dataset = dataset.cache()

        if patch_size is not None:
            dataset = dataset.map(random_crop(patch_size),
                                  num_parallel_calls=num_parallel_calls)

        # apply image augmentations.
        if transforms is not None:
            for fun in transforms:
                dataset = dataset.map(fun,
                                      num_parallel_calls=num_parallel_calls)

        if shuffle and shuffle_buffer >= 2:
            dataset = dataset.shuffle(shuffle_buffer,
                                      reshuffle_each_iteration=True)

        dataset = dataset.batch(batch_size,
                                drop_remainder=drop_remainder).prefetch(
                                    buffer_size=prefetch_buffer)

    return dataset


def create_dataset_for_training(filename_pattern,
                                batch_size,
                                parser_fn,
                                transforms=None,
                                **kwargs):
    '''convenience wrapper for create_dataset for training set.

    See create_dataset for args.

    '''
    params = dict(shuffle=True, drop_remainder=True)
    kwargs.update(params)
    dataset = create_dataset(filename_pattern, batch_size, parser_fn,
                             transforms, **kwargs)
    return dataset


def create_dataset_for_validation(filename_pattern,
                                  batch_size,
                                  parser_fn,
                                  transforms=None):
    '''convenience wrapper for create_dataset for validation set.

    See create_dataset for args.

    '''
    return create_dataset(
        filename_pattern,
        batch_size,
        parser_fn,
        shuffle=True,  # for layers that might depend on batch composition.
        shuffle_buffer=100,
        drop_remainder=True,  # needed for keras training loop.
        transforms=transforms)


def create_linear_dataset(fname,
                          batch_size,
                          parser_fn,
                          transforms=None,
                          prefetch_buffer=tf.data.experimental.AUTOTUNE):
    '''create linear iterator over a single dataset without shuffling or
    dropping of remainders.

    Parameters
    ----------
    filename_pattern : string
        string matching all filenames that should be used in the dataset.
        For example: 'data/train/*tfrecord'
    batch_size : int
        Size of batches to be generated.
    parser_fn : function
        Function to parse a single example. See RecordParser.parse.
    transforms : list of functions
        Transformation to be added. E.g. used for data augmentations.
    prefetch_buffer : int
        Number of samples to prefetch. Default: AUTOTUNE.

    Returns
    -------
    dataset : tf.data.Dataset
        Dataset of parsed and processed samples.

    '''
    num_parallel_calls = tf.data.experimental.AUTOTUNE

    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset(fname).map(
            parser_fn, num_parallel_calls=num_parallel_calls)

        if transforms is not None:
            for fun in transforms:
                dataset = dataset.map(fun,
                                      num_parallel_calls=num_parallel_calls)

        dataset = dataset.batch(batch_size,
                                drop_remainder=False).prefetch(prefetch_buffer)
        return dataset
