'''utility functions to write and read tfrecord datasets.

'''
import os
from glob import glob

from tqdm import tqdm
import tensorflow as tf


def tfrecord_from_iterable(output_path, iterable, serialize_fn, verbose=False):
    '''prepare tfrecords from a sampler.

    Parameters
    ----------
    output_path : string
        path to save tfrecord.
    iterable : iterable
        iterator over samples to be saved. The iterable is expected
        to have limited length.
    serialize_fn : function
        function to serialize samples to tfrecord examples.
        See RecordParser.serialize for an example.
    verbose : bool
        show progressbar.

    '''
    if verbose:
        iterable = tqdm(
            iterable, desc=os.path.basename(output_path), leave=True, ncols=80)
    with tf.io.TFRecordWriter(output_path) as writer:
        for sample in iterable:
            writer.write(serialize_fn(*sample).SerializeToString())


def tfrecord_from_sample(output_path, sample, serialize_fn):
    '''prepare a tensorflow record for a single sample.

    '''
    return tfrecord_from_iterable(output_path, [sample], serialize_fn)


def count_samples_in_records(filename_pattern):
    '''count the number of samples stored in a set of tfrecord files.

    '''
    # TODO replace with eager mode.
    return sum(1 for fname in glob(filename_pattern)
               for _ in tf.python_io.tf_record_iterator(fname))


# From https://www.tensorflow.org/tutorials/load_data/tf_records
# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
    '''Returns a bytes_list from a string / byte.'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    '''Returns a float_list from a float / double.'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    '''Returns an int64_list from a bool / enum / int / uint.'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# TODO Rewrite for base case.
class ImageToClassRecordParser:
    '''defines the serialize and parse function for a typical image
    classification dataset.

    '''
    label_key = 'label'
    image_key = 'image'

    def __init__(self, n_classes, image_dtype, fixed_ndim=None):
        '''
        '''
        self.n_classes = n_classes
        self.image_dtype = image_dtype
        self.fixed_ndim = fixed_ndim
        assert self.fixed_ndim is None or self.fixed_ndim >= 2

    def serialize(self, image, label):
        '''serializes a training tuple such that it can be written as tfrecord example.

        '''
        features = tf.train.Features(
            feature={
                'img_shape': _int64_feature(list(image.shape)),
                self.image_key: _bytes_feature(image.tostring()),
                self.label_key: _int64_feature(label)
            })
        return tf.train.Example(features=features)

    def parse(self, example):
        '''parse a tfrecord example.

        '''
        features = {
            # Extract features using the keys set during creation
            'img_shape': tf.io.FixedLenSequenceFeature([], tf.int64, True),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        sample = tf.io.parse_single_example(example, features)

        # Fixed shape appears to be necessary for training with keras.
        if self.fixed_ndim is not None:
            shape = tf.reshape(sample['img_shape'], (self.fixed_ndim, ))
        else:
            shape = sample['img_shape']

        image = tf.io.decode_raw(sample['image'], self.image_dtype)
        image = tf.reshape(image, shape)
        image = tf.cast(image, tf.float32)
        return {
            self.label_key: tf.one_hot(sample['label'], self.n_classes),
            self.image_key: image
        }
