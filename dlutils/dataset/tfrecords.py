'''utility functions to write and read tfrecord datasets.

'''
import abc
import os

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
        iterable = tqdm(iterable,
                        desc=os.path.basename(output_path),
                        leave=True,
                        ncols=80)
    with tf.io.TFRecordWriter(output_path) as writer:
        for sample in iterable:
            writer.write(serialize_fn(*sample).SerializeToString())


def tfrecord_from_sample(output_path, sample, serialize_fn, *args, **kwargs):
    '''prepare a tensorflow record for a single sample.

    '''
    return tfrecord_from_iterable(output_path, [sample], serialize_fn, *args,
                                  **kwargs)


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


class RecordParserBase(abc.ABC):
    '''Base class for RecordParsers.

    This serves the purpose of
    - Defining the RecordParser skeleton
    - Facilitating future extension of parsers.

    '''
    @abc.abstractmethod
    def serialize(self, *args):
        '''convert a training sample into a serializeable tf.Example.

        '''
        pass

    @abc.abstractmethod
    def parse(self, example):
        '''convert the serialized string tensor into the training sample.

        '''
        pass


class ImageToClassRecordParser(RecordParserBase):
    '''defines the serialize and parse function for a typical image
    classification dataset.

    '''
    label_key = 'label'
    image_key = 'image'
    shape_key = 'shape'

    def __init__(self, n_classes, image_dtype, fixed_ndim=None):
        '''
        Parameters
        ----------
        n_classes : int
            number of classes for the label; used for one-hot encoding.
        image_dtype : tf.dtype
            data type of the input image. Note that the output of parse()
            still casts to tf.float32.
        fixed_ndim : int
            length of the image shape. Needed for training loop
            for determining the tensor shapes downstream.

        '''
        self.n_classes = n_classes
        self.image_dtype = image_dtype
        self.fixed_ndim = fixed_ndim
        assert self.fixed_ndim is None or self.fixed_ndim >= 2

    def serialize(self, image, label):
        '''serializes a training tuple such that it can be written as tfrecord example.

        Parameters
        ----------
        image : array-like
            image to be serialized. Needs to have image_dtype.
        label : int
            corresponding label to be serialized.

        '''
        features = tf.train.Features(
            feature={
                self.shape_key: _int64_feature(list(image.shape)),
                self.image_key: _bytes_feature(image.tostring()),
                self.label_key: _int64_feature(label)
            })
        return tf.train.Example(features=features)

    def parse(self, example):
        '''parse a tfrecord example.

        Returns
        -------
        sample : dict of tensors
            sample containing image and label. Note that the image is
            always converted to tf.float32 and the label is one-hot encoded.

        '''
        features = {
            # Extract features using the keys set during creation
            self.shape_key:
            tf.io.FixedLenSequenceFeature([], tf.int64, True),
            self.label_key:
            tf.io.FixedLenFeature([], tf.int64),
            self.image_key:
            tf.io.FixedLenFeature([], tf.string),
        }
        sample = tf.io.parse_single_example(example, features)

        # Fixed shape appears to be necessary for training with keras.
        if self.fixed_ndim is not None:
            shape = tf.ensure_shape(sample[self.shape_key], (self.fixed_ndim, ))
        else:
            shape = sample[self.shape_key]

        image = tf.io.decode_raw(sample[self.image_key], self.image_dtype)
        image = tf.reshape(image, shape)
        image = tf.cast(image, tf.float32)
        return {
            self.label_key: tf.one_hot(sample[self.label_key], self.n_classes),
            self.image_key: image
        }


class ImageToSegmentationRecordParser(RecordParserBase):
    '''defines the serialize and parse function for a typical image
    segmentation dataset.

    '''

    segm_key = 'segm'
    image_key = 'image'
    img_shape_key = 'img_shape'
    segm_shape_key = 'segm_shape'

    def __init__(self, image_dtype, segm_dtype, fixed_ndim=None):
        '''
        Parameters
        ----------
        n_classes : int
            number of classes for the label; used for one-hot encoding.
        image_dtype : tf.dtype
            data type of the input image. Note that the output of parse()
            still casts to tf.float32.
        segm_dtype : tf.dtype
            data type of the input image.
        fixed_ndim : int
            length of the image shape. Needed for training loop
            for determining the tensor shapes downstream.

        '''
        self.image_dtype = image_dtype
        self.segm_dtype = segm_dtype
        self.fixed_ndim = fixed_ndim
        assert fixed_ndim is None or fixed_ndim >= 2

    def serialize(self, image, segm):
        '''serializes a training tuple such that it can be written as tfrecord example.

        Parameters
        ----------
        image : array-like
            image to be serialized. Needs to have image_dtype.
        segm : array-like
            corresponding segmentation to be serialized. Needs to have segm_dtype.

        '''
        targets = {self.segm_key: segm}

        for key, target in targets.items():
            if not all(x == y
                       for x, y in zip(image.shape[:-1], target.shape[:-1])):
                raise ValueError(
                    'Image and {} do not have the same shape: {} vs {}'.format(
                        key, image.shape, target.shape))

        features = tf.train.Features(
            feature={
                self.img_shape_key: _int64_feature(list(image.shape)),
                self.segm_shape_key: _int64_feature(list(segm.shape)),
                self.image_key: _bytes_feature(image.tostring()),
                **{
                    key: _bytes_feature(target.tostring())
                    for key, target in targets.items()
                }
            })
        return tf.train.Example(features=features)

    def parse(self, example):
        '''parse a tfrecord example.

        Returns
        -------
        sample : dict of tensors
            sample containing image and label. Note that the image is
            always converted to tf.float32.

        '''
        features = {
            # Extract features using the keys set during creation
            self.img_shape_key:
            tf.io.FixedLenSequenceFeature([], tf.int64, True),
            self.segm_shape_key:
            tf.io.FixedLenSequenceFeature([], tf.int64, True),
            self.image_key:
            tf.io.FixedLenFeature([], tf.string),
            self.segm_key:
            tf.io.FixedLenFeature([], tf.string),
        }
        sample = tf.io.parse_single_example(example, features)

        # Fixed shape appears to be necessary for training with keras.
        shapes = {}
        for key in [self.img_shape_key, self.segm_shape_key]:
            if self.fixed_ndim is not None:
                shapes[key] = tf.ensure_shape(sample[key], (self.fixed_ndim, ))

        def _reshape_and_cast(val, shape, dtype):
            '''this ensures that tensorflow "knows" the shape of the resulting
            tensors.
            '''
            return tf.reshape(tf.io.decode_raw(val, dtype), shape)

        parsed = {
            key: _reshape_and_cast(sample[key], shape, dtype)
            for key, shape, dtype in zip([self.image_key, self.segm_key], [
                shapes[self.img_shape_key], shapes[self.segm_shape_key]
            ], [self.image_dtype, self.segm_dtype])
        }
        parsed[self.image_key] = tf.cast(parsed[self.image_key], tf.float32)
        return parsed
