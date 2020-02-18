from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
import numpy as np

from dlutils.dataset.tfrecords import tfrecord_from_iterable
from dlutils.dataset.tfrecords import tfrecord_from_sample
from dlutils.dataset.tfrecords import ImageToClassRecordParser


def data_generator(n_samples):
    '''yields the first n_samples from the fashion_mnist dataset.

    '''
    (x_train, y_train), (_, _) = fashion_mnist.load_data()
    for sample, target in zip(x_train, y_train[:n_samples]):
        yield sample, target


def test_tfrecord_from_iterable(tmpdir):
    '''test the multi-sample record writer.

    '''
    max_samples = 10
    parser = ImageToClassRecordParser(
        n_classes=10,  # Fixed by dataset.
        image_dtype=tf.uint8)
    output_folder = tmpdir / 'tfrec_iter'
    output_folder.mkdir()
    output_path = str(output_folder / 'fashion_mnist.tfrec')

    tfrecord_from_iterable(output_path, data_generator(max_samples),
                           parser.serialize)

    dataset = tf.data.TFRecordDataset(output_path).map(parser.parse)

    # check if the items drawn from the tfrecord are identical to
    # those from the oracle.
    for counter, (item, (ref_img, ref_label)) in enumerate(
            zip(dataset.take(max_samples), data_generator(max_samples))):
        assert np.argmax(item[parser.label_key].numpy()) == ref_label
        assert np.all(item[parser.image_key].numpy() == ref_img)

    assert counter + 1 == max_samples


def test_tfrecord_from_sample(tmpdir):
    '''test the single-sample record writer.

    '''
    parser = ImageToClassRecordParser(n_classes=10, image_dtype=tf.uint8)
    output_folder = tmpdir / 'tfrec_single'
    output_folder.mkdir()
    output_path = str(output_folder / 'fashion_mnist.tfrec')

    sample = next(data_generator(1))
    tfrecord_from_sample(output_path, sample, parser.serialize)

    dataset = tf.data.TFRecordDataset(output_path).map(parser.parse)

    # check if the items drawn from the tfrecord are identical to
    # those from the oracle.
    for item in dataset.take(1):
        assert np.argmax(item[parser.label_key].numpy()) == sample[1]
        assert np.all(item[parser.image_key].numpy() == sample[0])
