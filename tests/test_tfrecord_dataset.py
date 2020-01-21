import itertools
import abc
from glob import glob

import pytest
import numpy as np
import tensorflow as tf

from dlutils.dataset.tfrecords import tfrecord_from_iterable
from dlutils.dataset.tfrecords import ImageToClassRecordParser
from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser
from dlutils.dataset.dataset import create_dataset
from dlutils.dataset.dataset import create_linear_dataset


@pytest.fixture(autouse=True)
def clean_session():
    '''
    '''
    yield  # run the test here. Afterwards, clean up...
    tf.keras.backend.clear_session()


class Setup(abc.ABC):
    '''base for setting up tfrecord datasets for testing the
    tf.data pipeline.

    '''

    def __init__(self, n_files, n_samples_per_file, output_folder):
        '''
        '''
        self.n_files = n_files
        self.n_samples_per_file = n_samples_per_file
        self.fname_pattern = None
        self.run(output_folder)

    @property
    def expected_samples(self):
        '''
        '''
        n_samples = self.n_files * self.n_samples_per_file
        assert n_samples >= 1
        return n_samples

    @property
    @abc.abstractmethod
    def parser(self):
        '''provides the tfrecord serialize/parse methods.
        '''
        pass

    @abc.abstractmethod
    def _generator(self):
        '''generates an infinite number of fake data samples.
        '''
        pass

    def run(self, output_folder):
        '''actual creation of tfrecords.

        '''
        output_folder = output_folder / 'tfrecords'
        output_folder.mkdir()

        gen = self._generator()

        for ii in range(self.n_files):
            output_path = str(output_folder / 'record{:02}.tfrec'.format(ii))
            tfrecord_from_iterable(output_path, [
                sample
                for _, sample in zip(range(self.n_samples_per_file), gen)
            ], self.parser.serialize)

        self.fname_pattern = str(output_folder / 'record*.tfrec')


class ClassificationDatasetSetup(Setup):
    '''create an image-to-class dataset.

    '''
    parser = ImageToClassRecordParser(
        image_dtype=tf.uint8, n_classes=10, fixed_ndim=3)
    img_shape = (28, 28, 1)

    def _generator(self):
        '''generates images of random noise and labels as
        function of the sample count.

        '''
        counter = 0
        while True:
            image = np.random.randint(
                0, 255, np.prod(self.img_shape),
                dtype=np.uint8).reshape(self.img_shape)
            label = counter % self.parser.n_classes
            yield image, label
            counter += 1


class SegmentationDatasetSetup(Setup):
    '''create an image-to-segmentation dataset.

    '''
    parser = ImageToSegmentationRecordParser(
        image_dtype=tf.uint8, segm_dtype=tf.uint8, fixed_ndim=3)

    img_shape = (16, 16, 1)
    segm_shape = img_shape
    n_classes = 3

    def _generator(self):
        '''generates images of random noise and label maps as
        function of the sample count.
        '''
        counter = 0
        while True:
            image = np.random.randint(
                0, 255, np.prod(self.img_shape),
                dtype=np.uint8).reshape(self.img_shape)
            segm = np.ones(
                self.segm_shape, dtype=np.uint8) * counter % self.n_classes
            yield image, segm
            counter += 1


@pytest.mark.parametrize(
    'n_files, n_samples_per_file, batch_size, shuffle, drop_remainder, cache_after_parse',
    itertools.product([1, 3], [5], [1, 3], [False, True], [False, True],
                      [False, True]))
def test_create_dataset_clf(tmpdir, n_files, n_samples_per_file, batch_size,
                            shuffle, drop_remainder, cache_after_parse):
    '''image-to-class dataset.
    '''
    # test setup
    setup = ClassificationDatasetSetup(n_files, n_samples_per_file, tmpdir)

    # load dataset.
    dataset = create_dataset(
        setup.fname_pattern,
        batch_size,
        setup.parser.parse,
        shuffle_buffer=5,
        shuffle=shuffle,
        drop_remainder=drop_remainder,
        cache_after_parse=cache_after_parse)

    counter = 0
    for batch in dataset:
        image_batch = batch['image']
        label_batch = batch['label']
        assert np.all(image_batch.shape[1:] == setup.img_shape)
        assert np.all(label_batch.shape[1:] == (
            setup.parser.n_classes, ))  # one-hot encoding.

        if drop_remainder:
            assert label_batch.shape[0] == batch_size
            assert image_batch.shape[0] == batch_size
        else:
            assert label_batch.shape[0] <= batch_size
            assert image_batch.shape[0] <= batch_size

        # check value range
        for image, label in zip(image_batch, label_batch):
            assert np.all(0 <= label) and np.all(label <= 1)
            assert np.all(0 <= image) and np.all(image <= 255)

            counter += 1

    # check correct number of samples
    # Depending on the batch size, some samples might be dropped.
    expected_samples = setup.expected_samples - (
        setup.expected_samples % batch_size if drop_remainder else 0)
    assert counter == expected_samples


@pytest.mark.parametrize(
    'n_files, n_samples_per_file, batch_size, shuffle, drop_remainder, cache_after_parse,',
    itertools.product([1, 3], [5], [1, 3], [False, True], [False, True],
                      [False, True]))
def test_create_dataset_segm(tmpdir, n_files, n_samples_per_file, batch_size,
                             shuffle, drop_remainder, cache_after_parse):
    '''image-to-segmentation dataset.
    '''
    # test setup
    setup = SegmentationDatasetSetup(n_files, n_samples_per_file, tmpdir)

    # load dataset.
    dataset = create_dataset(
        setup.fname_pattern,
        batch_size,
        setup.parser.parse,
        shuffle_buffer=5,
        shuffle=shuffle,
        drop_remainder=drop_remainder,
        cache_after_parse=cache_after_parse)

    counter = 0
    for batch in dataset:
        image_batch = batch['image']
        segm_batch = batch['segm']
        assert np.all(image_batch.shape[1:] == setup.img_shape)
        assert np.all(segm_batch.shape[1:] == setup.segm_shape)

        if drop_remainder:
            assert segm_batch.shape[0] == batch_size
            assert image_batch.shape[0] == batch_size
        else:
            assert segm_batch.shape[0] <= batch_size
            assert image_batch.shape[0] <= batch_size

        # check value range
        for image, label in zip(image_batch, segm_batch):
            assert np.all(0 <= label) and np.all(label <= setup.n_classes)
            assert np.all(0 <= image) and np.all(image <= 255)

            counter += 1

    # check correct number of samples
    # Depending on the batch size, some samples might be dropped.
    expected_samples = setup.expected_samples - (
        setup.expected_samples % batch_size if drop_remainder else 0)
    assert counter == expected_samples


@pytest.mark.parametrize('batch_size, patch_size',
                         itertools.product([1, 3], [(4, 4, 1), (5, 3, 1)]))
def test_create_dataset_with_patches(tmpdir, batch_size, patch_size):
    '''test image-to-segmentation with patch sampling.
    '''
    drop_remainder = True

    # test setup
    setup = SegmentationDatasetSetup(2, 7, tmpdir)

    # load dataset.
    dataset = create_dataset(
        setup.fname_pattern,
        batch_size,
        setup.parser.parse,
        patch_size=patch_size,
        shuffle_buffer=5,
        drop_remainder=drop_remainder,
        cache_after_parse=False)

    counter = 0
    for batch in dataset:
        image_batch = batch['image']
        segm_batch = batch['segm']
        assert np.all(image_batch.shape[1:] == patch_size)
        assert np.all(segm_batch.shape[1:] == patch_size)

        if drop_remainder:
            assert segm_batch.shape[0] == batch_size
            assert image_batch.shape[0] == batch_size
        else:
            assert segm_batch.shape[0] <= batch_size
            assert image_batch.shape[0] <= batch_size

        # check value range
        for image, label in zip(image_batch, segm_batch):
            assert np.all(0 <= label) and np.all(label <= setup.n_classes)
            assert np.all(0 <= image) and np.all(image <= 255)

            counter += 1

    # check correct number of samples
    # Depending on the batch size, some samples might be dropped.
    expected_samples = setup.expected_samples - (
        setup.expected_samples % batch_size if drop_remainder else 0)
    assert counter == expected_samples


@pytest.mark.parametrize('batch_size', [1, 2, 3])
def test_linear_dataset(tmpdir, batch_size):
    '''
    '''
    setup = SegmentationDatasetSetup(1, 7, tmpdir)

    fname = glob(setup.fname_pattern)[0]
    dataset = create_linear_dataset(fname, batch_size, setup.parser.parse)

    counter = 0
    for batch in dataset:
        image_batch = batch['image']
        segm_batch = batch['segm']
        assert np.all(image_batch.shape[1:] == setup.img_shape)
        assert np.all(segm_batch.shape[1:] == setup.segm_shape)

        assert segm_batch.shape[0] <= batch_size
        assert image_batch.shape[0] <= batch_size

        # check value range
        for image, label in zip(image_batch, segm_batch):
            # based on how SegmentationDatasetSetup generates the fake labels:
            assert np.all(label == counter % setup.n_classes)
            assert np.all(0 <= image) and np.all(image <= 255)
            counter += 1

    # check correct number of samples
    assert counter == setup.expected_samples


def test_training_from_dataset(tmpdir):
    '''
    '''
    batch_size = 2
    patch_size = (4, 4, 1)
    drop_remainder = True

    setup = SegmentationDatasetSetup(2, 14, tmpdir)

    def _dict_to_tuple(sample):
        '''In TF 2.0, keras' model.fit() doesnt accept tf.data.Datasets
        with dictionaries, so we have to convert it into a tuple.

        '''
        return sample['image'], sample['segm']

    # load dataset.
    dataset = create_dataset(
        setup.fname_pattern,
        batch_size,
        setup.parser.parse,
        patch_size=patch_size,
        shuffle_buffer=5,
        drop_remainder=drop_remainder,
        transforms=[_dict_to_tuple],
        cache_after_parse=False)

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(4, kernel_size=3, padding='same'),
        tf.keras.layers.Conv2D(1, kernel_size=3, padding='same'),
    ])

    # train.
    model.compile(loss='mse')
    model.fit(dataset, epochs=5)


def test_training_with_multioutput(tmpdir):
    '''train a dummy network with multiple outputs using the tf.data.Dataset.
    The tf.data pipeline yields tuples for inputs/targets.

    '''
    batch_size = 2
    drop_remainder = True

    tf.random.set_seed(13)

    setup = ClassificationDatasetSetup(2, 14, tmpdir)

    def _dict_to_multi_tuple(sample):
        '''In TF 2.0, keras' model.fit() doesnt accept tf.data.Datasets
        with dictionaries, so we have to convert it into a tuple.

        This also creates a fake multitarget output.

        '''
        return sample['image'], (
            sample['label'], 2 * tf.reduce_mean(sample['image']) * tf.ones(3))

    # load dataset.
    dataset = create_dataset(
        setup.fname_pattern,
        batch_size,
        setup.parser.parse,
        patch_size=None,
        shuffle_buffer=5,
        drop_remainder=drop_remainder,
        transforms=[_dict_to_multi_tuple],
        cache_after_parse=False)

    # create model and train
    def _construct_model():
        '''create model with two outputs.

        '''

        first_input = tf.keras.layers.Input(shape=(16, 16, 1))
        x = tf.keras.layers.Conv2D(
            4, kernel_size=3, padding='same')(first_input)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        first_output = tf.keras.layers.Dense(1, name='first')(x)
        second_output = tf.keras.layers.Dense(3, name='second')(x)

        return tf.keras.Model(
            inputs=[first_input], outputs=[first_output, second_output])

    model = _construct_model()
    model.compile(loss={'first': 'mse', 'second': 'mae'})

    loss_before = model.evaluate(dataset)[0]
    model.fit(dataset, epochs=5)
    loss_after = model.evaluate(dataset)[0]

    assert loss_before * 0.95 >= loss_after


def test_training_with_multioutput_dict(tmpdir):
    '''train a dummy network with multiple outputs using the tf.data.Dataset.

    '''
    batch_size = 2
    drop_remainder = True

    tf.random.set_seed(13)

    setup = ClassificationDatasetSetup(2, 14, tmpdir)

    def _dict_to_tuple(sample):
        '''In TF 2.0, keras' model.fit() doesnt accept tf.data.Datasets
        with dictionaries, so we have to convert it into a tuple.

        This also creates a fake multitarget output.

        '''
        return {
            'image': sample['image']
        }, {
            'first': sample['label'],
            'second': 2 * tf.reduce_mean(sample['image']) * tf.ones(3)
        }

    # load dataset.
    dataset = create_dataset(
        setup.fname_pattern,
        batch_size,
        setup.parser.parse,
        patch_size=None,
        shuffle_buffer=5,
        drop_remainder=drop_remainder,
        transforms=[_dict_to_tuple],
        cache_after_parse=False)

    # create model and train
    def _construct_model():
        '''create model with two outputs.

        '''

        first_input = tf.keras.layers.Input(shape=(16, 16, 1), name='image')
        x = tf.keras.layers.Conv2D(
            4, kernel_size=3, padding='same')(first_input)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        first_output = tf.keras.layers.Dense(1, name='first')(x)
        second_output = tf.keras.layers.Dense(3, name='second')(x)

        return tf.keras.Model(
            inputs=[first_input], outputs=[first_output, second_output])

    model = _construct_model()
    model.compile(loss={'first': 'mse', 'second': 'mae'})

    loss_before = model.evaluate(dataset)[0]
    model.fit(dataset, epochs=5)
    loss_after = model.evaluate(dataset)[0]

    assert loss_before * 0.95 >= loss_after
