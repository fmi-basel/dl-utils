import pytest
import tensorflow as tf
import numpy as np

from dlutils.losses.jaccard_loss import JaccardLoss

# TODO update once binary and multiclass are treated separately


def test_JaccardLoss():
    '''Verifies that the soft Jaccard loss behaves as keras MeanIoU when 
    probabilities are either 0 or 1
    '''
    np.random.seed(25)
    n_classes = 5

    # random labels, 5 classes, batch size = 4
    yt = np.random.choice(range(n_classes), size=(4, 10, 10, 1))
    yp = np.random.choice(range(n_classes), size=(4, 10, 10, 1))

    m = tf.keras.metrics.MeanIoU(num_classes=n_classes)
    m.update_state(yt, tf.cast(yp, tf.int32))
    expected_loss = 1. - m.result().numpy()

    one_hot = tf.cast(tf.one_hot(tf.squeeze(yt, -1), n_classes), tf.float32)
    probs = tf.cast(tf.one_hot(tf.squeeze(yp, -1), n_classes), tf.float32)
    loss = JaccardLoss(eps=0)(one_hot, probs)

    np.testing.assert_almost_equal(loss, expected_loss, decimal=3)

    perfect_loss = JaccardLoss(eps=0)(one_hot, one_hot)
    np.testing.assert_almost_equal(perfect_loss, 0., decimal=3)


def test_JaccardLoss_fgonly():
    '''Verifies that the soft Jaccard loss behaves as keras MeanIoU when 
    probabilities are either 0 or 1 AND annotations are partial,
    contains unannot label < 0.
    '''
    np.random.seed(25)
    n_classes = 4

    # random labels, 4 classes + unannot label = -1, batch size = 4
    yt = np.random.choice(range(-1, n_classes), size=(4, 10, 10, 1))
    yp = np.random.choice(range(0, n_classes), size=(4, 10, 10, 1))
    labelled_mask = yt >= 0

    m = tf.keras.metrics.MeanIoU(num_classes=n_classes)
    m.update_state(yt[labelled_mask], tf.cast(yp[labelled_mask], tf.int32))
    expected_loss = 1. - m.result().numpy()

    one_hot = tf.cast(tf.one_hot(tf.squeeze(yt, -1), n_classes), tf.float32)
    probs = tf.cast(tf.one_hot(tf.squeeze(yp, -1), n_classes), tf.float32)
    loss = JaccardLoss(eps=0, fg_only=False)(one_hot, probs)

    loss_fg = JaccardLoss(eps=0, fg_only=True)(one_hot, probs)

    np.testing.assert_almost_equal(loss, expected_loss, decimal=3)

    # ~perfect_loss = JaccardLoss(eps=0)(one_hot, one_hot)
    # ~np.testing.assert_almost_equal(perfect_loss, 0., decimal=3)


def test_JaccardLoss_training():
    '''Verifies that the JaccardLoss can be used to learn a simple thresholding operation.'''

    np.random.seed(25)
    raw = np.random.normal(size=(1, 10, 10, 1)).astype(np.float32)
    yt = (raw > 0.0).astype(np.float32)
    dataset = tf.data.Dataset.from_tensors((raw, yt))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1,
                               kernel_size=1,
                               padding='same',
                               activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10.),
                  loss=JaccardLoss())

    loss_before = model.evaluate(dataset)
    model.fit(dataset, epochs=100)
    loss_after = model.evaluate(dataset)

    assert loss_before * 0.95 >= loss_after
    assert loss_after < 0.001
