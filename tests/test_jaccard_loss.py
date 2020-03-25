import pytest
import tensorflow as tf
import numpy as np

from dlutils.losses.jaccard_loss import JaccardLoss, BinaryJaccardLoss


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


def test_JaccardLoss_hinge():
    '''Test that hinge loss falls back to hard prediction when pred is 
    over the hinge threshold'''

    np.random.seed(25)
    n_classes = 5

    yt = np.random.choice(range(n_classes), size=(4, 10, 10, 1))
    yp = yt

    one_hot = tf.cast(tf.one_hot(tf.squeeze(yt, -1), n_classes), tf.float32)
    probs = tf.cast(tf.one_hot(tf.squeeze(yp, -1), n_classes), tf.float32)

    # yp == to groundtruth --> perfect loss
    loss = JaccardLoss(eps=0)(one_hot, probs)
    np.testing.assert_almost_equal(loss, 0.)

    # replace hard prediction by 0.1, 0.9 --> imperfect loss
    probs = 0.8 * probs + 0.1
    loss = JaccardLoss(eps=0)(one_hot, probs)
    assert loss > 0.

    # hinged loss with  0.2 threshold should fall back to hard prediction
    loss = JaccardLoss(eps=0, hinge_probs=(0.2, 0.8))(one_hot, probs)
    np.testing.assert_almost_equal(loss, 0., decimal=3)


def test_BinaryJaccardLoss():
    '''Verifies that the soft Jaccard loss on binary classification task
    behaves as IoU calculated in numpy.
    '''
    def numpy_iou(yt, yp):
        intersection = (yt * yp).sum()
        union = (yt + yp).sum() - intersection
        return 1. - intersection / union

    np.random.seed(17)

    # random labels
    yt = np.random.choice([0, 1], size=(4, 10, 10, 1))
    yp = np.random.choice([0, 1], size=(4, 10, 10, 1))

    expected_loss = numpy_iou(yt, yp)
    loss = BinaryJaccardLoss(eps=0)(yt.astype(np.float32),
                                    yp.astype(np.float32))
    np.testing.assert_almost_equal(loss, expected_loss, decimal=3)

    # random labels with unannot mask
    yt = np.random.choice([-1, 0, 1], size=(4, 10, 10, 1))
    yp = np.random.choice([0, 1], size=(4, 10, 10, 1))
    labelled_mask = yt >= 0

    expected_loss = numpy_iou(yt[labelled_mask], yp[labelled_mask])
    loss = BinaryJaccardLoss(eps=0)(yt.astype(np.float32),
                                    yp.astype(np.float32))
    np.testing.assert_almost_equal(loss, expected_loss, decimal=3)


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
