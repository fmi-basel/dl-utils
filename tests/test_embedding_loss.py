import pytest
import tensorflow as tf
import numpy as np

from scipy.ndimage.measurements import mean as label_mean
from dlutils.losses.embedding.embedding_loss import _unbatched_soft_jaccard, _unbatched_label_to_hot, _unbatched_embedding_center, _unbatched_embeddings_to_prob, InstanceEmbeddingLossBase, InstanceMeanIoUEmbeddingLoss


def test__unbatched_soft_jaccard():
    '''Verifies that the soft Jaccard loss behaves as keras MeanIoU when 
    probabilities are either 0 or 1 and that background masking works
    '''

    np.random.seed(25)
    n_classes = 5

    # random labels, 5 classes, batch size = 4
    yt = np.random.choice(range(n_classes), size=(10, 10, 1))
    yp = np.random.choice(range(n_classes), size=(10, 10, 1))

    m = tf.keras.metrics.MeanIoU(num_classes=n_classes)
    m.update_state(yt, tf.cast(yp, tf.int32))
    expected_loss = 1. - m.result().numpy()

    print(yt.squeeze())
    print(yp.squeeze())
    one_hot = tf.cast(tf.one_hot(tf.squeeze(yt, -1), n_classes), tf.float32)
    probs = tf.cast(tf.one_hot(tf.squeeze(yp, -1), n_classes), tf.float32)
    loss = _unbatched_soft_jaccard(one_hot, probs, fg_only=False,
                                   eps=1e-6).numpy().mean()

    np.testing.assert_almost_equal(loss, expected_loss, decimal=3)

    # check with/without background on simple example
    yt = np.array([0, 0, 1, 1, 2, 2])[..., None]
    yp = np.array([0, 1, 0, 1, 2, 2])[..., None]

    one_hot = tf.cast(tf.one_hot(tf.squeeze(yt, -1), 3), tf.float32)
    probs = tf.cast(tf.one_hot(tf.squeeze(yp, -1), 3), tf.float32)

    loss = _unbatched_soft_jaccard(one_hot, probs,
                                   fg_only=False).numpy().mean()
    loss_fg = _unbatched_soft_jaccard(one_hot[..., 1:],
                                      probs[..., 1:],
                                      fg_only=True).numpy().mean()

    np.testing.assert_almost_equal(loss, ((1 - 1 / 3) + (1 - 1 / 3)) / 3,
                                   decimal=3)
    np.testing.assert_almost_equal(loss_fg, (1 - 1 / 2) / 2, decimal=3)


def test__unbatched_label_to_hot():

    np.random.seed(25)
    labels = np.random.choice(range(5), size=(10, 10, 1)).astype(np.int32)
    #remove label id 3
    labels[labels == 3] = 0

    hot_labels = _unbatched_label_to_hot(labels)

    # #channels == #unique labels - bg and label 3
    assert hot_labels.shape == (10, 10, 3)

    for idx, l in enumerate([1, 2, 4]):
        hot_slice = hot_labels[..., idx].numpy().astype(bool)
        l_mask = labels.squeeze() == l

        np.testing.assert_array_equal(hot_slice, l_mask)


def test__unbatched_embedding_center():

    np.random.seed(25)
    labels = np.random.choice(range(5), size=(10, 10, 1)).astype(np.int32)
    hot_labels = _unbatched_label_to_hot(labels)

    yp = np.random.rand(10, 10, 3).astype(np.float32)

    centers = _unbatched_embedding_center(hot_labels, yp)
    assert centers.shape == (1, 1, 4, 3)

    expected_centers = np.stack([
        label_mean(p, labels.squeeze(), [1, 2, 3, 4])
        for p in np.moveaxis(yp, -1, 0)
    ],
                                axis=-1)
    np.testing.assert_array_almost_equal(centers.numpy().squeeze(),
                                         expected_centers)


def test__unbatched_embeddings_to_prob():
    '''check that perfect embeddings/centers fall back on one hot encoded labels'''

    np.random.seed(25)
    labels = np.random.choice(range(5), size=(10, 10, 1)).astype(np.int32)
    centers = np.array([[1], [2], [3], [4]], dtype=np.float32)

    probs = _unbatched_embeddings_to_prob(labels.astype(np.float32),
                                          centers,
                                          margin=0.5,
                                          clip_probs=None)

    for idx in range(4):
        prob_slice_thresh = probs[..., idx].numpy() > 0.5
        l_mask = labels.squeeze() == idx + 1

        np.testing.assert_array_equal(prob_slice_thresh, l_mask)


def test__unbatched_embeddings_to_prob_1D():
    '''checks prob/margin relationship in 1D case'''

    yp = np.arange(100, dtype=np.float32)[..., None]
    centers = np.array([[0]], dtype=np.float32)

    for margin in range(1, 20):
        probs = _unbatched_embeddings_to_prob(yp,
                                              centers,
                                              margin=margin,
                                              clip_probs=None)
        first_negative = np.argwhere((probs.numpy() < 0.5).squeeze())[0, 0]

        # check that first prob<0.5 is ~ margin away from center
        assert first_negative == margin + 1 or first_negative == margin


def test_InstanceEmbeddingLossBase():
    class InstanceMeanIoUEmbeddingLoss(InstanceEmbeddingLossBase):
        def _unbatched_loss(self, packed):
            y_true, y_pred = packed
            y_true = tf.cast(y_true, tf.float32)

            return tf.math.reduce_mean(tf.abs(y_true - y_pred))

    yt = np.broadcast_to(
        np.arange(10, dtype=np.float32)[:, None, None, None],
        (10, 10, 10, 1)).copy()
    yp = (yt + 1).astype(np.float32)

    loss = InstanceMeanIoUEmbeddingLoss()(yt, yp)
    np.testing.assert_almost_equal(loss, 1.)

    # perfect prediction for samples 0 and 5
    yp[0] = 0
    yp[5] = 5
    loss = InstanceMeanIoUEmbeddingLoss()(yt, yp)
    np.testing.assert_almost_equal(loss, 0.8)

    # unlabel (set negative labels) for samples 0 and 5 so that they are ignored in loss
    yt[0] = -1
    yt[5] = -1
    loss = InstanceMeanIoUEmbeddingLoss()(yt, yp)
    np.testing.assert_almost_equal(loss, 1.)

    # unlabel all
    yt[:] = -1
    loss = InstanceMeanIoUEmbeddingLoss()(yt, yp)
    np.testing.assert_almost_equal(loss, 0.)


def test_InstanceMeanIoUEmbeddingLoss():

    np.random.seed(25)
    n_classes = 5

    # random labels, 5 classes, batch size = 4
    yt = np.random.choice(range(n_classes),
                          size=(4, 10, 10, 1)).astype(np.int32)
    yp_prefect = np.broadcast_to(yt.astype(np.float32), (4, 10, 10, 1))

    loss_perfect = InstanceMeanIoUEmbeddingLoss(margin=0.001)(
        yt, yp_prefect).numpy()
    loss_clipped = InstanceMeanIoUEmbeddingLoss(margin=0.001,
                                                clip_probs=(0.01, 0.99))(
                                                    yt, yp_prefect).numpy()
    loss_marginA = InstanceMeanIoUEmbeddingLoss(margin=0.5)(
        yt, yp_prefect).numpy()
    loss_marginB = InstanceMeanIoUEmbeddingLoss(margin=0.7)(
        yt, yp_prefect).numpy()

    np.testing.assert_almost_equal(loss_perfect, 0.)
    assert loss_perfect < loss_clipped
    assert loss_perfect < loss_marginA
    assert loss_marginA < loss_marginB
