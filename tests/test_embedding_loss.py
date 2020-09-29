import pytest
import tensorflow as tf
import numpy as np

from scipy.ndimage.measurements import mean as label_mean
from skimage.segmentation import relabel_sequential as sk_relabel_sequential
from dlutils.losses.embedding.embedding_loss import InstanceEmbeddingLossBase, SpatialInstanceEmbeddingLossBase, InstanceMeanIoUEmbeddingLoss, MarginInstanceEmbeddingLoss, relabel_sequential


class DummySpatialInstanceEmbeddingLoss(SpatialInstanceEmbeddingLossBase):
    def _center_dist_to_probs(self, one_hot, center_dist):
        pass


def test__unbatched_soft_jaccard():
    '''Verifies that the soft Jaccard loss behaves as keras MeanIoU when 
    probabilities are either 0 or 1 and that background masking works
    '''

    _unbatched_soft_jaccard = DummySpatialInstanceEmbeddingLoss(
    )._unbatched_soft_jaccard

    np.random.seed(25)
    n_classes = 5

    # random labels, 5 classes, batch size = 4
    yt = np.random.choice(range(n_classes), size=(10, 10, 1))
    yp = np.random.choice(range(n_classes), size=(10, 10, 1))

    m = tf.keras.metrics.MeanIoU(num_classes=n_classes)
    m.update_state(yt, tf.cast(yp, tf.int32))
    expected_loss = 1. - m.result().numpy()

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

    _unbatched_label_to_hot = DummySpatialInstanceEmbeddingLoss(
    )._unbatched_label_to_hot

    np.random.seed(25)
    labels = np.random.choice(range(5), size=(10, 10, 1)).astype(np.int32)

    hot_labels = _unbatched_label_to_hot(labels)

    # #channels == #unique labels - bg
    assert hot_labels.shape == (10, 10, 4)

    for idx, l in enumerate([1, 2, 3, 4]):
        hot_slice = hot_labels[..., idx].numpy().astype(bool)
        l_mask = labels.squeeze() == l

        np.testing.assert_array_equal(hot_slice, l_mask)


def test_relabel_sequential():

    np.random.seed(25)
    labels = np.random.choice([-1, 0, 2, 3, 4, 5],
                              size=(10, 10, 1)).astype(np.int32)

    # already sequential labels
    sk_sequential_labels = sk_relabel_sequential(labels + 1)[0] - 1
    tf_sequential_labels = relabel_sequential(labels)
    assert set(np.unique(sk_sequential_labels)) == set(
        np.unique(tf_sequential_labels))

    # non sequential labels
    labels[labels == 2] = 0
    labels[labels == 4] = -1
    sk_sequential_labels = sk_relabel_sequential(labels + 1)[0] - 1
    tf_sequential_labels = relabel_sequential(labels)
    assert set(np.unique(sk_sequential_labels)) == set(
        np.unique(tf_sequential_labels))


def test__unbatched_embedding_center():

    _unbatched_label_to_hot = DummySpatialInstanceEmbeddingLoss(
    )._unbatched_label_to_hot
    _unbatched_embedding_center = DummySpatialInstanceEmbeddingLoss(
    )._unbatched_embedding_center

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


def test_InstanceEmbeddingLossBase():
    '''Checks that the reduction of _unbatched_loss ignores unlabeled entries'''
    class InstanceMeanIoUEmbeddingLoss(InstanceEmbeddingLossBase):
        def _unbatched_loss(self, packed):
            y_true, y_pred = packed
            y_true = tf.cast(y_true, tf.float32)

            return tf.math.reduce_mean(tf.abs(y_true - y_pred))

    yt = np.broadcast_to(
        np.arange(1, 11, dtype=np.float32)[:, None, None, None],
        (10, 10, 10, 1)).copy()
    yp = (yt + 1).astype(np.float32)

    loss = InstanceMeanIoUEmbeddingLoss()(yt, yp)
    np.testing.assert_almost_equal(loss, 1.)

    # perfect prediction for samples 0 and 5
    yp[0] = 1
    yp[5] = 6
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

    # background should be excluded
    yt[:] = 0
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


def test__InstanceMeanIoUEmbeddingLoss_margin():
    '''Checks that first prob<0.5 is ~ margin away from center in 1D case'''

    yp = np.arange(100, dtype=np.float32)[..., None]
    centers = np.array([[0]], dtype=np.float32)

    for margin in range(1, 20):
        loss_cls = InstanceMeanIoUEmbeddingLoss(margin=margin)

        center_dist = loss_cls._unbatched_embeddings_to_center_dist(
            yp, centers)
        probs = loss_cls._center_dist_to_probs(None, center_dist)
        first_negative = np.argwhere((probs.numpy() < 0.5).squeeze())[0, 0]

        assert first_negative == margin + 1 or first_negative == margin


@pytest.mark.parametrize(
    "intra_margin,inter_margin",
    [(3, 10), (1, 2), (0.1, 5.)],
)
def test_MarginInstanceEmbeddingLoss(intra_margin, inter_margin):

    margin_loss = MarginInstanceEmbeddingLoss(intra_margin, inter_margin)

    # random labels, 5 classes, batch size = 4
    np.random.seed(11)
    yt = np.random.choice(range(5), size=(4, 10, 10, 1)).astype(np.int32)
    # perfect embedding of size 10, more than inter_margin appart from each other
    yp_prefect = np.tile(yt, (1, 1, 1, 10)) * 1.1 * inter_margin
    yp_prefect = yp_prefect.astype(np.float32)
    loss_perfect = margin_loss(yt, yp_prefect)
    np.testing.assert_almost_equal(loss_perfect, 0.)

    # batch 1, 1d sample with 2 elements, single instance and embeddign of size 1
    yt = np.ones((1, 2, 1), dtype=np.int32)
    yp = np.array([[[1], [1]]], dtype=np.float32)
    np.testing.assert_almost_equal(margin_loss(yt, yp), 0.)

    yp = np.array([[[1], [1 + intra_margin]]], dtype=np.float32)
    np.testing.assert_almost_equal(margin_loss(yt, yp), 0.)

    yp = np.array([[[1], [1 + 2 * intra_margin]]], dtype=np.float32)
    np.testing.assert_almost_equal(margin_loss(yt, yp), 0.)

    yp = np.array([[[1], [1 + 2.1 * intra_margin]]], dtype=np.float32)
    assert margin_loss(yt, yp) > 0

    yp = np.array([[[1], [1 + 10 * intra_margin]]], dtype=np.float32)
    assert margin_loss(yt, yp) > 0

    # batch 1, 1d sample with 2 elements, 2 instances and embeddign of size 1
    yt = np.array([[[1], [2]]], dtype=np.int32)
    yp = np.array([[[1], [1]]], dtype=np.float32)
    assert margin_loss(yt, yp) > 0.

    yp = np.array([[[1], [1 + 0.5 * inter_margin]]], dtype=np.float32)
    assert margin_loss(yt, yp) > 0

    yp = np.array([[[1], [1 + 1. * inter_margin]]], dtype=np.float32)
    np.testing.assert_almost_equal(margin_loss(yt, yp), 0.)

    yp = np.array([[[1], [1 + 2. * inter_margin]]], dtype=np.float32)
    np.testing.assert_almost_equal(margin_loss(yt, yp), 0.)
