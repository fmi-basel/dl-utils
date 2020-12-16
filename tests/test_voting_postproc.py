import pytest
import tensorflow as tf
import numpy as np

from dlutils.postprocessing.voting import count_votes, embeddings_to_labels, seeded_embeddings_to_labels


def test_3D_count_votes():
    '''Tests 3D Hough voting'''

    np.random.seed(42)

    embeddings = np.zeros((20, 20, 20, 3), dtype=np.float32)
    embeddings[2:5, 8:12, 13:19] = [3, 9, 15]
    embeddings[10:13, 1:4, 2:5] = [11, 3, 2]

    votes = count_votes(embeddings[embeddings[..., 0] > 0],
                        (20, 20, 20)).numpy()

    assert votes[3, 9, 15] == 72
    assert votes[11, 3, 2] == 27
    assert votes.sum() == 99


def test_2D_count_votes():
    '''Tests 2D Hough voting'''

    np.random.seed(42)

    embeddings = np.zeros((20, 20, 2), dtype=np.float32)
    embeddings[2:5, 8:12] = [3, 9]
    embeddings[10:13, 1:4] = [11, 3]

    votes = count_votes(embeddings[embeddings[..., 0] > 0], (20, 20)).numpy()

    assert votes[3, 9] == 12
    assert votes[11, 3] == 9
    assert votes.sum() == 21


def test_embeddings_to_labels():
    '''Test conversion of spatial embeddings to labels using a Hough voting scheme'''

    np.random.seed(42)

    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 30:70] = True
    mask[80:90, 80:90] = True

    embeddings = np.zeros((100, 100, 2), dtype=np.float32)
    embeddings[10:20, 30:70] = [12, 49]
    embeddings[80:90, 80:90] = [85, 82]
    embeddings += np.random.rand(100, 100, 2) * 2

    labels = embeddings_to_labels(embeddings,
                                  mask,
                                  peak_min_distance=10,
                                  spacing=1.,
                                  min_count=5).numpy()

    assert np.all(labels[10:20, 30:70] == 1) or np.all(labels[10:20,
                                                              30:70] == 2)
    assert np.all(labels[80:90, 80:90] == 1) or np.all(labels[80:90,
                                                              80:90] == 2)
    assert np.all(labels[~mask] == 0)
    assert np.unique(labels).tolist() == [0, 1, 2]


def test_seeded_embeddings_to_labels():
    '''Test conversion of spatial embeddings to labels using user-provided seeds'''

    annot = np.zeros((100, 100), dtype=np.uint32)
    annot[10:40, 25:50] = 1
    annot[60:80, 10:30] = 2
    annot[20:65, 80:85] = 3
    mask = (annot > 0)

    centers_lut = np.array([[0, 0], [20, 30], [70, 20], [40, 80]])
    embeddings = centers_lut[annot]
    centers = centers_lut[1:]

    labels = seeded_embeddings_to_labels(embeddings.astype(np.float32),
                                         fg_mask=mask,
                                         seeds=centers,
                                         dist_threshold=None)
    labels = labels.numpy().astype(np.uint32)
    assert np.all(annot == labels)

    # split embeddings of label 3, keep same seeds ######################
    embeddings[50:65, 80:85] = np.array([[60, 80]])

    labels = seeded_embeddings_to_labels(embeddings.astype(np.float32),
                                         fg_mask=mask,
                                         seeds=centers,
                                         dist_threshold=None)
    labels = labels.numpy().astype(np.uint32)
    assert np.all(annot == labels)

    labels = seeded_embeddings_to_labels(embeddings.astype(np.float32),
                                         fg_mask=mask,
                                         seeds=centers,
                                         dist_threshold=0.)
    labels = labels.numpy().astype(np.uint32)
    assert not np.all(annot == labels)
    annot[50:65, 80:85] = 0
    assert np.all(annot == labels)


# draft testing inclusion of postprocessing steps as part of a model ######


def build_simple_model():
    ''''''
    input = tf.keras.layers.Input(batch_shape=(None, None, None, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')(input)
    embeddings = tf.keras.layers.Conv2D(2, kernel_size=1, name='embeddings')(x)
    classes = tf.keras.layers.Conv2D(2, kernel_size=1,
                                     name='semantic_classes')(x)
    return tf.keras.models.Model(inputs=input, outputs=[embeddings, classes])


class PostprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, peak_min_distance, spacing, min_count, *args, **kwargs):
        # a quick and dirty implementation
        super().__init__(self, *args, **kwargs)
        self.peak_min_distance = peak_min_distance
        self.spacing = spacing
        self.min_count = min_count

    def call(self, input):
        embeddings, fg_mask = input
        labels = embeddings_to_labels(embeddings,
                                      fg_mask,
                                      peak_min_distance=self.peak_min_distance,
                                      spacing=self.spacing,
                                      min_count=self.min_count)
        return labels


def add_voting_postprocessing(model,
                              peak_min_distance,
                              spacing=1.,
                              min_count=5):
    '''Appends the instance segmentation post-processing step to the given model.
    assumes batch_size == 1
    '''

    embeddings, classes = model.outputs
    fg_mask = tf.argmax(classes, axis=-1, output_type=tf.int32)[0]
    embeddings = embeddings[0]

    labels = PostprocessingLayer(peak_min_distance=peak_min_distance,
                                 min_count=min_count,
                                 spacing=spacing)([embeddings, fg_mask])

    return tf.keras.models.Model(inputs=model.inputs,
                                 outputs=[labels],
                                 name=model.name)


def test_build_inference_model():
    '''test post processing as part of a model (draft for future reference).
    
    To work with keras API, the post processing must be wrapped in a keras Layer
    '''

    model = build_simple_model()
    # doesnt throw anymore at build time.
    inference_model = add_voting_postprocessing(model, peak_min_distance=5)

    # doesnt throw at runtime either
    some_input = np.random.randn(1, 32, 32, 1)
    labels = inference_model.predict(some_input)

    # and it seems to be able to recompile for other sizes, too.
    some_other_input = np.random.randn(1, 64, 64, 1)
    labels = inference_model.predict(some_other_input)
