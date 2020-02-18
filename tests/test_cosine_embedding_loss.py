from dlutils.losses.embedding_loss import cosine_embedding_loss


from tensorflow.keras import backend as K
from tensorflow.keras import layers

import numpy as np
from sklearn.metrics import log_loss

import pytest


labels = np.transpose(np.asarray([
 [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 2., 2., 2., 2., 0., 0., 0.,]]))

aligned_orthogonal = np.transpose(np.asarray([
 [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1.,],
 [0., 0., 0., 0., 0., 1., 1., 3., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 2., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,]]))
 
not_aligned_orthogonal = np.transpose(np.asarray([
 [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1.,],
 [0., 0., 0., 0., 0., 1., 1., 3., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,]]))

aligned_not_orthogonal = np.transpose(np.asarray([
 [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1.,],
 [0., 0., 0., 0., 0., 1., 1., 3., 1., 1., 0., 0., 0., 1., 1., 1., 2., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,]]))

@pytest.mark.xfail
@pytest.mark.parametrize(
    'yt, yp, neighbor_distance, include_background, expected_loss',[
                [labels, aligned_orthogonal, 7, True, 0.],
                [labels, aligned_orthogonal, 1, True, 0.],
                [labels, aligned_orthogonal, 7, False, 0.],
                [labels, aligned_orthogonal, 1, False, 0.],
                
                [labels, not_aligned_orthogonal, 7, True, 0.0253734],
                [labels, not_aligned_orthogonal, 1, True, 0.0253734],
                [labels, not_aligned_orthogonal, 7, False, 0.0380603],
                [labels, not_aligned_orthogonal, 1, False, 0.0380603],
                
                [labels, aligned_not_orthogonal, 7, True, 1/3],
                [labels, aligned_not_orthogonal, 1, True, 0.],
                [labels, aligned_not_orthogonal, 7, False, 1.],
                [labels, aligned_not_orthogonal, 1, False, 0.],])
def test_cosine_embedding_loss(yt, yp, neighbor_distance, include_background, expected_loss):
    '''
    '''
    y_true = layers.Input(shape=(None,1,))
    y_pred = layers.Input(shape=(None,4,))
    loss_func = K.Function([y_true, y_pred], [cosine_embedding_loss(neighbor_distance=neighbor_distance, include_background=include_background)(y_true, y_pred)])

    loss = loss_func([yt[None, ...], yp[None, ...]])
    print(loss)
    np.testing.assert_almost_equal(loss, expected_loss)
    
