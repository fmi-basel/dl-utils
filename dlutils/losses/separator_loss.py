from keras import backend as K
from tensorflow import where, gather_nd


def masked_separator_loss(mask_val):
    '''generate an l1-loss function that is only evaluated
    where the target doesnt equal mask_val.

    Parameters
    ----------
    mask_val : float
        Value in target map to be ignored.

    Returns
    -------
    loss : loss function
        Masked loss function.

    '''

    def loss(y_true, y_pred):
        '''
        '''
        indices = where(K.not_equal(y_true, mask_val))
        targets = gather_nd(y_true, indices)
        predictions = gather_nd(y_pred, indices)

        loss = K.abs(predictions - targets)

        normalizer = K.minimum(1, K.shape(indices)[0])
        normalizer = K.cast(normalizer, K.floatx())

        return K.sum(loss) / normalizer

    return loss
