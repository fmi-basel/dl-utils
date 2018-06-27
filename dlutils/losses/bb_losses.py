from keras import backend as K
from tensorflow import where, gather_nd


def masked_smooth_l1_loss(mask_val, sigma=1.0):
    '''create masked, smooth l1 loss for bounding box regression.

    Loss is only evaluated where target value is not equal to mask_val.

    Parameters
    ----------
    sigma : float
        sigma to adjust turning point of smooth l1 loss.
        Larger values of sigma pull the transition point closer to 0.

    Returns
    -------
    smooth_l1 : loss function
        Smooth L1 loss function.
    '''
    assert sigma > 0
    sigma_squared = sigma * sigma

    def smooth_l1(y_true, y_pred):
        '''
        '''
        regression = y_pred
        indices = where(K.not_equal(y_true, mask_val))

        regression = gather_nd(regression, indices)
        targets = gather_nd(y_true, indices)

        regression_diff = regression - targets
        regression_diff = K.abs(regression_diff)

        # smoothen.
        regression_loss = where(
            K.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * K.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared)

        normalizer = K.maximum(1, K.shape(indices)[0])
        normalizer = K.cast(normalizer, K.floatx())

        return K.sum(regression_loss) / normalizer

    return smooth_l1


def focal_loss(alpha=0.5, gamma=0.5):
    '''create weighted focal loss.

    Based on:

    [1] Lin et al. Focal loss for Dense Object Detection, arxiv 2018.

    Parameters
    ----------
    alpha : float, [0, 1]
        positive class weight.
    gamma : float, [0, ..]
        focussing exponent.

    

    '''

    def focal(y_true, y_pred):
        '''
        '''
        labels = y_true
        predictions = y_pred

        # compute the focal loss
        alpha_factor = K.ones_like(labels) * alpha
        alpha_factor = where(
            K.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = where(K.equal(labels, 1), 1 - predictions, predictions)
        focal_weight = alpha_factor * focal_weight**gamma

        cls_loss = focal_weight * K.binary_crossentropy(labels, predictions)

        # compute the normalizer: the number of positive anchors
        normalizer = where(K.equal(labels, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(1.0, normalizer)

        return K.sum(cls_loss) / normalizer

    return focal
