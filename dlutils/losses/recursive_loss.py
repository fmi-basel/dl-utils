import tensorflow as tf
from tensorflow.keras import backend as K


def recursive_loss(loss_fun, alpha=0.4):
    '''Applies the given loss function to intermediate outputs of a
    recurrent network as described in Recurrent U-Net.
    
    Args:
        loss_fun: a keras API compatible loss function returning a reduced output
        alpha: alpha <1 monotonically increases the weight of later outputs 
             and inversely for alpha >1 
    
    Notes:
    A weighted average of iterations' losses is returned instead of weighted sum as in original paper
    
    Wang, Wei, et al. "Recurrent U-Net for Resource-Constrained 
    Segmentation." arXiv preprint arXiv:1906.04913 (2019).
    
    https://arxiv.org/pdf/1906.04913.pdf
    '''
    def loss(y_true, y_preds):
        '''
        Args:
            y_true: target tensor
            y_preds: stack of intermediate outputs (axis=0) (i.e. stack of y_pred)
        '''

        n_recursion = K.shape(y_preds)[0]
        wt = alpha**K.cast(n_recursion - (tf.range(n_recursion) + 1),
                           tf.float32)
        wt = wt / tf.math.reduce_sum(
            wt
        )  # normalize so that loss magnitude does not depend on number of iterations

        # NOTE: special handling
        # keras model.fit somehow squeezes the last dimension when outputs has an extra dim???
        # special handling required when channel size = 1
        if len(y_true.shape) == len(y_preds.shape):
            y_preds = y_preds[..., None]

            # check that y_true/y_preds batch and spatial dimensions match
            tf.Assert(
                tf.reduce_all(
                    tf.equal(tf.shape(y_true)[:-1],
                             tf.shape(y_preds)[1:-1])),
                [tf.shape(y_true), tf.shape(y_preds)])

        def partial_loss_fun(y_pred):
            return loss_fun(y_true, y_pred)

        loss = tf.map_fn(partial_loss_fun, y_preds, tf.float32)

        return K.sum(wt * loss)

    return loss
