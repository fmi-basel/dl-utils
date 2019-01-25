from keras import backend as K

from tensorflow import where, gather_nd
from keras.layers import multiply, add
from keras.layers import Lambda


def laplacian_loss(curvature=0., sampling=None):
    '''generate an l1-loss where Laplacian.... mask...
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
    
    def laplace2D(x):
        '''Compute the 2D laplacian of an array
        
        Notes
        -----
        see https://www.tensorflow.org/tutorials/non-ml/pdes
        '''
        
        laplace_kernel = K.constant([[0.5, 1.0, 0.5],
                                     [1.0, -6., 1.0],
                                     [0.5, 1.0, 0.5]],
                                      shape = [3, 3, 1, 1])
        y = K.depthwise_conv2d(x, laplace_kernel, (1, 1), padding='same')
        
        return y
        
    def laplace3D(x):
        '''
        '''
        # TODO
        # select based on input/sampling param
        # anisotropic sampling
        # https://en.wikipedia.org/wiki/Discrete_Laplace_operator
        # https://mathoverflow.net/questions/32026/how-to-fill-in-3-dimensional-laplacian-kernels
        pass

    def loss(y_true, y_pred):
        '''
        '''
        lapl = laplace2D(y_pred)
        
        # get indices
        bg_indices = where(K.equal(y_true, 0))
        # ~ bg_indices = where(K.less_equal(y_true, 0))
        border_indices = where(K.equal(y_true, -1))
        fg_indices = where(K.greater(y_true, 0))
        
        # normalization factor = number of indices in each class
        bg_normalizer = K.maximum(1, K.shape(bg_indices)[0])
        bg_normalizer = K.cast(bg_normalizer, K.floatx())
        border_normalizer = K.maximum(1, K.shape(border_indices)[0])
        border_normalizer = K.cast(border_normalizer, K.floatx())
        fg_normalizer = K.maximum(1, K.shape(fg_indices)[0])
        fg_normalizer = K.cast(fg_normalizer, K.floatx())
        
        # gather values for each class
        bg_pred = gather_nd(y_pred, bg_indices)
        bg_lapl = gather_nd(lapl, bg_indices)
        fg_pred = gather_nd(y_pred, fg_indices)
        fg_lapl = gather_nd(lapl, fg_indices)
        border_lapl = gather_nd(lapl, border_indices)
        
        # define losses
        # background=0
        loss_bg = K.abs(bg_pred-0.)
        loss_bg = K.sum(loss_bg)/bg_normalizer
        # background should be smooth: lapl=0
        loss_bg_lapl = K.abs(bg_lapl-0.)
        loss_bg_lapl= K.sum(loss_bg_lapl)/bg_normalizer
        # foreground>1
        loss_fg = K.clip(1.-fg_pred, 0., None)#K.abs(fg_pred-1.)#
        loss_fg = K.sum(loss_fg)/fg_normalizer
        # foreground should be concave down: lapl < 0
        loss_fg_lapl = K.abs(fg_lapl+curvature)#K.clip(fg_lapl+curvature, 0., None)
        loss_fg_lapl = K.sum(loss_fg_lapl)/fg_normalizer
        # touching object should be separated by a valley (concave up): lapl > 0
        loss_border_lapl = K.abs(border_lapl-curvature)#K.clip(-border_lapl+curvature, 0., None)
        loss_border_lapl = K.sum(loss_border_lapl)/border_normalizer
        
        return 0.25*loss_bg + 0.0*loss_bg_lapl + 0.25*loss_fg + 0.25*loss_fg_lapl + 0.25*loss_border_lapl
    return loss

def laplacian_smoothing_loss(min_curvature=0., sampling=None):
    ''' regression of distance transform with laplacian smoothing
    '''
    
    def laplace2D(x):
        '''Compute the 2D laplacian of an array
        
        Notes
        -----
        see https://www.tensorflow.org/tutorials/non-ml/pdes
        '''
        
        laplace_kernel = K.constant([[0.5, 1.0, 0.5],
                                     [1.0, -6., 1.0],
                                     [0.5, 1.0, 0.5]],
                                      shape = [3, 3, 1, 1])
        y = K.depthwise_conv2d(x, laplace_kernel, (1, 1), padding='same')
        
        return y
        
    def laplace3D(x):
        '''
        '''
        # TODO
        # select based on input/sampling param
        # anisotropic sampling
        # https://en.wikipedia.org/wiki/Discrete_Laplace_operator
        # https://mathoverflow.net/questions/32026/how-to-fill-in-3-dimensional-laplacian-kernels
        pass

    def loss(y_true, y_pred):
        '''
        '''  
        
        # TODO find out how to have 2 losses on the same output
        # temporary fix get back fg mask from distance transform
        # ONLY WORKS IF DISTANCE TRANSFORM IS NOT SMOOTHED BY GAUSSIAN
        labels = K.greater(y_true, 0)
        labels = K.cast(labels, K.floatx())
        
        loss_transform = K.mean(K.abs(y_pred - y_true))
        # ~ loss_transform = K.mean(K.square(y_pred - y_true))
        
        
        lapl = laplace2D(y_pred)
        
        # get indices
        bg_indices = where(K.equal(labels, 0))
        fg_indices = where(K.greater(labels, 0))
        
        # normalization factor = # indices in each class
        bg_normalizer = K.maximum(1, K.shape(bg_indices)[0])
        bg_normalizer = K.cast(bg_normalizer, K.floatx())
        fg_normalizer = K.maximum(1, K.shape(fg_indices)[0])
        fg_normalizer = K.cast(fg_normalizer, K.floatx())
        
        # gather values for each class
        bg_lapl = gather_nd(lapl, bg_indices)
        fg_lapl = gather_nd(lapl, fg_indices)
        
        # background should be smooth: lapl=0
        loss_bg_lapl = K.abs(bg_lapl-0.)
        loss_bg_lapl= K.sum(loss_bg_lapl)/bg_normalizer
        # foreground should be concave down: lapl < 0
        loss_fg_lapl = K.clip(fg_lapl+min_curvature, 0., None)#K.abs(fg_lapl+min_curvature)
        loss_fg_lapl = K.sum(loss_fg_lapl)/fg_normalizer
        
        return loss_transform  + loss_fg_lapl #+ loss_bg_lapl
    return loss
