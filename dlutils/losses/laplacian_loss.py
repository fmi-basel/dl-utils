from keras import backend as K
import tensorflow as tf

from tensorflow import where, gather_nd, unique
from keras.layers import multiply, add
from keras.layers import Lambda

def laplacian_loss(min_curvature=0.0, sampling=None):
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
        
    def laplace3D(x, sampling):
        '''Compute the 3D laplacian of an array
        
        Notes
        -----
        depthwise_conv3d not available --> emulate with tf.scan
        https://github.com/tensorflow/tensorflow/issues/7278
        -----
        '''
        # TODO verify with separate test
        
        if sampling is None:
            sampling = (1.0, 1.0, 1.0)
        else:
            scaling = sampling[-1]
            sampling = tuple(x/scaling for x in sampling)
        
        w = tuple(1. / x**2 for x in sampling)
        cen = -2*sum(w)
            
        laplace_kernel = K.constant([[[0.0, 0.0,  0.0],
                                      [0.0, w[0], 0.0],
                                      [0.0, 0.0,  0.0]],
                                     
                                     [[0.0,  w[1], 0.0],
                                      [w[2], cen,  w[2]],
                                      [0.0,  w[1], 0.0]],
                                     
                                     [[0.0, 0.0,  0.0],
                                      [0.0, w[0], 0.0],
                                      [0.0, 0.0,  0.0]]],
                                     
                                      shape = [3, 3, 3, 1, 1])
                                      
        def flat_conv( _,elem):
            x = elem[0]
            kernel = elem[1]
    
            y = tf.nn.conv3d(x,kernel,[1,1,1,1,1],"SAME")
            return y
            
        kernel_scan = tf.transpose(laplace_kernel,[3,0,1,2,4])
        kernel_scan = tf.expand_dims(kernel_scan,axis=4)
        x_scan = tf.transpose(x,[4,0,1,2,3])
        x_scan = tf.expand_dims(x_scan,axis=-1)
        
        y = tf.scan(flat_conv, (x_scan,kernel_scan), initializer=tf.zeros_like(x) )
        
        y = tf.transpose(y,[1,2,3,4,0,5])
        y = tf.squeeze(y ,-1)

        return y

    def loss(y_true, y_pred):
        '''
        '''
        
        # extract pre-computed area normalization layer
        area_norm = y_true[...,1:]
        y_true = y_true[...,:1]
        
        if K.ndim(y_pred) == 5:
            lapl = laplace3D(y_pred, sampling)
        else:
            lapl = laplace2D(y_pred)
        
        # get indices
        bg_indices = where(K.equal(y_true, 0))
        separator_indices = where(K.equal(y_true, -1))
        edge_indices = where(K.equal(y_true, -2))
        
        # normalization factor = number of indices in each class
        bg_normalizer = K.maximum(1, K.shape(bg_indices)[0])
        bg_normalizer = K.cast(bg_normalizer, K.floatx())
        separator_normalizer = K.maximum(1, K.shape(separator_indices)[0])
        separator_normalizer = K.cast(separator_normalizer, K.floatx())
        edge_normalizer = K.maximum(1, K.shape(edge_indices)[0])
        edge_normalizer = K.cast(edge_normalizer, K.floatx())
       
        # gather values for each class
        bg_pred = gather_nd(y_pred, bg_indices)
        bg_lapl = gather_nd(lapl, bg_indices)
        separator_lapl = gather_nd(lapl, separator_indices)
        edge_pred = gather_nd(y_pred, edge_indices)
        edge_lapl = gather_nd(lapl, edge_indices)
        
        # define losses
        # background=0.
        loss_bg = K.abs(0.-bg_pred)
        loss_bg = K.sum(loss_bg)/bg_normalizer
        # background smooth: lapl=0
        loss_bg_lapl = K.abs(0.-bg_lapl)
        loss_bg_lapl = K.sum(loss_bg_lapl)/bg_normalizer
        # separators should be (concave up): lapl < 0
        loss_separator_lapl = K.clip(-separator_lapl, 0., None)
        loss_separator_lapl = K.sum(loss_separator_lapl)/separator_normalizer
        # border should be (concave down): lapl < 0
        loss_edge_lapl = K.clip(-edge_lapl, 0., None)
        loss_edge_lapl = K.sum(loss_edge_lapl)/edge_normalizer
        # border >= 0.
        loss_edge = K.clip(0.-edge_pred, 0., None)
        loss_edge = K.sum(loss_edge)/edge_normalizer
        
        # ~ # foreground > 1.0; lapl < 0; weigthed normalized per instance
        loss_fg = K.clip(1.-y_pred, 0., None)#K.abs(1.-y_pred)
        loss_fg = K.sum(loss_fg * area_norm)
        loss_fg_lapl = K.clip(lapl+min_curvature, 0., None)
        loss_fg_lapl = K.sum(loss_fg_lapl * area_norm)
        
        return loss_bg + loss_fg_lapl + loss_edge_lapl# + loss_fg
        # ~ return loss_bg + 0.*loss_bg_lapl + loss_fg + loss_fg_lapl + loss_separator_lapl + 0.*loss_edge_lapl + 0.*loss_edge
    return loss


def laplacian_smoothing_loss(min_curvature=1.0, sampling=None):
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
        
    def laplace3D(x, sampling):
        '''Compute the 3D laplacian of an array
        
        Notes
        -----
        depthwise_conv3d not available --> emulate with tf.scan
        https://github.com/tensorflow/tensorflow/issues/7278
        -----
        '''
        # TODO verify with separate test
        
        if sampling is None:
            sampling = (1.0, 1.0, 1.0)
        else:
            scaling = sampling[-1]
            sampling = tuple(x/scaling for x in sampling)
        
        w = tuple(1. / x**2 for x in sampling)
        cen = -2*sum(w)
            
        laplace_kernel = K.constant([[[0.0, 0.0,  0.0],
                                      [0.0, w[0], 0.0],
                                      [0.0, 0.0,  0.0]],
                                     
                                     [[0.0,  w[1], 0.0],
                                      [w[2], cen,  w[2]],
                                      [0.0,  w[1], 0.0]],
                                     
                                     [[0.0, 0.0,  0.0],
                                      [0.0, w[0], 0.0],
                                      [0.0, 0.0,  0.0]]],
                                     
                                      shape = [3, 3, 3, 1, 1])
                                      
        def flat_conv( _,elem):
            x = elem[0]
            kernel = elem[1]
    
            y = tf.nn.conv3d(x,kernel,[1,1,1,1,1],"SAME")
            return y
            
        kernel_scan = tf.transpose(laplace_kernel,[3,0,1,2,4])
        kernel_scan = tf.expand_dims(kernel_scan,axis=4)
        x_scan = tf.transpose(x,[4,0,1,2,3])
        x_scan = tf.expand_dims(x_scan,axis=-1)
        
        y = tf.scan(flat_conv, (x_scan,kernel_scan), initializer=tf.zeros_like(x) )
        
        y = tf.transpose(y,[1,2,3,4,0,5])
        y = tf.squeeze(y ,-1)

        return y

    def loss(y_true, y_pred):
        '''
        '''  
        
        # TODO find out how to have 2 losses on the same output
        # temporary fix get back fg mask from distance transform
        # ONLY WORKS IF DISTANCE TRANSFORM IS NOT SMOOTHED BY GAUSSIAN
        labels = K.greater(y_true, 0.5)
        labels = K.cast(labels, K.floatx())
        
        loss_transform = K.mean(K.abs(y_pred - y_true))
        # ~ loss_transform = K.mean(K.square(y_pred - y_true))
        
        if K.ndim(y_pred) == 5:
            lapl = laplace3D(y_pred, sampling)
        else:
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
