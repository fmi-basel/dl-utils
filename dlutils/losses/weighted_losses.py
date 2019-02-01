from keras import backend as K

def instance_normalized_l1_loss():
    '''Compute L1 loss with the first channel of y_true and weight the result with its second channel

    '''
    
    def loss(y_true, y_pred):
        '''
        '''
        # extract pre-computed area normalization channel
        area_norm = y_true[...,1:]
        y_true = y_true[...,0:1]
        
        loss = K.abs( y_pred-y_true )
        loss = K.sum(loss * area_norm)
        
        return loss
        
    return loss
