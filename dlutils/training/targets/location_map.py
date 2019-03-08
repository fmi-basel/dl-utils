import numpy as np
from .normalization_mask import generate_normalization_mask

def generate_locationmap(shape, period=50., offset=0., function_type='triangle'):
    '''Returns a periodic intensity map encoding image location
    
    Notes:
    ------
    Generates ndim+1 location maps with 1/4 period offset along each dimension.
    '''
    ndim = len(shape)
    if isinstance(period, (list,tuple)):
        if len(period) == 1:
            period = period*ndim
        elif len(period) != ndim:
            raise ValueError('wrong dimension of period argument, expected 1 or {}, got: {}'.format(ndim, len(period)) )
    elif isinstance(period, (float,int)):
        period = (period,)*ndim
        
    if isinstance(offset, (list,tuple)):
        if len(offset) == 1:
            offset = offset*ndim
        elif len(offset) != ndim:
            raise ValueError('wrong dimension of offset argument, expected 1 or {}, got: {}'.format(ndim, len(offset)) )
    elif isinstance(offset, (float,int)):
        offset = (offset,)*ndim
        
    known_functions = ['sine', 'triangle']
    if function_type not in known_functions:
        raise NotImplementedError(
                    'Function "{}" not supported, available options are {}'.
                    format(function_type, known_functions))
    
    period = np.asarray(period)
    offset = np.asarray(offset)
    amplitude = (1./ndim,)*ndim
        
    def sine_parametric_f(amplitude, period, offset):
        
        def f1d(x, amplitude, w, phase):
            return amplitude*np.sin(x*w+phase)
            
        w = 2*np.pi/period
        phase = offset*2*np.pi
        
        def f(*args):   
            return sum( f1d(*params) for params in zip(args,amplitude, w, phase))
        return f
        
    def triangle_parametric_f(amplitude, period, offset):
            
        def f1d(x, amplitude, period, offset):
            y = (x+period*offset) % period
            y_part2 = y > period/2
            y[y_part2] = period-y[y_part2]
            y = 4*y/period-1. # rescale to [-1,1]
            return y*amplitude
        
        def f(*args):   
            return sum( f1d(*params) for params in zip(args,amplitude, period, offset))
        return f
        
    if function_type == 'triangle':
        parametric_f = triangle_parametric_f
    elif function_type == 'sine':
        parametric_f = sine_parametric_f
    
    # separate dimensions    
    loc_map = np.zeros(shape=shape+(ndim+1,), dtype=np.float32)
    
    loc_map[...,0] = np.fromfunction(parametric_f(amplitude, period, offset), 
                                                  shape, dtype=np.float32)
                                                  
    for n in range(1,ndim+1):
        relative_offset = np.zeros_like(offset)
        relative_offset[n-1] = 0.25
        loc_map[...,n] = np.fromfunction(parametric_f(amplitude, period, offset + relative_offset ),
                                                      shape, dtype=np.float32)

    return loc_map


def generate_locationmap_target(segmentation, location_map):
    '''generate target having instance wise mean intensity of location_map:
    
    output channels:
    ---------
    0: instance wise mean of location_map
    1: instance normalization
    '''
        
    location_map_mean = np.zeros_like(location_map, dtype=np.float32)#-2. # background --> label= -2
    for label in np.unique(segmentation):
        if label > 0:
            indices = np.where(segmentation==label)
            for map_dim in range(location_map.shape[-1]):
                vals = location_map[...,map_dim][indices]
                location_map_mean[...,map_dim][indices] = vals.mean()

    normalization = generate_normalization_mask(segmentation)
    normalization = np.expand_dims(normalization, axis=-1)
    
    return np.concatenate([location_map_mean, normalization], axis=-1)

