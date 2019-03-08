from dlutils.training.targets.location_map import generate_locationmap

import numpy as np
import pytest
from scipy.signal import sawtooth


map_params = [
    ((247,500), (51,51), (0.78,0.3), 'sine'),
    ((247,500), (48,20), (0.3,0.0), 'sine'),
    ((150,70,120), (70,70,70), (0.3,0.3,0.3), 'sine'),
    ((150,70,120), (48,20,30), (0.0,0.0,0.3), 'sine'),
    ((1,70,120), (48,20,30), (0.0,0.0,0.3), 'sine'),
    ((150,70,1), (48,20,30), (0.0,0.0,0.3), 'sine'),
    
    ((247,500), (51,51), (0.78,0.3), 'triangle'),
    ((247,500), (48,20), (0.3,0.0), 'triangle'),
    ((150,70,120), (70,70,70), (0.3,0.3,0.3), 'triangle'),
    ((150,70,120), (48,20,30), (0.0,0.0,0.3), 'triangle'),
    ((1,70,120), (48,20,30), (0.0,0.0,0.3), 'triangle'),
    ((150,70,1), (48,20,30), (0.0,0.0,0.3), 'triangle'),
    ]

@pytest.mark.parametrize(
    "shape, period, offset, function_type",map_params)
def test_locationmap_extrema(shape, period, offset, function_type):
    
    loc_map = generate_locationmap(shape, period, offset, function_type)
    
    assert loc_map.max() <= 1.
    assert loc_map.min() >= -1.
    
    if all( s >= p for s,p in zip(shape,period)):
        assert loc_map.max() == pytest.approx(1., abs=0.1)
        assert loc_map.min() == pytest.approx(-1., abs=0.1)
        

@pytest.mark.parametrize(
    "shape, period, offset, function_type",map_params)
def test_locationmap_periodicity(shape, period, offset, function_type):
    
    loc_map = generate_locationmap(shape, period, offset, function_type)
    
    assert loc_map.max() <= 1.
    assert loc_map.min() <= 1.
    
    ndim = len(shape)
    for dim in range(ndim):
        for n in range(2*period[dim],shape[dim], period[dim]):
            idxs_a = [slice(None)]*ndim
            idxs_a[dim] = slice(n-period[dim],n)
            idxs_a = tuple(idxs_a)
            
            idxs_b = [slice(None)]*ndim
            idxs_b[dim] = slice(n-2*period[dim],n-period[dim])
            idxs_b = tuple(idxs_b)
            
            np.testing.assert_almost_equal(loc_map[idxs_a], loc_map[idxs_b], decimal=3)
            
            
@pytest.mark.parametrize(
    "shape, period, offset, function_type",[
    ((247,500), (51,51), (0.0,0.0), 'sine'),
    ((247,500), (48,20), (0.0,0.0), 'sine'),
    ((150,70,120), (70,70,70), (0.0,0.0,0.0), 'sine'),
    ((150,70,120), (48,20,30), (0.0,0.0,0.0), 'sine'),
    ((1,70,120), (48,20,30), (0.0,0.0,0.0), 'sine'),
    ((150,70,1), (48,20,30), (0.0,0.0,0.0), 'sine'),
    
    ((247,500), (51,51), (0.0,0.0), 'triangle'),
    ((247,500), (48,20), (0.0,0.0), 'triangle'),
    ((150,70,120), (70,70,70), (0.0,0.0,0.0), 'triangle'),
    ((150,70,120), (48,20,30), (0.0,0.0,0.0), 'triangle'),
    ((1,70,120), (48,20,30), (0.0,0.0,0.0), 'triangle'),
    ((150,70,1), (48,20,30), (0.0,0.0,0.0), 'triangle'),
    ])    
def test_locationmap_function(shape, period, offset, function_type):
    
    loc_map = generate_locationmap(shape, period, offset, function_type)[...,0]

    
    ndim = len(shape)
    for dim in range(ndim):
        idxs = [slice(0,1)]*ndim
        idxs[dim] = slice(None)
        idxs = tuple(idxs)
        
        edge = loc_map[idxs].squeeze()
        if all(s>0 for s in edge.shape):
            x = np.asarray(range(shape[dim]))
            w = 2*np.pi/period[dim]
            phase = offset[dim]*2*np.pi
            
            if function_type == 'sine':
                f_edge = np.sin(x*w+phase)/ndim
                
            elif function_type == 'triangle':
                f_edge = sawtooth(x*w+phase, 0.5) /ndim
                f_edge = f_edge - (ndim-1)/ndim # sawtooth(0) = -1 on other dims
                
                
            np.testing.assert_almost_equal(edge,f_edge, decimal=3)
    
    
@pytest.mark.parametrize(
    "shape, period, offset, function_type",[
    ((247,500), 51, 0.0, 'sine'),
    ((247,500), (51), (0.0), 'sine'),
    ((247,500), [51], [0.0], 'sine'),
    ((247,500), [51,45], [0.0,0.2], 'sine'),
    ((247,500), np.asarray([51,45]), np.asarray([0.0,0.2]), 'sine'),
    ])     
def test_locationmap_arguments(shape, period, offset, function_type):

    loc_map = generate_locationmap(shape, period, offset, function_type)


if __name__ == '__main__':
    test_locationmap_extrema(shape=(45,45), period=10, offset=0, function_type='sine')
