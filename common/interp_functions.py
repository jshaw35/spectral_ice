# Functions for interpolatings grids via python.
# Jonah Shaw, 2021/02/11

##################################################

# from Common.imports import *

import xarray as xr
import numpy as np
import xesmf as xe

##################################################

def interp_like2D(ds_in, target=None, method='bilinear',verbose=False,regridder=None):
    '''
    Wrapper function for xESMF regridder object.
    Takes an xarray dataarray or dataset and interpolates to
    the lat-lon grid of the target for all variables with lat-lon vars.
    Returns interpolated array and regridder object for repeated use.
    Either target or regridder have to be supplied in the function call.
    '''
    
    # Convert to dataset using name as variable if object is dataarray (fixes things for whatever reason)
#     if ds_in.name:
    try:
        name = ds_in.name
    except:
        name = 'VAR'
    
    if isinstance(ds_in,xr.DataArray):
        ds_in = ds_in.to_dataset(name=name)
    if isinstance(target,xr.DataArray):        
        target = target.to_dataset(name=name)
    
    # Get correct horizontal coordinate names (differs between models, obs)
    ds_lat,ds_lon = get_horiz_coords(ds_in, verbose=verbose)
    
    # Remove non-horizontal variables from the input dataset (returned as drops):
    try:
        ds_clean,drops = get_horiz_vars(ds_in, verbose=verbose)
    except:
        ds_clean = ds_in
        drops = {}
    
    # Rotate and wrap longitude appropriately:
    ds_clean.coords[ds_lon] = ds_clean.coords[ds_lon] % 360
    ds_clean = ds_clean.sortby(ds_clean.coords[ds_lon]) # Needed to avoid no data at 180 lon
    ds_clean = ds_clean.astype(dtype='float32',order='C') # Set to 'C-Contiguous' ordering to avoid an error from sortby
    
    # Create target grid and regridder if it wasn't been passed
    if not regridder:
        targ_lat,tar_lon = get_horiz_coords(target, verbose=verbose)
    
        # Create target lat-lon grid from target input
        targ_out = xr.Dataset({targ_lat: ([targ_lat], target[targ_lat].values),
                         tar_lon: ([tar_lon], target[tar_lon].values),}
                             )
        regridder = xe.Regridder(ds_clean, targ_out, method)
    
    # Regrid!
    try:
        ds_out = regridder(ds_clean)
    except:
        print('Regrid failed. Input was: ', ds_clean)
        return None
    
    # Add lat/lon free dropped variables back in (what about just lat or lon?)
    for i in drops:
        ds_out[i] = drops[i]
#         if not any(get_horiz_coords(i, verbose=verbose)):
#             print("I should add %s back in." % i)

    return ds_out,regridder
    
    
def get_horiz_coords(ds,verbose=False):
    '''
    Helper function for interpolation. Distinguishes between
    netCDF files using lat-lon and latitude-longitude conventions.
    '''
    
    if 'lat' in ds.coords:
        _lat = 'lat'
    elif 'latitude' in ds.coords:
        _lat = 'latitude'
    else:
        if verbose: print('Recognizable latitude coordinate not found.')
        _lat = False
    
    if 'lon' in ds.coords:
        _lon = 'lon'
    elif 'longitude' in ds.coords:
        _lon = 'longitude'
    else:
        if verbose: print('Recognizable longitude coordinate not found.')
        _lon = False
    
    return _lat,_lon
    
    
def get_horiz_vars(ds_in,verbose=False):
    '''
    Helper function for interp_like2D. 
    Removes non-lat/lon variables that would cause the interpolation to fail.
    Transposes variables so that any remaining coordinates come before lat/lon?
    ^See documentation of xESMF for explanation.
    '''
    ds = ds_in.copy()
    
    coords = ds.coords # Data coords to check against
    drops = {}
    for i in ds.variables:
        if i not in coords: # Cross-check against coordinates
            _horiz_coords = get_horiz_coords(ds[i])
            if False in _horiz_coords:
                if verbose: print('Will not interpolate "%s", horizontal coordinates not found.' % i)
                drops[i] = ds[i] # Save variable so it isn't lost to append later.
                ds = ds.drop(i)
            else: # This is where I would transpose if necessary, but that is not yet an issue.
                pass
    #             test_ds[i] = test_ds[i].transpose(...,_horiz_coords[0],_horiz_coords[1])

        else:
            if verbose: print('Excluded "%s" because found in coords: ' % i)
            
    return ds, drops


def interp_files(in_paths,out_paths,target,use_dask=False):
    '''
    This is a wrapper for interp_like2D to process multiple files.
    I am trying to build in parallizability, but dask isn't working for me yet.
    '''
    # Need to figure out how to use dask here.
#     https://docs.dask.org/en/latest/delayed-best-practices.html
    
    # Process the first file to create a regridder.
    target_ds = xr.open_dataset(target)
    regridder = interp_and_save(in_paths[0],out_paths[0],target=target_ds)
    target_ds.close()
    
    results = []
    for _in,_out in zip(in_paths[1:],out_paths[1:]):
        if use_dask:
            _task = dask.delayed(interp_and_save)(_in,_out,regridder=regridder)
            results.append(_task)
        else:
            interp_and_save(_in,_out,regridder=regridder)
            
    if use_dask: dash.compute(*results)
        

def interp_and_save(in_path,out_path, **kwargs):
    '''
    To be called by interp_files().
    Organized this way to work well with Dask.
    '''
    # kwargs will be the target or regridder and potentially the method as well
    
    ds = xr.open_dataset(in_path)
    
    ds_out,regrdr = interp_like2D(ds, **kwargs) 
    
    ds_out.to_netcdf(out_path)
    ds.close()
    
    return regrdr #?