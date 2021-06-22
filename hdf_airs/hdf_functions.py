import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from pyhdf.SD import SD, SDC

import xarray as xr
import pandas as pd



def hdf_to_netcdf(FILE_NAME):
    '''
    Read from CALIOP hdf file to netcdf format.
    Could take fields as an argument and handle errors better.
    '''
    
    datafields = ['Pressure_Mean', 'Pressure_Standard_Deviation',
              'Temperature_Mean', 'Temperature_Standard_Deviation',
              'Cloud_Free_Samples','Cloud_Samples','Cloud_Rejected_Samples',
              'Cloud_Accepted_Samples','Unknown_Cloud_Transparent_Samples',
              'Unknown_Cloud_Opaque_Samples', 'Water_Cloud_Transparent_Samples',
              'Water_Cloud_Opaque_Samples', 'Ice_Cloud_Transparent_Samples',
              'Ice_Cloud_Opaque_Samples',
             ]

    # datafields = ['Cloud_Accepted_Samples','Ice_Cloud_Opaque_Samples']

    # Open hdf file
    hdf = SD(FILE_NAME, SDC.READ)

    # Read geolocation datasets (and days observed)
    latitude = hdf.select('Latitude_Midpoint')
    lat = latitude[:]

    longitude = hdf.select('Longitude_Midpoint')
    lon = longitude[:]

    altitude = hdf.select('Altitude_Midpoint')
    alt = altitude[:]

    days_observed = hdf.select('Days_Of_Month_Observed')

    intdays = validate_field(days_observed) # Coded as a 32bit unsigned int
    ndays = vcalc_days(days_observed[:]) # Convert to the actual number of days

    var_dict = {}
    # Add to variables dictionary
    var_dict['Days_Of_Month_Observed'] = (('lat','lon'), intdays)
    var_dict['N_Days_Observed'] = (('lat','lon'), ndays)

    for field in datafields:
        # Read dataset.
        data3D = hdf.select(field)
        data = validate_field(data3D)

        var_dict[field] = (('lat','lon','alt'), data)

    # Create dataset
    ds = xr.Dataset(
        var_dict,
        coords={
            'lat': lat,
            'lon': lon,
            'alt': alt,  
        },
    )
    
    return ds


def calc_days(uint):
    "Return 1 if a>b, otherwise return 0"
    return f'{uint:b}'.count('1')
vcalc_days = np.vectorize(calc_days) # this is so cool!!


def validate_field(hdf_selected):
    
    data = hdf_selected[:] # selects all regardless of the shape?
    #^Tested with: all((hdf_selected[:,:] == hdf_selected[:]).flatten()), so good I think
    
    # Read attributes.
    attrs = hdf_selected.attributes(full=1)

    # Get fill values
    fva=attrs["fillvalue"]
    _FillValue = fva[0]

    # Get units
    ua=attrs["units"]
    units = ua[0]

    # Get valid range
    vra=attrs["valid_range"]
    vra_str = vra[0]

    # Valid attribute is string 'min...max'. Process from string
    smin, smax = vra_str.split("...")
    valid_min = float(smin)
    valid_max = float(smax)

    # Invalid values
    invalid = np.logical_or(data > valid_max,
                            data < valid_min)
    invalid = np.logical_or(invalid, data == _FillValue)
    # Invalids as nan
    data = np.where(invalid, np.nan, data)
    
    return data