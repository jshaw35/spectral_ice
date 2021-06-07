from Common.imports import *

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy as cy

np.seterr(divide='ignore', invalid='ignore') # Fails to remove error message.

# Functions from Abisko 2019 examples
def masked_average(xa:xr.DataArray,
                   dim=None,
                   weights:xr.DataArray=None,
                   mask:xr.DataArray=None):
    """
    This function will average
    :param xa: dataArray
    :param dim: dimension or list of dimensions. e.g. 'lat' or ['lat','lon','time']
    :param weights: weights (as xarray)
    :param mask: mask (as xarray), True where values to be masked.
    :return: masked average xarray
    """
    #lest make a copy of the xa
    with xr.set_options(keep_attrs=True): # testing this
        xa_copy:xr.DataArray = xa.copy()

        if mask is not None:
            xa_weighted_average = __weighted_average_with_mask(
                dim, mask, weights, xa, xa_copy
            )
        elif weights is not None:
            xa_weighted_average = __weighted_average(
                dim, weights, xa, xa_copy
            )
        else:
            xa_weighted_average =  xa.mean(dim)

    return xa_weighted_average


def __weighted_average(dim, weights, xa, xa_copy):
    '''helper function for unmasked_average'''
    _, weights_all_dims = xr.broadcast(xa, weights)  # broadcast to all dims
    x_times_w = xa_copy * weights_all_dims
    xw_sum = x_times_w.sum(dim)
    x_tot = weights_all_dims.where(xa_copy.notnull()).sum(dim=dim)
    xa_weighted_average = xw_sum / x_tot
    return xa_weighted_average


# I want to retain the metadata
def __weighted_average_with_mask(dim, mask, weights, xa, xa_copy):
    '''helper function for masked_average'''
    _, mask_all_dims = xr.broadcast(xa, mask)  # broadcast to all dims
    xa_copy = xa_copy.where(np.logical_not(mask))
    if weights is not None:
        _, weights_all_dims = xr.broadcast(xa, weights)  # broadcast to all dims
        weights_all_dims = weights_all_dims.where(~mask_all_dims)
        x_times_w = xa_copy * weights_all_dims
        xw_sum = x_times_w.sum(dim=dim)
        x_tot = weights_all_dims.where(xa_copy.notnull()).sum(dim=dim)
        xa_weighted_average = xw_sum / x_tot
    else:
        xa_weighted_average = xa_copy.mean(dim)
    return xa_weighted_average

def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

def add_map_features(ax):
    '''
    Single line command for xarray plots
    '''
    gl = ax.gridlines()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5);
    ax.add_feature(cfeature.BORDERS, linewidth=0.5);
    gl.xlabels_top = False
    gl.ylabels_right = False
    

def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection':projection}, **kwargs)

def interpretNS(stringin):
    '''
    Interprets the string name of CALIOP longitudinally 
    averaged SLF by isotherm data and returns a weight 
    for the latitude range
    '''
    lat_range = stringin[11:-4]
    first = lat_range[:3]
    second = lat_range[-3:]
    
    if first[-1] is 'N':
        low = int(first[:2])
    else: low = -1 * int(first[:2])
    if second[-1] is 'N':
        high = int(second[:2])
    else: high = -1 * int(second[:2])
    avg_lat = np.abs(np.mean([low,high]))
    weight = np.cos(np.pi/180*avg_lat)
    
    return weight, min([low, high]), max([low,high])


def select_loc_to_pandas(dataset, coords):
    '''
    This function takes an xarray dataset and a (lat, lon) coordinate iterable.
    It selects the data for the location and returns a pandas datafram object.
    '''
    _xr_ds = xr.Dataset() # empty xarray Dataset
    for vals in dataset:
        _da = dataset[vals]        
        _da = _da.sel(lat=coords[0], lon=coords[1], method='nearest')
        _xr_ds[vals]=_da
    _df = _xr_ds.to_dataframe()
    return _df


def process_caliop(files, obs_dir):
    all_caliop = pd.DataFrame()
    weights = 0; avg = np.zeros(5);
    for file in files:
        _path = obs_dir + file # Get full file path
        _name = 'CALIOP_' + file[11:-4]   # Pick out latitude data from name
        _weight, _, _ = interpretNS(file)
        _slice = pd.read_table(_path, sep="\s+", names=['Isotherm', _name])
        all_caliop = pd.concat([all_caliop, _slice[_name]], axis=1, sort=False)

        # Do the averaging
        avg += _weight * _slice[_name]
        weights += _weight

    # Add the Isotherm colum and set it as an index
    all_caliop = pd.concat([all_caliop, _slice['Isotherm']], axis=1, sort=False)
    all_caliop = all_caliop.set_index('Isotherm')
    all_caliop['CALIOP Average'] = np.array(avg / weights)
    
    return all_caliop

def plot_slf_isotherms(ds, var=None, isovar=None):
    '''
    simple way to visualize SLF, first arg is the xarray ds
    optional second arg is the SLF variable as a string
    '''
    if isovar != None:
        isostr = isovar
    else: isostr = 'isotherms_mpc'
    
    if var != None:
        slf_isotm = ds[var]
    else:
        try: # bad fix!
            slf_isotm = ds['SLF_ISOTM_AVG']    
        except KeyError:
            slf_isotm = ds['CT_SLF_ISOTM_AVG']
    
    fig1, axes1 = plt.subplots(nrows=3,ncols=3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[20,10]);

    cmin_p = np.nanmin(slf_isotm)
    cmax_p = np.nanmax(slf_isotm)

    cmap_p = 'bwr'
    nlevels = 41
    cmap2 = plt.get_cmap(cmap_p)

    if cmin_p == cmax_p:
       cmax_p = cmax_p + 0.00001

    levels = np.linspace(cmin_p,cmax_p,nlevels)

    for data, ax in zip(slf_isotm, axes1.flatten()):
        iso = data[isostr].values - 273.15
        map = data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='bwr', 
                        robust=True, add_colorbar = False, levels=levels)

        ax.set_title('SLF at %s' % str(iso), fontsize=18)
        ax.coastlines()

    cb_ax = fig1.add_axes([0.325, 0.05, 0.4, 0.04])
    cbar = plt.colorbar(map, cax=cb_ax, extend='both', orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.set_xlabel('Supercooled Liquid Fraction', fontsize=16)

    fig1.suptitle('SLF distribution across isotherms', fontsize=28)

    return fig1

def add_weights(ds):
    '''
    Add variable to ds for weighting of lat,lon variables
    Update to only require existing "lat" and "lon" variables. 
    Adding mask argument.
    '''
    
    try:
        lat = ds['lat']
    except:
        lat = ds['latitude']
        
    try:
        lon = ds['lon']
    except:
        lon = ds['longitude']
    
    _ones = xr.ones_like(lon)
    _gw = np.cos(lat*np.pi/180)
    
    _weights = (_gw @ _ones) / _gw.sum() 
    _weights_norm = _weights / _weights.sum()
#     xr.testing.assert_allclose(_weights_norm.sum().values,1) # doesn't work for some reason

    # approach
#     ds['cell_weight'] = (_gw @ _ones) / _gw.sum()
    
    # this might be a better way to add the weights as a coordinate
    ds = ds.assign_coords(cell_weight=_weights_norm)
    
    return ds

def process_for_slf(in_path, out_vars):
    '''
    Add SLF-relevant variables to netcdf file
    return a xr dataset with just variables of interest
    '''

    '''    
try:
    _ds = xr.open_dataset('%s/%s.nc' % (run_dir,case))
except:
    _ds = xr.open_dataset('%s/atm/hist/%s.cam.h0.2000-01.nc' % (run_dir,case))
if (len(_ds['time']) > 1):
    try:
        ds = _ds.sel(time=slice('0001-04-01', '0002-03-01'))
    except:
        ds = _ds.sel(time=slice('2000-04-01', '2001-03-01'))
else:
    ds = _ds
ds = add_weights(ds) # still has time here

ds['CT_SLF'] = ds['CT_SLFXCLD_ISOTM']/ds['CT_CLD_ISOTM']
ct_slf_noresm = ds['CT_SLF']

ds['CT_SLF_ISOTM_AVG'] = ds['CT_SLF'].mean(dim = 'time', skipna=True)
'''

    model_dir, case, _ = in_path.split('/')
    
    try:
        ds = xr.open_dataset(in_path + '.nc')
    except:
        ds = xr.open_dataset('%s/%s/atm/hist/%s.cam.h0.2000-01.nc' % (model_dir, case, case))
    ds = add_weights(ds)

    # Create new variable by dividing out the cloud fraction near each isotherm
    ds['SLF_ISOTM'] = ds['SLFXCLD_ISOTM'] / ds['CLD_ISOTM']

    # Select dates after a 3 month wind-up and average slf, unless it is a monthlong test run
    if (len(ds['time']) > 1):
        try:
            ds['SLF_ISOTM_AVG'] = ds['SLF_ISOTM'].sel(time=slice('0001-04-01',
                                '0002-03-01')).mean(dim = 'time', skipna=True)
        except:
            ds['SLF_ISOTM_AVG'] = ds['SLF_ISOTM'].sel(time=slice('2001-04-01',
                                '2002-03-01')).mean(dim = 'time', skipna=True)
            
    else: 
        ds['SLF_ISOTM_AVG'] = ds['SLF_ISOTM'].mean(dim = 'time', skipna=True)
            
    ds_out = ds[out_vars]
    ds.close()
    
    return ds_out

def noresm_slf_to_df(ds, slf_files):
    '''
    Applies appropriate latitude masks to NorESM SLF based on CALIOP file names
    '''
    df = pd.DataFrame()

    df['Isotherm'] = ds['isotherms_mpc'].values - 273.15
    df['NorESM_Average'] = 100*masked_average(ds['SLF_ISOTM_AVG'], dim=['lat','lon'], weights=ds['cell_weight'])
    
    df['NorESM_Average_STD'] = 100*np.std(ds['SLF_ISOTM_AVG'], axis=(1,2))

    # Add each latitude range from NorESM, and the models stdev range
    for i in slf_files:
        _, _lowlat, _highlat = interpretNS(i)
        _mask = np.bitwise_or(ds['lat']<_lowlat, ds['lat']>_highlat)
        
        zone_mean = masked_average(ds['SLF_ISOTM_AVG'], dim=['lat','lon'], weights=ds['cell_weight'], mask=_mask)
        df['NorESM' + i[10:-4]] = 100*zone_mean

        # Add Standard Deviation
        df['NorESM' + i[10:-4] + '_STD'] = 100*np.std(ds['SLF_ISOTM_AVG'].sel(lat=slice(_lowlat,_highlat)), axis=(1,2)) 
        
        
    df = df.set_index('Isotherm')
    
    return df

def regress_1d(xdata, ydata, **kwargs):
    '''
    Returns an sklearn regression object trained on the passed data.
    Might be generalizable to higher dimensions.
    '''
    x = np.array(xdata).reshape(-1,1)
    y = np.array(ydata).reshape(-1,1)
    
    regressor = LinearRegression(**kwargs).fit(x, y)
    
    return regressor

# Weighting function from http://xarray.pydata.org/en/stable/examples/monthly-means.html
# Now handles NaNs (by min_count=1)
def season_mean(ds, calendar='standard'):
    # Make a DataArray of season/year groups
#     year_season = xr.DataArray(ds.time.to_index().to_period(freq='Q-NOV').to_timestamp(how='E'),
#                                coords=[ds.time], name='year_season')

    
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = xr.DataArray(get_dpm(ds.time.to_index(), calendar=calendar),
                                coords=[ds.time], name='month_length')

    # Calculate the weights by grouping by 'time.season'
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

    # Calculate the weighted average
    with xr.set_options(keep_attrs=True): # jks keep attributes
        return (ds * weights).groupby('time.season').sum(dim='time', min_count=1)


def leap_year(year, calendar='standard'):
    """Determine if year is a leap year"""
    leap = False
    if ((calendar in ['standard', 'gregorian',
        'proleptic_gregorian', 'julian']) and
        (year % 4 == 0)):
        leap = True
        if ((calendar == 'proleptic_gregorian') and
            (year % 100 == 0) and
            (year % 400 != 0)):
            leap = False
        elif ((calendar in ['standard', 'gregorian']) and
                 (year % 100 == 0) and (year % 400 != 0) and
                 (year < 1583)):
            leap = False
    return leap

def get_dpm(time, calendar='standard'):
    """
    return a array of days per month corresponding to the months provided in `months`
    """
    
    dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}
    
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar) and month == 2:
            month_length[i] += 1
    return month_length

def share_ylims(axes):
    '''
    For 1D plots. Finds the global max and min so plots share bounds and are easier 
    to interpret.
    '''
    try:
        axes = axes.flat # so single iteration works
    except:
        axes = axes
        
    ymin, ymax = axes[0].get_ylim() # initialize values
    for ax in axes[1:]:
        _ymin, _ymax = ax.get_ylim()
        if _ymin < ymin: 
            ymin = _ymin
        if _ymax > ymax: 
            ymax = _ymax
                
    for ax in axes:
        ax.set_ylim([ymin,ymax])
        
def to_png(file, filename, loc='/glade/u/home/jonahshaw/figures/'):
    '''
    Simple function for one-line saving.
    Saves to "/glade/u/home/jonahshaw/figures" by default
    '''
    output_dir = loc
    ext = 'png'
    full_path = '%s%s.%s' % (output_dir,filename,ext)

    if not os.path.exists(output_dir + filename):
        file.savefig(full_path,format = 'png', dpi = 200)
#         file.clf()
        
    else:
        print('File already exists, rename or delete.')
        
def average_and_wrap(da,wrap=True):
        '''
        Helper function for cloud_polar_plot
        '''
        dat = da.groupby('time.month').mean() # Create monthly averages
#         _dat = _dat.mean(['lat','lon'])#.values # Get np.array average, ! JKS use masked average

        dat2 = add_weights(dat)
        _weights = dat2['cell_weight']
            
        out_dat = masked_average(dat2, dim=['lat','lon'], weights=_weights)
        
        if wrap:
            out_dat = np.append(out_dat, out_dat[0]) # wrap
        
        return _dat
    
def mute_ax(ax):
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('')
    ax.legend().set_visible(False)
    
# def align_yaxis(ax1, v1, ax2, v2):
#     """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
#     _, y1 = ax1.transData.transform((0, v1))
#     _, y2 = ax2.transData.transform((0, v2))
#     inv = ax2.transData.inverted()
#     _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
#     miny, maxy = ax2.get_ylim()
#     ax2.set_ylim(miny+dy, maxy+dy)

def align_yaxis(ax1, ax2):
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax1.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)
    
    
def calculate(cntl,test):
    """
    Calculate Taylor statistics for making taylor diagrams.
    Works with masked array if masked with NaNs.
    """
    
    _cntl = add_weights(cntl)
       
    mask = np.bitwise_or(xr.ufuncs.isnan(cntl),xr.ufuncs.isnan(test)) # mask means hide
#     mask = np.bitwise_or(cntl == np.nan,test == np.nan) # mask means hide
    
    wgt = np.array(_cntl['cell_weight'])
#     wgt = wgt * mask # does this work since one or zero?
    print(np.nansum(wgt))
    wgt = np.where(~mask,wgt,np.nan) # erroring
    
    sumwgt = np.nansum(wgt) # this is probably where the error is. 
    
    
    print(np.nansum(wgt))
#     wgt.plot()
    
    # calculate sums and means
    # These weights are not masked, so their sum is too high. This should be fixed now.
    meantest = np.nansum(wgt*test)/sumwgt
    meancntl = np.nansum(wgt*cntl)/sumwgt

    # calculate variances
    stdtest = (np.nansum(wgt*(test-meantest)**2.0)/sumwgt)**0.5
    stdcntl = (np.nansum(wgt*(cntl-meancntl)**2.0)/sumwgt)**0.5

    # calculate correlation coefficient
    ccnum = np.nansum(wgt*(test-meantest)*(cntl-meancntl))
    ccdem = sumwgt*stdtest*stdcntl
    corr = ccnum/ccdem

    # calculate variance ratio
    ratio = stdtest/stdcntl

    # calculate bias
    bias = (meantest - meancntl)/np.abs(meancntl)
    #self.bias = meantest - meancntl
    # Calculate the absolute bias
    bias_abs = meantest - meancntl

    # calculate centered pattern RMS difference
    try:
        rmssum = np.nansum(wgt*((test-meantest)-(cntl-meancntl))**2.0)
        
    except:
        print('test: ',test.shape)
        print('meantest: ',meantest.shape)
        print('cntl: ',cntl.shape)
        print('meancntl: ',meancntl.shape)
        print(((test-meantest)-(cntl-meancntl)).shape)
        print(((test-meantest)-(cntl-meancntl)).lat)
        print(((test-meantest)-(cntl-meancntl)).lon)
    rmserr = (rmssum/sumwgt)**0.5
    rmsnorm = rmserr/stdcntl
    
#     return corr,ratio,bias,rmsnorm
    return bias,corr,rmsnorm,ratio,bias_abs


def dual_mask(da1,da2):
    '''
    Take in two dataarrays masked with Nans. Calculate a shared mask.
    Return both arrays with the shared mask.
    '''
    
    mask = np.bitwise_or(xr.ufuncs.isnan(da1),xr.ufuncs.isnan(da2)) # mask means hide
    
    da1_out = da1.where(~mask) # dual-masked arrays
    da2_out = da2.where(~mask) # 
    
    return da1_out,da2_out
    