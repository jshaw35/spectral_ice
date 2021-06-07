# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
from datetime import datetime
import sys

import pandas as pd
import xarray as xr

import numpy as np
from numpy.polynomial.polynomial import polyfit

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

import cartopy as crt
import cartopy.crs as ccrs

import seaborn as sns
import importlib as il
import glob
import random

import pickle
import io

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

from collections import Iterable
from collections import deque
#import pyaerocom as pya

# libraries (additions from Jen)
from pathlib import Path
import pathlib
import numpy.linalg as LA
import timeit
from cartopy import config
import scipy.stats as stats # imports stats functions https://docs.scipy.org/doc/scipy/reference/stats.html
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import re # regular expressions

# Plotting a la Kay 2015:
# these modules aren't in my local conda environment, but they also aren't essential
try:
    import cmaps  # for NCL colormaps
except:
    pass
# For the interpolation curvilinear grids (anything on the globe!)
try:
    import xesmf as xe
except:
    pass
# For running command line operations through python
from subprocess import run

def load_and_reload():
    '''
    the code below automatically reload modules that
    have being changed when you run any cell.
    If you want to call in directly from a notebook you
    can use:
    Example
    ---
    >>> %load_ext autoreload
    >>> %autoreload 2
    '''
    from IPython import get_ipython

    try:
        _ipython = get_ipython()
        _ipython.magic('load_ext autoreload')
        _ipython.magic('autoreload 2')
    except:
        # in case we are running a script
        pass


load_and_reload()
