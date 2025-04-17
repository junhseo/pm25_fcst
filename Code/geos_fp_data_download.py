from datetime import date, datetime
from netCDF4 import Dataset
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy
import os
import numpy as np
import numpy.ma as ma
import matplotlib
import glob
import wget
from datetime import timedelta, date
import matplotlib.ticker as ticker
import matplotlib.patheffects as PathEffects
from matplotlib.ticker import AutoMinorLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
import warnings
warnings.filterwarnings('ignore')

now = datetime.now()
print('Running on ',  now.strftime("%d/%m/%Y %H:%M:%S"))
today = date.today()
date_org = today.strftime("%Y%m%d")
print(f'Today is {date_org}')
next_day = today + timedelta(1)
next_day = next_day.strftime("%Y%m%d")
print(f'Next day is {next_day}')
the_next_day = today + timedelta(2)
the_next_day = the_next_day.strftime("%Y%m%d")
print(f'The next day is {the_next_day}')
d = today.strftime("%d")
m = today.strftime("%m")
y = today.strftime("%Y")

if not os.path.exists('/raid5/jseo7/geos_data/' + date_org + '/'):
  os.mkdir('/raid5/jseo7/geos_data/' + date_org + '/')
if not os.path.exists('/raid5/jseo7/geos_data/' + date_org + '/aer/'):
  os.mkdir('/raid5/jseo7/geos_data/' + date_org + '/aer/')
if not os.path.exists('/raid5/jseo7/geos_data/' + date_org + '/slv/'):
  os.mkdir('/raid5/jseo7/geos_data/' + date_org + '/slv/')
file_savepath = '/raid5/jseo7/geos_data/' + date_org + '/'
aer_savepath = '/raid5/jseo7/geos_data/' + date_org + '/aer/'
slv_savepath = '/raid5/jseo7/geos_data/' + date_org + '/slv/'


endpart = ['_0130.V01.nc4', '_0430.V01.nc4', '_0730.V01.nc4', '_1030.V01.nc4', '_1330.V01.nc4', '_1630.V01.nc4', '_1930.V01.nc4', '_2230.V01.nc4', '_0130.V01.nc4', '_0430.V01.nc4', '_0730.V01.nc4', '_1030.V01.nc4',
           '_1330.V01.nc4', '_1630.V01.nc4', '_1930.V01.nc4', '_2230.V01.nc4', '_0130.V01.nc4', '_0430.V01.nc4', '_0730.V01.nc4', '_1030.V01.nc4', '_1330.V01.nc4', '_1630.V01.nc4', '_1930.V01.nc4', '_2230.V01.nc4']
aer_filelist = []
slv_filelist = []

for i in range(0, 24):
    if i < 8:
        date_mod = date_org
        aergne = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y'+y+'/M'+m + \
            '/D'+d+'/H00/GEOS.fp.fcst.tavg3_2d_aer_Nx.' + \
            str(date_org)+'_00+'+date_mod+endpart[i]
        slvgne = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y'+y+'/M'+m + \
            '/D'+d+'/H00/GEOS.fp.fcst.tavg1_2d_slv_Nx.' + \
            str(date_org)+'_00+'+date_mod+endpart[i]
        aer_filelist.append(aergne)
        slv_filelist.append(slvgne)
    elif i < 16:
        date_mod = next_day
        aergne = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y'+y+'/M'+m + \
            '/D'+d+'/H00/GEOS.fp.fcst.tavg3_2d_aer_Nx.' + \
            str(date_org)+'_00+'+date_mod+endpart[i]
        slvgne = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y'+y+'/M'+m + \
            '/D'+d+'/H00/GEOS.fp.fcst.tavg1_2d_slv_Nx.' + \
            str(date_org)+'_00+'+date_mod+endpart[i]
        aer_filelist.append(aergne)
        slv_filelist.append(slvgne)
    else:
        date_mod = the_next_day
        aergne = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y'+y+'/M'+m + \
            '/D'+d+'/H00/GEOS.fp.fcst.tavg3_2d_aer_Nx.' + \
            str(date_org)+'_00+'+date_mod+endpart[i]
        slvgne = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y'+y+'/M'+m + \
            '/D'+d+'/H00/GEOS.fp.fcst.tavg1_2d_slv_Nx.' + \
            str(date_org)+'_00+'+date_mod+endpart[i]
        aer_filelist.append(aergne)
        slv_filelist.append(slvgne)

for fname in aer_filelist:
    if not os.path.exists(aer_savepath + fname[-62:]):
        print(f'\nDownload..{fname}')
        wget.download(fname, out=aer_savepath)
for fname in slv_filelist:
    if not os.path.exists(slv_savepath + fname[-62:]):
        print(f'\nDownload..{fname}')
        wget.download(fname, out=slv_savepath)