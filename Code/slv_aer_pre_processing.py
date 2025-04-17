# import necessary modules
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import pandas as pd
import glob
import warnings
from datetime import date, timedelta, datetime
from pysolar.solar import get_altitude
import datetime as dt
import math
import pymysql
from pickle import load

warnings.filterwarnings("ignore")


def slv_aer_CSV_save(trgDate, mainPath='/raid5/jseo7/geos_data/'):
    def get_SZA(row):
        time_obj = row.datetime
        time_obj = time_obj.replace(tzinfo=dt.timezone.utc)
        lat, lon = row['Lat'], row['Lon']
        return 90 - get_altitude(lat, lon, time_obj)

    today = trgDate
    date_set = today.strftime("%Y%m%d")
    print('Running on ' + date_set)
    file_list = glob.glob(mainPath + date_set + '/aer/*.nc4')
    file_list.sort()
    Index_File = mainPath + 'station_270_updates_Feb_11_2025.csv'
    idx_input = pd.read_csv(Index_File)
    dates = list(map(lambda x: x[-33:-25], file_list))
    sts_i = idx_input['IDXi']
    sts_j = idx_input['IDXj']
    t2 = idx_input['stationid']
    t3 = idx_input['sitename']
    t4 = idx_input['agencyname']
    t5 = idx_input['Latitude']
    t6 = idx_input['Longitude']
    t7 = idx_input['SRadius']
    t8 = idx_input['MERRALat']
    t9 = idx_input['MERRALon']
    print(dates)

    for date in np.unique(dates):
        aer_date = glob.glob(mainPath + date_set + '/aer/*.' + date + '*.nc4')
        slv_date = glob.glob(mainPath + date_set + '/slv/*.' + date + '*.nc4')
        pair_dates = [(aer_file, slv_file)
                      for aer_file in aer_date for slv_file in slv_date if aer_file[-21:-8] == slv_file[-21:-8]]
        aer_date.sort()
        slv_date.sort()
        ps = []
        u10m = []
        v10m = []
        wind = []
        t10m = []
        t500 = []
        t850 = []
        q850 = []
        q500 = []
        qv10m = []
        utc_date = []
        utc_time = []
        bcs = []
        dus = []
        ocs = []
        so2 = []
        so4 = []
        sss = []
        tot = []
        nis = []
        station = []
        city = []
        country = []
        lat = []
        lon = []
        idxi = []
        idxj = []
        sradius = []
        merralat = []
        merralon = []

        for aer_file, slv_file in pair_dates:
            time = aer_file[-12:-8]
            date = slv_file[-21:-13]
            print(date + ' at ' + time)
            # # Read Location List for Delhi

            fh = Dataset(slv_file, mode='r')
            tps = fh.variables['PS'][:, :, :]
            tqv10m = fh.variables['QV10M'][:, :, :]
            tq500 = fh.variables['Q500'][:, :, :]
            tq850 = fh.variables['Q850'][:, :, :]
            tt10m = fh.variables['T10M'][:, :, :]
            tt850 = fh.variables['T850'][:, :, :]
            tt500 = fh.variables['T500'][:, :, :]
            xxx = ma.masked_invalid(tt850)
            xxx = xxx * 1000.0
            xxx = ma.round(xxx, decimals=5)
            tt850 = xxx.data
            tu10m = fh.variables['U10M'][:, :, :]
            tv10m = fh.variables['V10M'][:, :, :]
            fh.close()
            # # AER File
            fh = Dataset(aer_file, mode='r')
            tmax1 = fh.variables['BCSMASS'][:, :, :]
            tmax2 = fh.variables['DUSMASS25'][:, :, :]
            tmax3 = fh.variables['OCSMASS'][:, :, :]
            tmax4 = fh.variables['SO2SMASS'][:, :, :]
            tmax5 = fh.variables['SO4SMASS'][:, :, :]
            tmax6 = fh.variables['SSSMASS25'][:, :, :]
            tmax7 = fh.variables['TOTEXTTAU'][:, :, :]
            tmax8 = fh.variables['NISMASS25'][:, :, :]
            fh.close()

            for istation in range(0, len(sts_i)):
                if (t6[istation] < 180.0):
                    x = sts_i[istation]
                    y = sts_j[istation]
                # for ik in range(0, 24):
                #     time = 0.5 + ik
                ps.append(round(tps[0, x, y], 3))
                qv10m.append(round(tqv10m[0, x, y] * 1000., 5))
                q500.append(round(tq500[0, x, y] * 1000., 5))
                q850.append(tq850[0, x, y])
                t10m.append(round(tt10m[0, x, y], 3))
                if (tt850[0, x, y] > 400.):
                    tt850[0, x, y] = (tt10m[0, x, y]) - 8.0

                t500.append(round(tt500[0, x, y], 3))
                t850.append(round(tt850[0, x, y], 3))
                u10m.append(round(tu10m[0, x, y], 3))
                v10m.append(round(tv10m[0, x, y], 3))
                twind = np.round(
                    np.sqrt([tu10m[0, x, y] ** 2 + tv10m[0, x, y] ** 2]), decimals=3)[0]
                wind.append(twind)
                utc_date.append(date)
                utc_time.append(time)

                bcs.append(round(tmax1[0, x, y] * 1.0e9, 5))
                dus.append(round(tmax2[0, x, y] * 1.0e9, 5))
                ocs.append(round(tmax3[0, x, y] * 1.0e9, 5))
                so2.append(round(tmax4[0, x, y] * 1.0e9, 5))
                so4.append(round(tmax5[0, x, y] * 1.0e9, 5))
                sss.append(round(tmax6[0, x, y] * 1.0e9, 5))
                tot.append(round(tmax7[0, x, y], 3))
                nis.append(round(tmax8[0, x, y] * 1.0e9, 5))

                # ngelid.append(t1[istation])
                station.append(t2[istation])
                city.append(t3[istation])
                country.append(t4[istation])
                lat.append(t5[istation])
                lon.append(t6[istation])
                sradius.append(t7[istation])
                merralat.append(t8[istation])
                merralon.append(t9[istation])
                idxi.append(x)
                idxj.append(y)

        #  ngel_pm2.append(round(-1.0,3))
        csv_input = pd.DataFrame()
        # csv_input['NGELID']=ngelid
        csv_input['Station'] = station
        csv_input['Site_Name'] = city
        csv_input['Agency_Name'] = country
        csv_input['Lat'] = lat
        csv_input['Lon'] = lon
        csv_input['SRadius'] = sradius
        csv_input['MERRALat'] = merralat
        csv_input['MERRALon'] = merralon
        csv_input['Lon'] = lon
        csv_input['IDXi'] = idxi
        csv_input['IDXj'] = idxj

        csv_input['PS'] = ps
        csv_input['QV10m'] = qv10m
        csv_input['Q500'] = q500
        csv_input['Q850'] = q850
        csv_input['T10m'] = t10m
        csv_input['T500'] = t500
        csv_input['T850'] = t850
        csv_input['WIND'] = wind

        csv_input['BCSMASS'] = bcs
        csv_input['DUSMASS25'] = dus
        csv_input['OCSMASS'] = ocs
        csv_input['SO2SMASS'] = so2
        csv_input['SO4SMASS'] = so4
        csv_input['SSSMASS25'] = sss
        csv_input['NISMASS25'] = nis
        csv_input['TOTEXTTAU'] = tot

        csv_input['UTC_DATE'] = utc_date
        csv_input['UTC_TIME'] = utc_time

        csv_input.to_csv(mainPath + 'output/' + date_set + '.slv.aer.csv', index=False)


def main():
    path = '/raid5/jseo7/geos_data/'
    trg_date = date.today()
    slv_aer_CSV_save(trg_date, path)

if __name__ == "__main__":
    main()