import pandas as pd
from timezonefinder import TimezoneFinder
from datetime import datetime
import datetime as dt
import pytz

# Timezone function
tf = TimezoneFinder()

def calculate_aqi(pm25):
    if pd.isna(pm25):
        return np.nan  # Return NaN (blank) if PM2.5 value is missing
    elif pm25 <= 9.0:
        return linear_conversion(pm25, 0, 9.0, 0, 50)
    elif pm25 <= 35.4:
        return linear_conversion(pm25, 9.1, 35.4, 51, 100)
    elif pm25 <= 55.4:
        return linear_conversion(pm25, 35.5, 55.4, 101, 150)
    elif pm25 <= 125.4:
        return linear_conversion(pm25, 55.5, 125.4, 151, 200)
    elif pm25 <= 225.4:
        return linear_conversion(pm25, 125.5, 225.4, 201, 300)
    elif pm25 <= 500.0:
        return linear_conversion(pm25, 225.5, 500.0, 301, 500)
    else:
        return "Out of Range"

def linear_conversion(x, x0, x1, y0, y1):
    return round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0)
    
def convert_utc_to_local(utc_time, lat, lon):
    timezone_str = tf.timezone_at(lat=lat, lng=lon)
    local_tz = pytz.timezone(timezone_str)
    utc_dt = pd.to_datetime(utc_time).tz_localize('UTC')
    local_dt = utc_dt.astimezone(local_tz)
    return local_dt

def check_and_calculate(date, station, df):
    date_station_rows = df[(df['UTC_DATE'] == date) & (df['Station'] == station)]
    rows_count = len(date_station_rows)
    if rows_count == 8:
        avg_pm25 = date_station_rows['3HR_PM_CONC_CNN'].mean()
        return {
            'Station': station,
            'Site_Name': date_station_rows['Site_Name'].iloc[0],
            'UTC_DATE': date,
            'average_PM2.5': avg_pm25
        }
    else:
        print(f'Except case, the station id is {station}, date is {date} and the number of data {rows_count}')
        return None

def format_station_id(station):
    """
    Converts numerical station IDs into 12-digit zero-padded format.
    Keeps alphanumeric station IDs unchanged.
    Converts scientific notation (e.g., 1.24E+11) to integer format.
    """
    try:
        station = str(station).strip()
        if station.replace('.', '').isdigit():  # Check if purely numerical
            return f"{int(float(station))}".zfill(12)
        return station  # Keep original if alphanumeric
    except:
        return station  # Return original if conversion fails

def main():
    main_path = '/raid5/jseo7/geos_data/'
    today = dt.date.today()
    date_set = today.strftime("%Y%m%d")
    print('Running on ' + date_set)
    csv_path = main_path + 'output_local_model/combined/' + str(date_set) + '_local_combined_pred.csv'
    df = pd.read_csv(csv_path)

    result_data = []
    for (date, station), group in df.groupby([df['UTC_DATE'], 'Station']):
        result = check_and_calculate(date, station, group)
        if result:
            result_data.append(result)
    
    result_df = pd.DataFrame(result_data)
    result_df['DAILY_AQI'] = result_df['average_PM2.5'].apply(calculate_aqi)
    
    if 'Site_Name' in result_df.columns:
        result_df['Site_Name'] = result_df['Site_Name'].apply(lambda x: str(x).replace(" ", "_").replace("'", "_").replace('-', "_").replace(",", "_"))
    
    # Apply sorting in ascending order and formatting of Station IDs
    result_df['Station'] = result_df['Station'].astype(str)
    result_df['Station'] = result_df['Station'].apply(format_station_id)
    result_df.sort_values(by=['Station'], ascending=True, inplace=True)
    
    new_order_AQI = ['Station', 'Site_Name', 'UTC_DATE', 'DAILY_AQI', 'average_PM2.5']
    result_df = result_df[new_order_AQI]
    
    results_path = '/raid5/jseo7/geos_data/output_local_model/aqi/' + str(date_set) + '_local_pred_aqi.csv'
    result_df.to_csv(results_path, index=False)
    
    df['3HR_AQI'] = df['3HR_PM_CONC_CNN'].apply(calculate_aqi)
    
    expected_columns = [
        'Station', 'Site_Name', 'Agency_Name', 'Lat', 'Lon', 'SRadius',
        'MERRALat', 'MERRALon', 'IDXi', 'IDXj', 'PS', 'QV10m', 'Q500', 'Q850',
        'T10m', 'T500', 'T850', 'WIND', 'BCSMASS', 'DUSMASS25', 'OCSMASS',
        'SO2SMASS', 'SO4SMASS', 'SSSMASS25', 'NISMASS25', 'TOTEXTTAU', 'UTC_DATE',
        'UTC_TIME', 'trgdate', 'SED', 'SZA', 'Merra2', 'Merra2_ML',
        'Individual_Station_Model_Pred', '3HR_PM_CONC_CNN', '3HR_AQI', 'All_Station_Model_Pred'
    ]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = None  
    
    df = df[expected_columns]
    df.to_csv(csv_path, index=False)
    
if __name__ == "__main__":
    main()
