import csv
import os
import pandas as pd
from datetime import datetime

# Step 1: Define file paths
station_file_path = r"/raid5/jseo7/geos_data/station_270_updates_Feb_11_2025.csv"
input_folder = r"/raid5/jseo7/geos_data/output_local_model/"
output_folder = r"/raid5/jseo7/geos_data/output_local_model/combined/"

# Step 2: Initialize the dictionary
continent_dict = {}

# Step 3: Read the CSV file and populate the dictionary
with open(station_file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
        continent = row['Continent'].strip().lower()
        sitename = row['sitename'].strip()  # Assuming this column is still 'sitename' in the station file
        
        if continent not in continent_dict:
            continent_dict[continent] = set()
        
        continent_dict[continent].add(sitename)

# Step 4: Get today's date
today_date = datetime.today().strftime("%Y%m%d")

# Step 5: Process the file for today's date
combined_data = pd.DataFrame()  # Initialize an empty DataFrame for combining data

for continent, sitenames in continent_dict.items():
    input_file = os.path.join(input_folder, f"{today_date}_{continent}_pred.csv")
    
    if not os.path.exists(input_file):
        print(f"File missing for {today_date} for {continent}. Skipping...")
        continue
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter rows based on the sitenames for the continent
    filtered_df = df[df['Site_Name'].isin(sitenames)]  # Updated to use 'Site_Name'
    
    # Append the filtered data to the combined DataFrame
    combined_data = pd.concat([combined_data, filtered_df])

# Save the combined data to a new file
if not combined_data.empty:
    output_file = os.path.join(output_folder, f"{today_date}_local_combined_pred.csv")
    combined_data.to_csv(output_file, index=False)
else:
    print(f"No data to save for {today_date}")
