"""
Test location:
lon_test = 24.113395
lat_test = 52.6448816667
date_test = '2017-07-16'

get_averages_for_nests(lon_test, lat_test, date_test)
get_sentinel_wetness(lon_test, lat_test, date_test, base_dir)

"""
import sys
import ee
import pandas as pd

ee.Initialize()
sys.path.append("C:/Users/tdow214/OneDrive - The University of Auckland/Documents/GitHub/Polesia-Landcover/Routines/")
from Wetness_functions import *

# Set dirs
base_dir = 'C:/Users/tdow214/OneDrive - The University of Auckland/Documents/Projects/WetEagles'
fp_target_flight_pts = f"{base_dir}/point_data_movement_v1.csv"
fp_target_nest_pts = f"{base_dir}/point_data_nests.csv"
fp_out_data_movement = f"{base_dir}/extracted_data_movement.csv"

# Run movement data through earth engine and export to csv
eagle_movement_df = pd.read_csv(fp_target_flight_pts)
eagle_movement_df = eagle_movement_df.set_index([pd.to_datetime(eagle_movement_df['date']), 'event.id'], inplace=False)
for index, row in eagle_movement_df.iterrows():
    print(row['location.long'], row['location.lat'], row['date'])
    # Analyse the movement location points
    sentinel_out = get_sentinel_wetness(row['location.long'], row['location.lat'], row['date'], base_dir, None)
    sen_new_row_dict = {'index': index,
                        'S1 Soil Moisture': sentinel_out['S1 Soil Moisture'],
                        'S1 total pixel count': sentinel_out['S1 total pixel count'],
                        'S1 Inundation count': sentinel_out['S1 Inundation count'],
                        'S2 total pixel count': sentinel_out['S2 total pixel count'],
                        'S2 Inundation count': sentinel_out['S2 Inundation count']}
    save_dict_to_csv(sen_new_row_dict, fp_out_data_movement)

# Run the nests data through earth engine and export to csv
eagle_nest_df = pd.read_csv(fp_target_nest_pts)
eagle_nest_df = eagle_nest_df.set_index([pd.to_datetime(eagle_nest_df['date']), 'event.id'], inplace=False)
for index, row in eagle_nest_df.iterrows():
    # Analyse the nest locations
    nests_out = get_averages_for_nests(row['location.lat'], row['location.long'], row['date'])
    sen_new_row_dict = {'index': index,
                        'S1 Soil Moisture': sentinel_out['S1 Soil Moisture'],
                        'S1 total pixel count': sentinel_out['S1 total pixel count'],
                        'S1 Inundation count': sentinel_out['S1 Inundation count'],
                        'S2 total pixel count': sentinel_out['S2 total pixel count'],
                        'S2 Inundation count': sentinel_out['S2 Inundation count']}
    save_dict_to_csv(sen_new_row_dict, fp_out_data_movement)

print('Done data extraction!')
