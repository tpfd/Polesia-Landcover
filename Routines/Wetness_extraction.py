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
from Wetness_functions import*

# Set dirs
base_dir = 'C:/Users/tdow214/OneDrive - The University of Auckland/Documents/Projects/WetEagles'
fp_target_flight_pts = f"{base_dir}/point_data_movement_v1.csv"
fp_target_nest_pts = f"{base_dir}/point_data_nests.csv"
fp_out_data_movement = f"{base_dir}/extracted_data_movement.csv"


# Run df through data collect from earth engine and export to csv for nests
eagle_nests_df = pd.read_csv(fp_target_nest_pts)
eagle_nests_df = eagle_nests_df.set_index([pd.to_datetime(eagle_nests_df['date']), 'event.id'], inplace=False)

for index, row in eagle_nests_df.iterrows():
    averages = get_averages_for_nests(row['location.long'], row['location.lat'], row['date'])
    save_data_to_csv(row, averages, fp_out_data)
    print('Appended:', index, averages)
print('Done nest data extraction!')


# Run movement data through earth engine and export to csv
eagle_movement_df = pd.read_csv(fp_target_flight_pts)
eagle_movement_df = eagle_movement_df.set_index([pd.to_datetime(eagle_movement_df['date']), 'event.id'], inplace=False)
for index, row in eagle_movement_df.iterrows():
    print(row['location.long'], row['location.lat'], row['date'])
    sentinel_out = get_sentinel_wetness(row['location.long'], row['location.lat'], row['date'])
    new_row_dict = {'index': index,
                    'S1_SM': sentinel_out['S1 Soil Moisture'],
                    'S1_binary': sentinel_out['S1 Surface Water binary'],
                    'S2_inundation':sentinel_out['S2 Inundation binary']}
    save_dict_to_csv(new_row_dict, fp_out_data_movement)
print('Done movement data extraction!')
