"""
Main run script for wetness extraction.
"""
import sys
import ee
import pandas as pd
sys.path.append("C:/Users/tdow214/OneDrive - The University of Auckland/Documents/GitHub/Polesia-Landcover/Routines/")
from Wetness_functions import *

# Set dirs
ee.Initialize()
base_dir = 'C:/Users/tdow214/OneDrive - The University of Auckland/Documents/Projects/WetEagles'
fp_target_flight_pts = f"{base_dir}/all point data.csv"
fp_target_nest_pts = f"{base_dir}/nest locations.csv"
fp_out_data_movement = f"{base_dir}/extracted_data_movement.csv"
fp_out_nest = f"{base_dir}/extracted_nest_conditions.csv"

# Run the nests data through earth engine and export to csv
eagle_nest_df = pd.read_csv(fp_target_nest_pts)
eagle_nest_df.rename(columns={eagle_nest_df.columns[0]: "id"}, inplace=True)
print('Starting nest extraction...')
for index, row in eagle_nest_df.iterrows():
    # Analyse the nest locations
    new_index_in = str(int(row['year'])) + '_' + str(int(row['id']))
    get_averages_for_nests(row['latitude'],
                           row['longitude'],
                           row['year'],
                           row['radius_km'],
                           new_index_in,
                           fp_out_nest)
print('Nest extraction done!')

# Run movement data through earth engine and export to csv
eagle_movement_df = pd.read_csv(fp_target_flight_pts)
eagle_movement_df = eagle_movement_df.set_index([pd.to_datetime(eagle_movement_df['date']), 'event.id'], inplace=False)
print('Starting movement extraction...')
for index, row in eagle_movement_df.iterrows():
    # Analyse the movement location points
    print('Getting movement data for:', index, row['date'])
    sentinel_out = get_sentinel_wetness(row['location.long'],
                                        row['location.lat'],
                                        row['date'],
                                        base_dir,
                                        None)
    sen_new_row_dict = {'index': index,
                        'S1 Soil Moisture': sentinel_out['S1 Soil Moisture'],
                        'S1 total pixel count': sentinel_out['S1 total pixel count'],
                        'S1 Inundation count': sentinel_out['S1 Inundation count'],
                        'S2 total pixel count': sentinel_out['S2 total pixel count'],
                        'S2 Inundation count': sentinel_out['S2 Inundation count']}
    save_dict_to_csv(sen_new_row_dict, fp_out_data_movement)
print('Movement extraction done!')
