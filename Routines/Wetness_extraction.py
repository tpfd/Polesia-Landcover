import sys
import ee
import pandas as pd
ee.Initialize()
sys.path.append("C:/Users/tdow214/OneDrive - The University of Auckland/Documents/GitHub/Polesia-Landcover/Routines/")
from Wetness_functions import*

# Set dirs and date range
base_dir = 'C:/Users/tdow214/OneDrive - The University of Auckland/Documents/Projects/WetEagles'
fp_target_ext = f"{base_dir}/Eagle_area.shp"
fp_target_pts = f"{base_dir}/point_data.csv"
fp_out_data = f"{base_dir}/extracted_data.csv"

# Load list of eagle points into a multi-index df
eagle_df = pd.read_csv(fp_target_pts)
eagle_df = eagle_df.set_index([pd.to_datetime(eagle_df['date']), 'event.id'], inplace=False)

# Run df through data collect from earth engine and export to csv
for index, row in eagle_df.iterrows():
    averages = get_averages(row['location.long'], row['location.lat'], row['date'])
    save_data_to_csv(row, averages, fp_out_data)
    print('Appended:', index, averages)





