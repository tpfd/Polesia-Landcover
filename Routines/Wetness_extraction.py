import sys
import ee
ee.Initialize()
sys.path.append("C:/Users/tdow214/OneDrive - The University of Auckland/Documents/GitHub/Polesia-Landcover/Routines/")
from Wetness_functions import*

# Set dirs and date range
base_dir = 'C:/Users/tdow214/OneDrive - The University of Auckland/Documents/Projects/WetEagles'
fp_target_ext = f"{base_dir}/Eagle_area.shp"
fp_target_pts = f"{base_dir}/point_data.csv"
years_to_map = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

print('Constructing date list for compositing...')
# Still using monthly composites as this is what the S1 flood frequency func expects
# Depending on results, may want to upgrade the flood func to do 'wetness' better (given that eagle data is more
# frequent than monthly).
date_list = generate_month_tuples(years_to_map)

print('Setting aoi..')
aoi = geemap.shp_to_ee(fp_target_ext)

print('Loading Eagle data points...')
gee_ready_points, raw_data = prepare_csv_for_gee(fp_target_pts, 3, 4)

print('Starting satellite data stack...')
image_collection = 'COPERNICUS/S1_GRD'
wet = compute_daily_wetness(image_collection, '2018-09-01', '2018-09-02', aoi)
wet_week = compute_weekly_wetness(image_collection, '2018-09-01', '2018-09-07', aoi)

print('Trying to visualise...')
visualize_image_collection(wet_week)
visualize_feature_collection(aoi)









