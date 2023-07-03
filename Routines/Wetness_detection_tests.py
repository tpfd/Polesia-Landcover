import sys
import ee
ee.Initialize()
sys.path.append("C:/Users/tdow214/OneDrive - The University of Auckland/Documents/GitHub/Polesia-Landcover/Routines/")
from Wetness_functions import *


base_dir = 'C:/Users/tdow214/OneDrive - The University of Auckland/Documents/Projects/WetEagles'
fp_out_test_results = f"{base_dir}/wetness_test_results.csv"

# Set locations and dates to be tested on
test_locs = [(30.494743, 50.99985, '2018-06-01', 'Lake'),
             (30.371750, 51.053743, '2019-08-14', 'Braided streams and marsh'),
             (30.310367, 51.167084, '2018-08-25', 'Agroforestry regrowth'),
             (27.327553, 51.749287, '2018-05-01', 'Marsh and swamp')]

for i in test_locs:
    sentinel_out = get_sentinel_wetness(i[0], i[1], i[2], base_dir, i[3])
    new_row_dict = {'index': i[3],
                    'S1 Soil Moisture': sentinel_out['S1 Soil Moisture'],
                    'S1 total pixel count': sentinel_out['S1 total pixel count'],
                    'S1 Inundation count': sentinel_out['S1 Inundation count'],
                    'S2 total pixel count': sentinel_out['S2 total pixel count'],
                    'S2 Inundation count': sentinel_out['S2 Inundation count']}
    save_dict_to_csv(new_row_dict, fp_out_test_results)
