import sys
import ee
ee.Initialize()
sys.path.append("C:/Users/tdow214/OneDrive - The University of Auckland/Documents/GitHub/Polesia-Landcover/Routines/")
from Wetness_functions import*

base_dir = 'C:/Users/tdow214/OneDrive - The University of Auckland/Documents/Projects/WetEagles'
fp_out_test_results = f"{base_dir}/wetness_test_results.csv"


# Set locations and dates to be tested on
test_locs = [(30.494743, 50.99985, '2018-06-01', 'Lake1'),
             (30.494743, 50.99985, '2017-06-06', 'Lake2'),
             (30.371750,  51.053743, '2017-06-21', 'Braided streams and marsh1'),
             (30.371750,  51.053743, '2019-08-10', 'Braided streams and marsh2'),
             (30.310367, 51.167084, '2018-07-09', 'Agroforestry regrowth1'),
             (30.310367, 51.167084, '2017-07-12', 'Agroforestry regrowth2'),
             (27.327553,  51.749287, '2018-05-01', 'Marsh and swamp1'),
             (27.327553,  51.749287, '2017-05-07', 'Marsh and swamp2')]

for i in test_locs:
    sentinel_out = get_sentinel_wetness(i[0], i[1], i[2])
    new_row_dict = {'index': i[3],
                    'S1 Soil Moisture': sentinel_out['S1 Soil Moisture'],
                    'S1 Inundation binary mean': sentinel_out['S1 Inundation binary mean'],
                    'S1 Inundation count': sentinel_out['S1 Inundation count'],
                    'S2 Inundation binary mean': sentinel_out['S2 Inundation binary mean'],
                    'S2 Inundation count': sentinel_out['S2 Inundation count']}
    save_dict_to_csv(new_row_dict, fp_out_test_results)

