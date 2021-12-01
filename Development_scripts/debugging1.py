import sys
import os
from geemap import geemap
sys.path.append("C:/Users/tpfdo/OneDrive/Documents/GitHub/Polesia-Landcover/Routines/")
sys.path.append("/home/markdj/repos/Polesia-Landcover/Routines/")
from Classification_tools import RF_model_and_train, accuracy_assessment_basic, map_target_area
from Satellite_data_handling import create_data_stack_v2, fetch_sentinel1_flood_index_v1, fetch_sentinel2_v4 #fetch_sentinel2_v3


s2_params = {
    'CLOUD_FILTER': 60,  # int, max cloud coverage (%) permitted in a scene
    'CLD_PRB_THRESH': 40,  # int, 's2cloudless' 'probability' band value > thresh = cloud
    'NIR_DRK_THRESH': 0.15,  # float, if Band 8 (NIR) < NIR_DRK_THRESH = possible shadow
    'CLD_PRJ_DIST': 1,  # int, max distance [TODO: km or 100m?] from cloud edge for possible shadow
    'BUFFER': 50,  # int, distance (m) used to buffer cloud edges
    # 'S2BANDS': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    'S2BANDS': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']  # list of str, which S2 bands to return?
}
year = 2018

# this should work
fp_train_ext = '/home/markdj/Dropbox/artio/polesia/Classified/19_tiles/19.shp'
aoi = geemap.shp_to_ee(fp_train_ext)
year = str(year)
date_list = [(year + '-03-01', year + '-03-30'),
             (year + '-04-01', year + '-04-30'), (year + '-05-01', year + '-05-31'),
             (year + '-06-01', year + '-06-30'), (year + '-07-01', year + '-07-30'),
             (year + '-10-01', year + '-10-30')]
stack, max_min_values_output = create_data_stack_v2(aoi, date_list, year, None)

# this shouldnt work
fp_train_ext = '/home/markdj/Dropbox/artio/polesia/Classified/19_tiles/47.shp'
aoi = geemap.shp_to_ee(fp_train_ext)
year = str(year)
date_list = [(year + '-03-01', year + '-03-30'),
             (year + '-04-01', year + '-04-30'), (year + '-05-01', year + '-05-31'),
             (year + '-06-01', year + '-06-30'), (year + '-07-01', year + '-07-30'),
             (year + '-10-01', year + '-10-30')]
stack, max_min_values_output = create_data_stack_v2(aoi, date_list, year, None)


# sub function tests
year = str(year)
date_list = [(year + '-03-01', year + '-03-30'),
             (year + '-04-01', year + '-04-30'), (year + '-05-01', year + '-05-31'),
             (year + '-06-01', year + '-06-30'), (year + '-07-01', year + '-07-30'),
             (year + '-10-01', year + '-10-30')]
fp_train_ext = '/home/markdj/Dropbox/artio/polesia/Classified/19_tiles/47.shp'
aoi = geemap.shp_to_ee(fp_train_ext)


flood_index = fetch_sentinel1_flood_index_v1(aoi,
                                             str((int(year)-1))+'-01-01',
                                             year+'-12-01',
                                             smoothing_radius=100.0,
                                             flood_thresh=-13.0)
print(flood_index.bandNames().getInfo())

s2_stack = fetch_sentinel2_v4(aoi, date_list, s2_params)
print(s2_stack.bandNames().getInfo())


## test other issues in processing
from Processing_tools import tile_polygon
from Utilities import table_writer, get_max_acc, match_result_lengths, get_list_of_files, line_plot

base_dir = '/home/markdj/Dropbox/artio/polesia'
fp_target_ext = f"{base_dir}/whole_map.shp"
fp_export_dir = f"{base_dir}/Classified/"
process_size = 1.0

process_dir = tile_polygon(fp_target_ext, process_size, fp_export_dir, 'processing_areas/')
process_list = get_list_of_files(process_dir, ".shp")
process_list.sort()

tile_size = 0.1
for i in process_list:
    print('map_target_area(): Generating tiles for processing area', str(i) + '...')
    process_num = i.split('.')[0].split('/')[-1] #TODO: process_num seems to only include the
    print(process_num)
