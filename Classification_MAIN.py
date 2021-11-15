"""
This script contains the user settings and primary functionality of the Polesia landcover mapping tool.

You will need:
> A polygon of the area that covers all your training data (fp_train_extent)
> Your training data as a shapefile of points, with all classes assigned a number in series (*_training_fpath)
> A polygon that covers all of the area you want mapped (fp_target_ext)
> A valid Google Earth Engine account that has been set up on the device you are running this script on.

You may want:
> A presets .csv file (fp_settings_txt), if you do not have this make sure to set: optimisation_toggle = True.

Glossary:
> class_col_name: the variable name (column name) of your classes in the points shapefile.
> scale: output mapping scale in in meters.
> years_to_map: years in which to generate landcover maps.

Random tips:
> fp_target_ext and fp_train_ext can be the same or a different shapefile, but must be specified in each case.
> Training is always carried out on 2018.
> Big areas mean splitting into lots of processing areas and sub-tiles. This will take time to generate and classify.

"""
import sys
import os
from geemap import geemap
import shutil
#sys.path.append("C:/Users/tpfdo/OneDrive/Documents/GitHub/Polesia-Landcover/Routines/")
sys.path.append("/home/markdj/repos/Polesia-Landcover/Routines/")
from Training_data_handling import run_resample_training_data, load_sample_training_data,\
    training_data_size_optimize, trees_size_optimize
from Satellite_data_handling import stack_builder_run
from Utilities import line_plot, load_presets, generate_empty_preset_table, get_list_of_files
from Classification_tools import apply_random_forest, generate_RF_model, accuracy_assessment_basic, \
    accuracy_assessment_full
from Processing_tools import tile_polygon

"""
User defined variables
"""
# Processing and output options
training_data_resample_toggle = False
plot_toggle = True
performance_stats_toggle = True
advanced_performance_stats_toggle = False
optimisation_toggle = False
use_presets = True

# User settings
class_col_name = 'VALUE'
scale = 20
years_to_map = [2018]

base_dir = '/home/markdj/Dropbox/artio/polesia'
# base_dir = 'D:/tpfdo/Documents/Artio_drive/Projects/Polesia'

# File paths and directories for classification pipeline
fp_train_ext = f"{base_dir}/Project_area.shp"
fp_target_ext = f"{base_dir}/whole_map.shp"
fp_export_dir = f"{base_dir}/Classified/"
# fp_settings_txt = f"{base_dir}/RF_classif_setting.csv"   # TODO: should there really be fixed paths in here? either all paths in .csv, or MAIN, not both
fp_settings_txt = f"{base_dir}/RF_classif_setting_mdj.csv"
plot_dir = f"{base_dir}/Plots/"

# File paths to shapefiles of target class points and the export dir for their resampling
complex_training_fpath = f"{base_dir}/Training_data/Complex_swamp_points_v4.shp"
simple_training_fpath = f"{base_dir}/Training_data/Simple_swamp_points_v4.shp"
class_export_dir = f"{base_dir}/Training_data/"



"""
Classification pipeline
"""
# Set training data name template
points_name_simple = class_export_dir + "Simple_points_"
points_name_complex = class_export_dir + "Complex_points_"

# Build the data stack for model training
aoi = geemap.shp_to_ee(fp_train_ext)
stack, training_bands, max_min_values_training = stack_builder_run(aoi, 2018)

# Load preset classification parameters if so toggled
if use_presets:
    if os.path.isfile(fp_settings_txt):
        ('loading preset classification parameters from file...')
        fp_train_simple_points, fp_train_complex_points, trees_complex, training_complex, trees_simple, \
        training_simple = load_presets(fp_settings_txt)
    else:
        print('No preset available, set use_presets to False, optimisation_toggle to True and run again (or check '
              'that your presets path is correct).')
        sys.exit()
else:
    print('No presets being used, optimisation_toggle must be set to True...')

# Resample 'raw' training data points to same number of points per class if so toggled
if training_data_resample_toggle:
    print('Resampling raw training data...')
    run_resample_training_data(complex_training_fpath, plot_toggle,
                               'Complex classes', 'Complex',
                               class_export_dir, plot_dir, class_col_name)
    run_resample_training_data(simple_training_fpath, plot_toggle,
                               'Simple classes', 'Simple',
                               class_export_dir, plot_dir, class_col_name)
else:
    pass

# Run tree and training data size optimization if so toggled
if optimisation_toggle:
    print('performing tree/training data optimisation...')
    # Optimize training data size
    test_val_complex, result_complex, acc_cT, training_complex = training_data_size_optimize('Complex',
                                                                                             training_bands,
                                                                                             points_name_simple,
                                                                                             points_name_complex)
    test_val_simple, result_simple, acc_sT, training_simple = training_data_size_optimize('Simple',
                                                                                          training_bands,
                                                                                          points_name_simple,
                                                                                          points_name_complex)
    fp_train_complex_points = points_name_complex + str(training_complex) + ".shp"
    fp_train_simple_points = points_name_simple+str(training_simple)+".shp"

    if plot_toggle:
        line_plot(test_val_complex,
                  result_complex,
                  'Training data sample size (complex)',
                  plot_dir)
        line_plot(test_val_simple,
                  result_simple,
                  'Training data sample size (simple)',
                  plot_dir)

    # Optimize number of trees
    train_complex, test_complex = load_sample_training_data(fp_train_complex_points, training_bands,
                                                            stack, scale, class_col_name)
    train_simple, test_simple = load_sample_training_data(fp_train_simple_points, training_bands,
                                                          stack, scale, class_col_name)

    test_val_complex, result_complex, acc_cTr, trees_complex = trees_size_optimize('Complex',
                                                                                   training_bands,
                                                                                   train_complex,
                                                                                   test_complex)
    test_val_simple, result_simple, acc_sTr, trees_simple = trees_size_optimize('Simple',
                                                                                training_bands,
                                                                                train_simple,
                                                                                test_simple)
    if plot_toggle:
        line_plot(test_val_complex,
                  result_complex,
                  'Tree size (complex)',
                  plot_dir)
        line_plot(test_val_simple,
                  result_simple,
                  'Tree size (simple)',
                  plot_dir)

    # Write out results to preset file
    preset_table = generate_empty_preset_table()
    preset_table['Value']['fp_train_simple_points'] = fp_train_simple_points
    preset_table['Value']['fp_train_complex_points'] = fp_train_complex_points
    preset_table['Value']['trees_complex'] = trees_complex
    preset_table['Value']['training_complex'] = training_complex
    preset_table['Value']['trees_simple'] = trees_simple
    preset_table['Value']['training_simple'] = training_simple
    preset_table.to_csv(fp_settings_txt)

else:
    # TODO: Tom, think this is just a repeat of the code under 'use presets'?
    print('No optimisation run. Using presets...')
    if os.path.isfile(fp_settings_txt):
        fp_train_simple_points, fp_train_complex_points, trees_complex, training_complex, trees_simple, \
        training_simple = load_presets(fp_settings_txt)
    else:
        print('No presets available, run with optimisation set to True to generate presets.')
        sys.exit()

# Load training data
train_complex, test_complex = load_sample_training_data(fp_train_complex_points, training_bands,
                                                        stack, scale, class_col_name)
train_simple, test_simple = load_sample_training_data(fp_train_simple_points, training_bands,
                                                      stack, scale, class_col_name)

# Generate the classification model
clf_complex = generate_RF_model('Complex', int(trees_complex), train_complex, class_col_name, training_bands)
clf_simple = generate_RF_model('Simple', int(trees_simple), train_simple, class_col_name, training_bands)

# Generate classification performance stats
if advanced_performance_stats_toggle:
    accuracy_assessment_full(clf_complex, test_complex, 'Complex_RF_', fp_export_dir)
    accuracy_assessment_full(clf_simple, test_simple, 'Simple_RF_', fp_export_dir)
elif performance_stats_toggle:
    accuracy_assessment_basic(clf_complex, test_complex, 'Complex_RF_')
    accuracy_assessment_basic(clf_simple, test_simple, 'Simple_RF_')
else:
    print('No accuracy assessment carried out.')

# Generate processing areas to allow band maths
print('Generating processing areas...')
process_size = 1.0
process_dir = tile_polygon(fp_target_ext, process_size, fp_export_dir, 'processing_areas/')
process_list = get_list_of_files(process_dir, ".shp")

# Tile processing areas to allow GEE extraction
tile_size = 0.1
for i in process_list:
    print('Generating tiles for processing area', str(i) + '...')
    process_num = i.split('.')[0].split('/')[-1]
    tile_dir = tile_polygon(i, tile_size, fp_export_dir, process_num + '_tiles/')

    # Run the classification for the desired years over the processing areas and tiles
    tile_list = get_list_of_files(fp_export_dir+process_num+'_tiles/', ".shp")
    process_aoi = geemap.shp_to_ee(i)
    for j in years_to_map:
        for k in tile_list:
            stack_map, training_bands_map = stack_builder_run(process_aoi, j, max_min_values_training)
            export_name = 'PArea_' + str(j)+'_tile_'+process_num+'_RF_'
            apply_random_forest(export_name + 'Complex', training_bands_map, k, stack_map, scale, fp_export_dir,
                                clf_complex)
            apply_random_forest(export_name + 'Simple', training_bands_map, k, stack_map, scale, fp_export_dir,
                                clf_simple)
    shutil.rmtree(tile_dir)

# Clean up
shutil.rmtree(process_dir)
