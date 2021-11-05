"""
This script contains the user settings and primary functionality of the Polesia landcover mapping tool.

You will need:
> A polygon of the area that covers all your training data (fp_train_extent)
> Your training data as a shapefile of points. (*_training_fpath)
> A polygon that covers all of the area you want mapped (fp_target_ext)

You may want:
> A presets .csv file (fp_settings_txt)

"""
import sys
import os
from geemap import geemap
import pandas as pd
sys.path.append("C:/Users/tpfdo/OneDrive/Documents/GitHub/Polesia-Landcover/Routines/")
from Training_data_handling import run_resample_training_data, load_sample_training_data,\
    training_data_size_optimize, trees_size_optimize
from Satellite_data_handling import stack_builder_run
from Utilities import line_plot, load_presets, generate_empty_preset_table
from Classification_tools import apply_random_forest, generate_RF_model

"""
User defined variables
"""
# Processing and output options
training_data_resample_toggle = False
plot_toggle = True
advanced_performance_stats_toggle = False
optimisation_toggle = False
use_presets = True

# User settings
class_col_name = 'VALUE'
scale = 20
years_to_map = [2018, 2019]
tile_size = 1000

# File paths and directories for classification pipeline
fp_train_ext = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Project_area.shp"
fp_target_ext = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Classif_area.shp"
fp_export_dir = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Classified/"
fp_settings_txt = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/RF_classif_setting.csv"
plot_dir = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Plots/"

# File paths to shapefiles of target class points and the export dir for their resampling
complex_training_fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_swamp_points_v4.shp"
simple_training_fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_swamp_points_v4.shp"
class_export_dir = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/"

"""
Classification pipeline
"""
# Set training data name template
points_name_simple = class_export_dir + "Simple_"
points_name_complex = class_export_dir + "Complex_"

# Build the data stack for model training
aoi = geemap.shp_to_ee(fp_train_ext)
stack, training_bands = stack_builder_run(aoi, 2018)

# Load preset classification parameters if so toggled
if use_presets:
    if os.path.isfile(fp_settings_txt):
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
    # Training data size
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

    # Number of trees
    test_val_complex, result_complex, acc_cTr, trees_complex = trees_size_optimize('Complex',
                                                                                   training_bands)
    test_val_simple, result_simple, acc_sTr, trees_simple = trees_size_optimize('Simple',
                                                                                training_bands)
    if plot_toggle:
        line_plot(test_val_complex,
                  result_complex,
                  'Tree size (complex)',
                  plot_dir)
        line_plot(test_val_simple,
                  result_simple,
                  'Tree size (simple)',
                  plot_dir)

    # Write out to preset file
    preset_table = generate_empty_preset_table()
    preset_table['Value']['fp_train_simple_points'] = fp_train_simple_points
    preset_table['Value']['fp_train_complex_points'] = fp_train_complex_points
    preset_table['Value']['trees_complex'] = trees_complex
    preset_table['Value']['training_complex'] = training_complex
    preset_table['Value']['trees_simple'] = trees_simple
    preset_table['Value']['training_simple'] = training_simple
    preset_table.to_csv(fp_settings_txt)

else:
    print('No optimisation run. Using presets...')
    if os.path.isfile(fp_settings_txt):
        fp_train_simple_points, fp_train_complex_points, trees_complex, training_complex, trees_simple, \
        training_simple = load_presets(fp_settings_txt)
    else:
        print('No presets available, run with Optimisation set to True to generate presets.')
        sys.exit()

# Load training data
train_complex, test_complex = load_sample_training_data(fp_train_complex_points, training_bands,
                                                        stack, scale, class_col_name)
train_simple, test_simple = load_sample_training_data(fp_train_simple_points, training_bands,
                                                      stack, scale, class_col_name)

# Generate the classification model
clf_complex = generate_RF_model('Complex', int(trees_complex), train_complex, class_col_name, training_bands)
clf_simple = generate_RF_model('Simple', int(trees_simple), train_simple, class_col_name, training_bands)

# Generate performance stats

# Run the classification for the desired years
for i in years_to_map:
    aoi = geemap.shp_to_ee(fp_target_ext)
    stack_map, training_bands_map = stack_builder_run(aoi, i)
    export_name = str(i)+'_RF_'
    apply_random_forest(export_name+'Complex', training_bands_map, fp_target_ext, stack_map, scale, fp_export_dir,
                        clf_complex)
    apply_random_forest(export_name + 'Simple', training_bands_map, fp_target_ext, stack_map, scale, fp_export_dir,
                        clf_simple)
