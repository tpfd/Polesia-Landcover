"""
This script contains the user settings and primary functionality of the Polesia landcover mapping tool.
"""
import sys
import os
from geemap import geemap
import pandas as pd
sys.path.append("C:/Users/tpfdo/OneDrive/Documents/GitHub/Polesia-Landcover/Routines/")
from Training_data_handling import run_resample_training_data, load_sample_training_data
from Satellite_data_handling import stack_builder_run


"""
User defined variables
"""
# Processing and output options
training_data_resample_toggle = True
plot_toggle = True
advanced_performance_stats_toggle = False
optimisation_toggle = False
use_presets = True

# Name of the classes column in the training data and output classification scale (m)
class_col_name = 'VALUE'
scale = 20

# File paths and directories for classification pipeline
fp_train_ext = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Project_area.shp"
fp_target_ext = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Classif_area.shp"
fp_export_dir = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Classified/"
fp_settings_txt = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/RF_classif_setting.txt"

# File paths to shapefiles of target classes
complex_training_fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_swamp_points_v4.shp"
simple_training_fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_swamp_points_v4.shp"


"""
Training data handling and set up
"""
# Load preset classification parameters if already in place
if use_presets:
    if os.path.isfile(fp_settings_txt):
            preset_table = pd.read_csv(fp_settings_txt)
            preset_table.index = preset_table.Variable
    else:
        print('No preset parameters available, run with optimisation toggle set to True first.')
        sys.exit()

if training_data_resample_toggle:
    run_resample_training_data(complex_training_fpath, plot_toggle, 'Complex classes', 'Complex')
    run_resample_training_data(simple_training_fpath, plot_toggle, 'Simple classes', 'Simple')

if optimisation_toggle:
    training_data_size_optimize()

else:
    if os.path.isfile(fp_settings_txt):
            preset_table = pd.read_csv(fp_settings_txt)
            preset_table.index = preset_table.Variable
    else:
        print('No preset parameters available, run with optimisation toggle set to True first.')
        sys.exit()
     # Load the presets
    fp_train_complex_points = preset_table.loc['fp_train_complex_points'][1]
    fp_train_simple_points = preset_table.loc['fp_train_simple_points'][1]


"""
Classification pipeline
"""
# Build the data stack
aoi = geemap.shp_to_ee(fp_train_ext)
stack, training_bands = stack_builder_run(aoi)

# Load training data and sample the stack with it
train_simple, test_simple = load_sample_training_data(fp_train_simple_points, training_bands,
                                                      stack, scale, class_col_name)
train_complex, test_complex = load_sample_training_data(fp_train_complex_points, training_bands,
                                                        stack, scale, class_col_name)



# Perform the classification
