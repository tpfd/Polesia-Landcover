import sys
import os
from geemap import geemap

sys.path.append("C:/Users/tdow214/OneDrive - The University of Auckland/Documents/GitHub/Polesia-Landcover/Routines/")
#sys.path.append("/home/markdj/repos/Polesia-Landcover/Routines/")
from Classification_tools import trees_size_optimize, training_data_size_optimize


"""
User defined variables
"""
# Processing and output options
accuracy_eval_toggle = False

# File paths and directories for classification pipeline
#base_dir = '/home/markdj/Dropbox/artio/polesia'
base_dir = 'D:/tpfdo/Documents/Artio_drive/Projects/Polesia'

fp_train_ext = f"{base_dir}/Project_area.shp"
root_train_fpath = f"{base_dir}/Training_data/"
complex_training_fpath = f"{base_dir}/Training_data/Complex_points_2000.shp"
simple_training_fpath = f"{base_dir}/Training_data/Simple_points_2000.shp"
plots_out_dir = f"{base_dir}/Plots/"

label = "VALUE"
scale = 20

"""
Optimization
> Prints out results and saves a simple summary line plot for each case.
"""
# Hard coded variables
aoi = geemap.shp_to_ee(fp_train_ext)

# Set up folders
if not os.path.isdir(plots_out_dir):
    os.mkdir(plots_out_dir)

# Optimize for tree size
trees_size_optimize(complex_training_fpath, aoi, 'Complex', label, plots_out_dir, scale)
trees_size_optimize(simple_training_fpath, aoi, 'Simple', label, plots_out_dir, scale)

# Optimize for training data size per class
training_data_size_optimize(root_train_fpath, aoi, 'Complex', label, 2018, plots_out_dir, scale)
training_data_size_optimize(root_train_fpath, aoi, 'Simple', label, 2018, plots_out_dir, scale)
