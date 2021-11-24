import sys
from geemap import geemap

sys.path.append("C:/Users/tpfdo/OneDrive/Documents/GitHub/Polesia-Landcover/Routines/")
#sys.path.append("/home/markdj/repos/Polesia-Landcover/Routines/")
from Classification_tools import primary_classification_function, accuracy_assessment_basic, map_target_area


"""
User defined variables
"""
# Processing and output options

# User settings
#base_dir = '/home/markdj/Dropbox/artio/polesia'
base_dir = 'D:/tpfdo/Documents/Artio_drive/Projects/Polesia'

# File paths and directories for classification pipeline
fp_train_ext = f"{base_dir}/Project_area.shp"
complex_training_fpath = f"{base_dir}/Training_data/Complex_points_2500_v4.shp"
simple_training_fpath = f"{base_dir}/Training_data/Simple_points_2500_v4.shp"
fp_target_ext = f"{base_dir}/whole_map.shp"
fp_export_dir = f"{base_dir}/Classified/"


"""
Classification pipeline
"""
# Hard coded variables
aoi = geemap.shp_to_ee(fp_train_ext)
label = 'VALUE'
training_year = "2018"
scale = 20
trees_complex = 250
trees_simple = 250
years_to_map = [2018]

# Set up the classifier and test the accuracy of it
clf_complex, test_complex, max_min_values_complex = primary_classification_function(training_year, scale, label, aoi,
                                                                                    complex_training_fpath,
                                                                                    trees_complex)
accuracy_assessment_basic(clf_complex, test_complex, 'Complex', label)

clf_simple, test_simple, max_min_values_simple = primary_classification_function(training_year, scale, label, aoi,
                                                                                 simple_training_fpath,
                                                                                 trees_simple)
accuracy_assessment_basic(clf_simple, test_simple, 'Simple', label)

# Use the classifiers to map the whole of your target areas (tiled)
map_target_area(fp_target_ext, fp_export_dir, years_to_map, scale, clf_complex, clf_simple)
