"""
This script contains all the functions related to the preparation and handling of the land cover class training
data, as required by the Polesia classification routines.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
from geemap import geemap
import numpy as np
import ee
import sys
sys.path.append("C:/Users/tpfdo/OneDrive/Documents/GitHub/Polesia-Landcover/Routines/")
from Classification_tools import apply_random_forest, accuracy_assessment_simple
from Utilities import get_max_acc, match_result_lengths

ee.Initialize()


def load_sample_training_data(fp_train_points, target_bands, stack, scale, label):
    """
    This function takes a filepath to a point shape file of your training data.
    It returns a training and test sampled dataset with a 70-30 split.
    It operates over the passed data stack.
    """
    print('Loading training data:', fp_train_points+'...')
    training_points = geemap.shp_to_ee(fp_train_points)
    data = stack.select(target_bands).sampleRegions(collection=training_points,
                                                    properties=[label],
                                                    scale=scale)

    # Split into train and test
    split = 0.7
    data = data.randomColumn(seed=0)
    train = data.filter(ee.Filter.lt('random', split))
    test = data.filter(ee.Filter.gte('random', split))
    print('Training data loaded and ready!')
    return train, test


def run_resample_training_data(in_fpath, plot_toggle, plot_title, type_name, class_export_dir, plot_dir,
                               class_col_name):
    # Load shapefile
    raw_df = gpd.read_file(in_fpath)
    class_count = raw_df[class_col_name].value_counts()

    if plot_toggle:
        class_count.plot(kind='bar', legend=False, title=plot_title+' pre-resample class distribution')
        plt.savefig(plot_dir+plot_title+' pre-resample class distribution.png')

    class_sizes = [500, 750, 1000, 1250, 1500, 1750, 2000,
                   2250, 2500, 2750, 2800, 2850, 2900, 3000,
                   3250, 3500, 4000, 4250, 4500, 4750, 5000]
    for i in class_sizes:
        resample_training_data(i, raw_df, type_name, class_export_dir, class_col_name)


def resample_training_data(sample_size, data, type_name, class_export_dir, class_col_name):
    """
    Inputs:
        > sample_size: int, the desired size of the training data classes, per class.
        > data, geopandas data frame, a gdf of point data from a shape file.
        > type_name: str, the prefix of the output dataset
    Outputs:
        > A shapefile of the specified sample size in which all classes have the same number
        of samples. Randomly sampled down when below the value and repeat sampled up when
        below the value.
    """
    sample_amounts = {1: sample_size,
                      2: sample_size,
                      3: sample_size,
                      4: sample_size,
                      5: sample_size,
                      6: sample_size,
                      7: sample_size,
                      8: sample_size,
                      9: sample_size,
                      10: sample_size,
                      11: sample_size,
                      12: sample_size,
                      13: sample_size}

    data = (data.groupby(class_col_name).apply(lambda g: g.sample(n=sample_amounts[g.name],
                                                                  replace=len(g) < sample_amounts[g.name]
                                                                  )).droplevel(0))
    # Export to new .shp
    export_name = class_export_dir + type_name + str(sample_size) + ".shp"
    data.to_file(export_name)
    print(str(sample_size), " exported")


def training_data_size_optimize(type_switch, bands_in, points_name_simple, points_name_complex):
    training_test_vals = [500, 750, 1000, 1250, 1500, 1750, 2000,
                          2250, 2500, 2750, 2800, 2850, 2900, 3000,
                          3250, 3500, 4000, 4250, 4500, 4750, 5000]
    result_trainsize_vals = []
    for i in training_test_vals:
        try:
            if type_switch == 'Complex':
                fp = points_name_complex + str(i) + ".shp"
                train, test = load_sample_training_data(fp, bands_in)
                clf = apply_random_forest(train, 'RF_complex_train_'+str(i), 150, bands_in)
                val = accuracy_assessment_simple(clf, test, 'RF_stackv2_train_'+str(i))
                result_trainsize_vals.append(val)
            elif type_switch == 'Simple':
                fp = points_name_simple + str(i) + ".shp"
                train, test = load_sample_training_data(fp, bands_in)
                clf = apply_random_forest(train, 'RF_complex_train_'+str(i), 150, bands_in)
                val = accuracy_assessment_simple(clf, test, 'RF_stackv2_train_'+str(i))
                result_trainsize_vals.append(val)
            else:
                print('Specify classes type: Simple or Complex')
                break
        except:
            val = np.nan
            result_trainsize_vals.append(val)
            break

    training_test_vals = match_result_lengths(training_test_vals, result_trainsize_vals)
    max_acc, training_size = get_max_acc(training_test_vals, result_trainsize_vals)
    return training_test_vals, result_trainsize_vals, max_acc, training_size


def trees_size_optimize(type_switch, bands_in, train, test):
    trees_test_vals = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450]
    result_trees_vals = []
    for i in trees_test_vals:
        try:
            if type_switch == 'Complex':
                clf = apply_random_forest(train, 'RF_complex_trees_'+str(i), i, bands_in)
                val = accuracy_assessment_simple(clf, test, 'RF_complex_trees'+str(i))
                result_trees_vals.append(val)
            elif type_switch == 'Simple':
                clf = apply_random_forest(train, 'RF_simple_trees_' + str(i), i, bands_in)
                val = accuracy_assessment_simple(clf, test, 'RF_simple_trees' + str(i))
                result_trees_vals.append(val)
            else:
                print('Specify classes type: Simple or Complex')
                break
        except:
            val = np.nan
            result_trees_vals.append(val)
            break

    trees_test_vals = match_result_lengths(trees_test_vals, result_trees_vals)
    max_acc, tree_size = get_max_acc(trees_test_vals, result_trees_vals)
    return trees_test_vals, result_trees_vals, max_acc, tree_size
