"""
This script contains all the functions related to the preparation and handling of the land cover class training
data, as required by the Polesia classification routines.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
from geemap import geemap
import ee
import sys

ee.Initialize()


def load_sample_training_data(fp_train_points, target_bands, stack, scale, label):
    """
    This function takes a filepath to a point shape file of your training data.
    It returns a training and test sampled dataset with a 70-30 split.
    It operates over the passed data stack.
    """
    print('Loading training data:', fp_train_points+'...')
    try:
        training_points = geemap.shp_to_ee(fp_train_points)
    except Exception as e:
        str(e)
        print('Error in Loading training data...', e)
        sys.exit()

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