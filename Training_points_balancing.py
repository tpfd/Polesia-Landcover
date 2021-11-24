"""
Script to test the training points and balance the classes
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import sys


def resample_training_data(sample_size, data, export_prefix, class_export_dir, class_col_name,
                           type_train):
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
    if type_train == 'Complex':
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
    elif type_train == 'Simple':
        sample_amounts = {1: sample_size,
                          2: sample_size,
                          3: sample_size,
                          4: sample_size,
                          5: sample_size,
                          6: sample_size,
                          7: sample_size,
                          8: sample_size,
                          9: sample_size}
    else:
        print('Type of training data, not recognised: use Simple or Complex')
        sys.exit()

    data = (data.groupby(class_col_name).apply(lambda g: g.sample(n=sample_amounts[g.name],
                                                                  replace=len(g) < sample_amounts[g.name]
                                                                  )).droplevel(0))
    # Export to new .shp
    export_name = class_export_dir + export_prefix + str(sample_size) + ".shp"
    data.to_file(export_name)
    print(str(sample_size), " exported")

"""
Complex swamps
"""
# Load shapefile complex
fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_swamp_points_v4.shp"
raw_df = gpd.read_file(fpath)

class_count = raw_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Complex training data classes - no balancing')
plt.show()

training_test = [250, 500, 750, 1000, 1250, 1500, 1750, 2000,
                 2250, 2500, 2750, 2800, 2850, 2900, 3000,
                 3250, 3500, 4000, 4250, 4500, 4750, 5000]
for i in training_test:
    resample_training_data(i,
                           raw_df,
                           'Complex_points_',
                           "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/",
                           'VALUE',
                           'Complex')


"""
Simple swamps
"""
fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_points_v4.shp"
raw_df = gpd.read_file(fpath)

class_count = raw_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Simple training data classes - no balancing')
plt.show()

training_test = [250, 500, 750, 1000, 1250, 1500, 1750, 2000,
                 2250, 2500, 2750, 2800, 2850, 2900, 3000,
                 3250, 3500, 4000, 4250, 4500, 4750, 5000]
for i in training_test:
    resample_training_data(i,
                           raw_df,
                           'Simple_points_',
                           "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/",
                           'VALUE',
                           'Simple')


