"""
Script to test the training points and balance the classes
"""
import geopandas as gpd
import matplotlib.pyplot as plt


def resample_training_data(sample_size, data, type_name):
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

    data = (data.groupby('VALUE').apply(lambda g: g.sample(n=sample_amounts[g.name],
                                                           replace=len(g) < sample_amounts[g.name])).droplevel(0))
    # Export to new .shp
    data.to_file("D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/"+type_name+
                 str(sample_size)+"_v4.shp")
    print(str(sample_size), " exported")


"""
Complex swamps
"""
# Load shapefile complex
fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_swamp_points_v4.shp"
raw_df = gpd.read_file(fpath)

class_count = raw_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Complex training points - all')
plt.show()

training_test = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 2800, 2850, 2900, 3000, 3250, 3500, 4000]
for i in training_test:
    resample_training_data(i, raw_df, 'Complex_points_')



"""
Simple swamps
"""
fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_points_v4.shp"
raw_df = gpd.read_file(fpath)

class_count = raw_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Simple training points - all')
plt.show()

training_test = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 2800, 2850, 2900, 3000, 3250, 3500, 4000]
for i in training_test:
    resample_training_data(i, raw_df, 'Simple_points_')


