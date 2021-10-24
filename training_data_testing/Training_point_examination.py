"""
Script to test the training points and balance the classes
"""
import geopandas as gpd
import matplotlib.pyplot as plt

"""
Complex swamps
"""
# Load shapefile complex
fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_points.shp"
raw_df = gpd.read_file(fpath)

class_count = raw_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Complex training points - all')
plt.show()

# Re-sample to 500 samples
sample_amounts = {0: 1000,
                  1: 1000,
                  2: 1000,
                  3: 1000,
                  4: 1000,
                  5: 1000,
                  6: 1000,
                  7: 1000,
                  8: 1000,
                  9: 1000,
                  11: 1000,
                  12: 1000,
                  13: 1000,
                  14: 1000,
                  15: 1000,
                  16: 1000,
                  17: 1000,
                  18: 1000,
                  19: 1000}

resampled_df = (raw_df.groupby('VALUE').apply(lambda g: g.sample(n=sample_amounts[g.name],
                                                                 replace=len(g) < sample_amounts[g.name])).droplevel(0))
class_count = resampled_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Complex training points - resampled')
plt.show()

# Export to new .shp
resampled_df.to_file("D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_points_500.shp")


"""
Simple swamps
"""
fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_points.shp"
raw_df = gpd.read_file(fpath)

class_count = raw_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Simple training points - all')
plt.show()

# Re-sample to 500 samples
sample_amounts = {0: 1000,
                  1: 1000,
                  2: 1000,
                  3: 1000,
                  4: 1000,
                  5: 1000,
                  6: 1000,
                  7: 1000,
                  8: 1000,
                  9: 1000,
                  11: 1000,
                  12: 1000,
                  13: 1000,
                  10: 1000,
                  17: 1000,
                  18: 1000,
                  19: 1000}

resampled_df = (raw_df.groupby('VALUE').apply(lambda g: g.sample(n=sample_amounts[g.name],
                                                                 replace=len(g) < sample_amounts[g.name])).droplevel(0))
class_count = resampled_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Simple training points - resampled')
plt.show()

# Export to new .shp
resampled_df.to_file("D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_points_500.shp")