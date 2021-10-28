"""
Script to test the training points and balance the classes
"""
import geopandas as gpd
import matplotlib.pyplot as plt

"""
Complex swamps
"""
# Load shapefile complex
fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_swamp_points_v3.shp"
raw_df = gpd.read_file(fpath)

class_count = raw_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Complex training points - all')
plt.show()

# Re-sample to 1750 samples
sample_amounts = {1: 2000,
                  2: 2000,
                  3: 2000,
                  4: 2000,
                  5: 2000,
                  6: 2000,
                  7: 2000,
                  8: 2000,
                  9: 2000,
                  10: 2000,
                  11: 2000,
                  12: 2000,
                  13: 2000,
                  14: 2000}

resampled_df = (raw_df.groupby('VALUE').apply(lambda g: g.sample(n=sample_amounts[g.name],
                                                                 replace=len(g) < sample_amounts[g.name])).droplevel(0))
class_count = resampled_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Complex training points - resampled')
plt.show()

# Export to new .shp
resampled_df.to_file("D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_points_2000_v3.shp")


"""
Simple swamps
"""
fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_points_v3.shp"
raw_df = gpd.read_file(fpath)

class_count = raw_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Simple training points - all')
plt.show()

# Re-sample to 2000 samples
sample_amounts = {1: 1000,
                  2: 1000,
                  3: 1000,
                  4: 1000,
                  5: 1000,
                  6: 1000,
                  7: 1000,
                  8: 1000,
                  9: 1000,
                  10: 1000,
                  11: 1000,
                  12: 1000}

resampled_df = (raw_df.groupby('VALUE').apply(lambda g: g.sample(n=sample_amounts[g.name],
                                                                 replace=len(g) < sample_amounts[g.name])).droplevel(0))
class_count = resampled_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Simple training points - resampled')
plt.show()

# Export to new .shp
resampled_df.to_file("D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_points_1000_v3.shp")