"""
Script to test the training points and balance the classes
"""
import geopandas as gpd
import matplotlib.pyplot as plt


# Load shapefile
fpath = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Training_points_20_20.shp"
raw_df = gpd.read_file(fpath)

class_count = raw_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Training points - all')
plt.show()

# Re-sample to 500 samples
sample_amounts = {0: 500,
                  1: 500,
                  2: 500,
                  3: 500,
                  4: 500,
                  5: 500,
                  6: 500,
                  7: 500,
                  8: 500,
                  9: 500,
                  10: 500,
                  11: 500,
                  12: 500,
                  13: 500,
                  14: 500,
                  15: 500,
                  16: 500,
                  17: 500,
                  18: 500,
                  19: 500}

resampled_df = (raw_df.groupby('VALUE').apply(lambda g: g.sample(n=sample_amounts[g.name],
                                                                 replace=len(g) < sample_amounts[g.name])).droplevel(0))
class_count = resampled_df['VALUE'].value_counts()
class_count.plot(kind='bar', legend=False, title='Training points - resampled')
plt.show()

# Export to new .shp
resampled_df.to_file("D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Training_points_500.shp")