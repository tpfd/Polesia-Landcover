"""
Script to test the training points and balance the classes
"""
import geopandas as gpd
import matplotlib.pyplot as plt


"""
Complex swamps
"""
def run_resample_training_data():
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


