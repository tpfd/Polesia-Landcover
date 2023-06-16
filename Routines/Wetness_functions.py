import ee
import csv
import pandas as pd
import numpy as np


def get_averages(latitude, longitude, date):
    """
    https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06#bands
    https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_007#description
    """
    # Load required datasets
    rainfall = ee.ImageCollection('NASA/GPM_L3/IMERG_V06').select('precipitationCal')
    soil_moisture = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007').select(['sm_rootzone_wetness'])

    # Create a point from the input coordinates
    point = ee.Geometry.Point(longitude, latitude)

    # Create a 25 km buffer around the point
    buffer_radius = 25000
    buffer = point.buffer(buffer_radius)

    # Convert date to Earth Engine's format and get the 3-day period around it
    start_date = ee.Date(date).advance(-1, 'day')
    end_date = ee.Date(date).advance(1, 'day')

    # Filter datasets by date
    rainfall_filtered = rainfall.filterDate(start_date, end_date)
    soil_moisture_filtered = soil_moisture.filterDate(start_date, end_date)

    # Check for data presence and extract the mean if there is data, otherwise set to np.nan
    if rainfall_filtered.size().getInfo() >= 0:
        rainfall_filtered_mean = rainfall_filtered.mean().reduceRegion(ee.Reducer.mean(),
                                                                       geometry=buffer, scale=10000, crs='EPSG:4326',
                                                                       bestEffort=True).get('precipitationCal')
        if rainfall_filtered_mean.getInfo() is None:
            rainfall_mean = np.nan
        else:
            rainfall_mean = np.around(rainfall_filtered_mean.getInfo(), decimals=4)
    else:
        rainfall_mean = np.nan

    if soil_moisture_filtered.size().getInfo() >= 0:
        soil_moisture_mean = soil_moisture_filtered.mean().reduceRegion(ee.Reducer.mean(),
                                                                        geometry=buffer, scale=10000, crs='EPSG:4326',
                                                                        bestEffort=True).get('sm_rootzone_wetness')
        if soil_moisture_mean.getInfo() is None:
            soil_moisture_mean = np.nan
        else:
            soil_moisture_mean = np.around(soil_moisture_mean.getInfo(), decimals=4)
    else:
        soil_moisture_mean = np.nan

    return {
        'rainfall_mean': rainfall_mean,
        'soil_moisture_mean': soil_moisture_mean
    }


def save_data_to_csv(row, averages, file_path):
    """
    Apologies for the hack job... I'm tired and hungry.
    Here is where the use of the multi-index to make a clean transition is supposed to happen.
    Instead, I have done some pandas hokey-pokey.
    """
    new_row = pd.DataFrame(averages, index=[0]).transpose()
    row_df = pd.Series.to_frame(row)
    row_out = pd.concat([new_row, row_df.loc[:]]).reset_index(drop=False)
    values_to_move = row_out.iloc[:2, 1].tolist()
    row_out.iloc[:2, 2] = values_to_move
    row_out = row_out.drop(row_out.columns[1], axis=1)
    row_out.iloc[2, 0] = 'ID'

    values = row_out.iloc[:, 1].tolist()
    header = row_out['index'].tolist()

    if not pd.io.common.file_exists(file_path):
        mode = 'w'  # Create a new file if it doesn't exist
    else:
        mode = 'a'  # Append to existing file

    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w':  # Write header if the file is new
            writer.writerow(header)
        writer.writerow(values)
