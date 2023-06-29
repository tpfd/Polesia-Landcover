import csv
import os
import pprint
from Sentinel2_datahandling import *
from Sentinel1_datahandling import *
import matplotlib.pyplot as plt
import numpy as np

def count_nonzero_pixels(binary_mask, band_name):
    """
    Counts the number of non-zero pixels in a binary mask image.
    Returns the count as an integer.
    Only works for binary image.
    """
    # Count non-zero pixels using reduceRegion with sum reducer
    count = binary_mask.reduceRegion(reducer=ee.Reducer.sum(),
                                     geometry=binary_mask.geometry(),
                                     scale=binary_mask.projection().nominalScale(),
                                     bestEffort=True).getInfo()

    # Retrieve the count from the result
    nonzero_count = count[band_name]
    return nonzero_count


def count_non_nan_pixels(image, selector):
    count = image.reduceRegion(reducer=ee.Reducer.count(),
                               geometry=image.select(selector).geometry(),
                               scale=image.select(selector).projection().nominalScale())
    count_value = ee.Number(count.get(selector))
    return count_value.getInfo()


def hist(image_collection, selector, aoi):
    # Compute histogram with automatic parameters
    histogram = image_collection.reduceRegion(ee.Reducer.reduceHistogram(
        scale=10, maxBuckets=500, region=aoi).getInfo())

    # Convert the histogram to numpy arrays
    a = np.array(histogram['histogram'])
    x = a[:, 0]  # array of bucket edge positions
    y = a[:, 1] / np.sum(a[:, 1])  # normalized array of bucket contents

    # Plot the histogram
    plt.grid()
    plt.plot(x, y, '.')
    plt.show()


def get_sentinel_wetness(longitude, latitude, date):
    """
    longitude = 24.1325316667
    latitude = 52.7465083333
    date = '2017-07-05'

    Binary, 0 = no water visible
    SM, higher numbers = dryer

    S1 10 m pixel size: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
    S2 10 m pixel size: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    """
    # Set the point of interest (longitude, latitude)
    point = ee.Geometry.Point(longitude, latitude)

    # 50 m radius data collect around that point (.5 to allow for corners and get 100x100)
    buffer_radius = 50.5
    buffer = point.buffer(buffer_radius)

    # Convert date to Earth Engine's format and get the 3-day period around it
    start_date = ee.Date(date).advance(-1, 'day')
    end_date = ee.Date(date).advance(1, 'day')

    # Load Sentinel-1 Synthetic Aperture Radar (SAR) collection
    s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(buffer) \
        .filterDate(start_date, end_date) \
        .select('VV')

    if s1_collection.size().getInfo() > 0:
        hist(s1_collection, 'VV', buffer)

        # Carry out pre-processing
        print('De-speckling S1...')
        s1_mean_image = s1_collection.mean().clip(buffer)
        vv = s1_mean_image.select('VV')
        vv_smoothed = vv.focal_median(30, 'circle', 'meters').rename('VV_Filtered')  # De-speckle
        s1_total_pixel_count = count_non_nan_pixels(vv_smoothed, 'VV_Filtered')

        hist(vv_smoothed, 'V_Filtered', buffer)

        # Calculate SM using an alpha approximation on the temporal mean
        print('Calculating SM from S1...')
        soil_moisture_value_out = compute_soil_moisture(vv_smoothed, buffer)

        # Calculate binary surface water using arbitrary canopy threshold
        print('Calculating surface water from S1...')
        s1_surface_water_mask = calculate_water_under_canopy(vv_smoothed)
        s1_inundation_count = count_nonzero_pixels(s1_surface_water_mask, 'Water')
    else:
        print('No S1 available')
        soil_moisture_value_out = np.nan
        s1_total_pixel_count = np.nan
        s1_inundation_count = np.nan

    # Load Sentinel-2 optical image collection
    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(buffer) \
        .filterDate(start_date, end_date)

    if sentinel2.size().getInfo() > 0:
        # Cloud mask
        print('Masking cloud S2...')
        s2_cloud_filtered = s2_cloud_masking(sentinel2, buffer)
        s2_total_pixel_count = count_non_nan_pixels(s2_cloud_filtered, 'B3')
        hist(s2_cloud_filtered, 'B3', buffer)

        # Calculate inundation binary using Sentinel-2
        print('Calculating S2 inundation...')
        s2_inundation = calculate_s2_inundation(s2_cloud_filtered)
        hist(s2_inundation, 'NDWI', buffer)

        # Get the mean inundation binary value for the given area
        s2_inundation_count = count_nonzero_pixels(s2_inundation, 'NDWI')
    else:
        print('No S2 available')
        s2_inundation_count = np.nan
        s2_total_pixel_count = np.nan

    result = {
        'S1 Soil Moisture': soil_moisture_value_out,
        'S1 total pixel count': s1_total_pixel_count,
        'S1 Inundation count': s1_inundation_count,
        'S2 total pixel count': s2_total_pixel_count,
        'S2 Inundation count': s2_inundation_count
    }
    pprint.pprint(result)
    return result



def get_averages_for_nests(latitude, longitude, date):
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
    buffer_radius = 12000
    buffer = point.buffer(buffer_radius)

    # Convert date to Earth Engine's format and get the 3-day period around it
    start_date = ee.Date(date).advance(-1, 'day')
    end_date = ee.Date(date).advance(1, 'day')

    # Filter datasets by date
    rainfall_filtered = rainfall.filterDate(start_date, end_date)
    soil_moisture_filtered = soil_moisture.filterDate(start_date, end_date)

    # Check for data presence and extract the mean if there is data, otherwise set to np.nan
    if rainfall_filtered.size().getInfo() > 0:
        rainfall_filtered_mean = rainfall_filtered.mean().reduceRegion(ee.Reducer.mean(),
                                                                       geometry=buffer, scale=10000, crs='EPSG:4326',
                                                                       bestEffort=True).get('precipitationCal')
        if rainfall_filtered_mean.getInfo() is None:
            rainfall_mean = np.nan
        else:
            rainfall_mean = np.around(rainfall_filtered_mean.getInfo(), decimals=4)
    else:
        rainfall_mean = np.nan

    if soil_moisture_filtered.size().getInfo() > 0:
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


def save_dict_to_csv(data_dict, filename):
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)
