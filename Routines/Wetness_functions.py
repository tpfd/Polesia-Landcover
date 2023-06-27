import ee
import csv
import os
import numpy as np


def calculate_surface_water(image):
    # Apply threshold to classify surface water below tree canopy
    tree_canopy = image.lt(-20)  # Example threshold value, adjust as needed
    surface_water = image.multiply(tree_canopy)
    return surface_water.rename('water')


def calculate_inundation(image):
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    threshold = 0.2
    inundated = ndwi.gt(threshold)
    return inundated


def get_sentinel_wetness(longitude, latitude, date):
    """
    longitude = 24.1325316667
    latitude = 52.7465083333
    date = '2017-07-05'

    indunation binary, 0 = no water visible (but cannot see through canopy)
    """
    # Set the point of interest (longitude, latitude)
    point = ee.Geometry.Point(longitude, latitude)

    # Convert date to Earth Engine's format and get the 3-day period around it
    start_date = ee.Date(date).advance(-1, 'day')
    end_date = ee.Date(date).advance(1, 'day')

    # Load Sentinel-1 Synthetic Aperture Radar (SAR) collection
    sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(point) \
        .filterDate(start_date, end_date) \
        .select('VV')

    if sentinel1.size().getInfo() > 0:
        # Calculate SM using an alpha approximation
        mean_image = sentinel1.mean()
        soil_moisture = mean_image.expression('(10**((VV*0.1) + 1.3)) / 100.0', {'VV': mean_image})
        soil_moisture_value = soil_moisture.sample(point, scale=10).first().get('constant')
        soil_moisture_value_out = np.around(soil_moisture_value.getInfo(), decimals=3)

        # Calculate binary surface water using arbitrary canopy threshold
        surface_water = sentinel1.map(calculate_surface_water).sum()
        surface_water_value = surface_water.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10).get('water')
        surface_water_value_out = surface_water_value.getInfo()
    else:
        soil_moisture_value_out = np.nan
        surface_water_value_out = np.nan

    # Load Sentinel-2 optical image collection
    sentinel2 = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(point) \
        .filterDate(start_date, end_date)

    if sentinel2.size().getInfo() >= 0:
        # Calculate frequency of inundation using Sentinel-2
        inundation = sentinel2.map(calculate_inundation).sum()

        # Get the inundation binary value for the given point
        inundation_value = inundation.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10).get('NDWI')
        inundation_value_out= inundation_value.getInfo()
    else:
        inundation_value_out = np.nan


    result = {
        'S1 Soil Moisture': soil_moisture_value_out,
        'S1 Surface Water binary': surface_water_value_out,
        'S2 Inundation binary': inundation_value_out
    }
    print(result)
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
