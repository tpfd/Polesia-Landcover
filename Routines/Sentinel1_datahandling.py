import ee
import numpy as np


def calculate_water_under_canopy(image):
    vv = image.select('VV_Filtered')
    water = vv.lt(-12).rename('Water')  # Threshold here for reflectance
    inundated = water.updateMask(water)
    return inundated


def compute_soil_moisture(image, aoi):
    """
    Soil moisture calculated with an alpha approximation on a temporal mean.
    """
    soil_moisture = image.expression('(10**((VV_Filtered*0.1) + 1.3)) / 100.0', {'VV_Filtered': image})
    mean_soil_moisture = soil_moisture.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10).get('constant')

    mean_soil_moisture_info = mean_soil_moisture.getInfo()
    if mean_soil_moisture_info is not None:
        return np.around(mean_soil_moisture_info, decimals=4)
    else:
        return np.nan
