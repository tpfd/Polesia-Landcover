import ee
import numpy as np


def calculate_s2_inundation(image):
    """
    https://medium.com/@melqkiades/water-detection-using-ndwi-on-google-earth-engine-2919a9bf1951
    https://eos.com/make-an-analysis/ndwi/
    """
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndwi_threshold = ndwi.gte(-0.01)
    masked_image = ndwi_threshold.rename('S2 Surface Water Binary')
    return masked_image, ndwi, ndwi_threshold


def s2_cloud_masking(collection, aoi):
    s2_image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first().clip(aoi)
    cloud_prob = s2_image.select('MSK_CLDPRB')
    mask = cloud_prob.lt(50)
    masked_image = s2_image.updateMask(mask)
    return masked_image
