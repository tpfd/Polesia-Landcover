import ee
import numpy as np


def calculate_s2_inundation(image):
    """
    https://medium.com/@melqkiades/water-detection-using-ndwi-on-google-earth-engine-2919a9bf1951
    https://eos.com/make-an-analysis/ndwi/
    """
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndwi_threshold = ndwi.gte(0.0)
    masked_image = ndwi.updateMask(ndwi_threshold)
    return masked_image


def s2_cloud_masking(collection, aoi):
    s2_image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first().clip(aoi)
    cloud_prob = s2_image.select('MSK_CLDPRB')
    mask = cloud_prob.lt(50)
    masked_image = s2_image.updateMask(mask)
    return masked_image
