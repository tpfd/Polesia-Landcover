import ee
import numpy as np


def calculate_s2_inundation(image):
    """
    https://medium.com/@melqkiades/water-detection-using-ndwi-on-google-earth-engine-2919a9bf1951
    """
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndwi_threshold = ndwi.gte(0.02)  # Slightly higher threshold to try and also get mixed water/stream edge
    inundated = ndwi_threshold.updateMask(ndwi_threshold)
    return inundated
