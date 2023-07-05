import csv
import os
import pprint
from Sentinel2_datahandling import *
from Sentinel1_datahandling import *
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta


def count_non_nan_pixels(image, selector):
    """
    Will count all pixels not actively masked, i.e. 1 and 0s if a binary array that has not had
    the mask applied.
    """
    band = image.select(selector)
    count = band.reduceRegion(reducer=ee.Reducer.count(),
                              geometry=image.geometry(),
                              scale=10,
                              maxPixels=1e9)
    count_value = count.get(selector).getInfo()
    return count_value


def compute_histogram(image, aoi, base_dir, site_name):
    """
    https://kaflekrishna.com.np/blog-detail/histogram-image-google-earth-engine-gee-python-api/

    Only plots the histogram for you if you pass a site name. Otherwise just gives you the pixel counts.
    """
    plt.clf()
    # Compute histogram for each image in the collection
    histogramDictionary = image.reduceRegion(**{
        'reducer': ee.Reducer.histogram(10),
        'geometry': aoi,
        'scale': 10,
        'maxPixels': 1e19
    })

    # Plot the histogram using matplotlib
    histogram = histogramDictionary.getInfo()
    bands = list(histogram.keys())

    for bnd in bands:
        # plot a bar chart
        y = histogram[bnd]['histogram']
        x = []
        for i in range(len(y)):
            x.append(histogram[bnd]['bucketMin'] + i * histogram[bnd]['bucketWidth'])
        data = pd.DataFrame({'x': np.around(x, decimals=4),
                             'y': y})

        if site_name:
            # Draw Plot
            sns.set(font_scale=2)
            fig, ax = plt.subplots(figsize=(15, 15), dpi=150)
            sns.barplot(
                data=data,
                x='x',
                y='y',
                ax=ax,
                edgecolor="black",
                facecolor="lightsteelblue")
            # For every axis, set the x and y major locator
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))

            # Adjust width gap to zero
            for patch in ax.patches:
                current_height = patch.get_width()
                patch.set_width(1)
                patch.set_y(patch.get_y() + current_height - 1)

            # figure label and title
            plt.title('Histogram for Band: {}'.format(bnd), fontsize=24)
            plt.ylabel('Frequency', fontsize=24)
            plt.xlabel('Pixel value', fontsize=24)
            # save the figure as JPG file
            save_dir = base_dir + '/Plots/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(save_dir + site_name + '_fig-{}.jpg'.format(bnd))
            plt.close()
        return x, y


def get_pixel_counts_from_hist(x_in, y_in):
    try:
        total_pixel_count = y_in[0] + y_in[1]
        inundation_count = y_in[1]
        water_flag = True
    except:
        total_pixel_count = y_in[0]
        inundation_count = y_in[0]
        if x_in[0] == 0:
            water_flag = False
        elif x_in[0] == 1:
            water_flag = True
        else:
            print('Error in binary categories')
    return int(total_pixel_count), int(inundation_count), water_flag


def get_sentinel_wetness(longitude, latitude, date_in, base_dir, site_name):
    """
    longitude = 24.1325316667
    latitude = 52.7465083333
    date = '2017-07-05'

    Binary, 0 = no water visible
    SM, higher numbers = dryer

    S1 10 m pixel size: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
    S2 10 m pixel size: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

    If you pass a site name, it will save histograms. Pass None to just process silently.
    """
    # Set the point of interest (longitude, latitude)
    point = ee.Geometry.Point(longitude, latitude)

    # 50 m radius data collect around that point (.5 to allow for corners and get 100x100)
    buffer_radius = 50.5
    buffer = point.buffer(buffer_radius)

    # Convert date to Earth Engine's format and get the 3-day period around it
    start_date = ee.Date(date_in).advance(-1, 'day')
    end_date = ee.Date(date_in).advance(1, 'day')

    # Load Sentinel-1 Synthetic Aperture Radar (SAR) collection
    s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(buffer) \
        .filterDate(start_date, end_date) \
        .select('VV')

    if s1_collection.size().getInfo() > 0:
        # Carry out pre-processing
        s1_mean_image = s1_collection.mean().clip(buffer)
        vv = s1_mean_image.select('VV')
        vv_smoothed = vv.focal_median(30, 'circle', 'meters').rename('VV_Filtered')  # De-speckle

        s1_data_presence = count_non_nan_pixels(vv_smoothed, 'VV_Filtered')
        if s1_data_presence > 0:
            # Plot histograms if running tests
            if site_name:
                compute_histogram(vv_smoothed, buffer, base_dir, site_name)
            else:
                pass

            # Calculate SM using an alpha approximation on the temporal mean
            soil_moisture_value_out = compute_soil_moisture(vv_smoothed, buffer)

            # Calculate binary surface water
            s1_masked_image, s1_water_threshold = calculate_water_under_canopy(vv_smoothed)
            s1_x, s1_y = compute_histogram(s1_masked_image.select('S1 Surface Water Binary'),
                                           buffer,
                                           base_dir,
                                           site_name)
            s1_total_pixel_count, s1_inundation_count, water_flag = get_pixel_counts_from_hist(s1_x, s1_y)
            if water_flag is True:
                pass
            else:
                s1_inundation_count = 0

        else:
            soil_moisture_value_out = np.nan
            s1_total_pixel_count = np.nan
            s1_inundation_count = np.nan

    else:
        soil_moisture_value_out = np.nan
        s1_total_pixel_count = np.nan
        s1_inundation_count = np.nan

    # Load Sentinel-2 optical image collection
    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(buffer) \
        .filterDate(start_date, end_date)

    if sentinel2.size().getInfo() > 0:
        # Cloud mask
        s2_cloud_filtered = s2_cloud_masking(sentinel2, buffer)

        s2_total_pixel_count = count_non_nan_pixels(s2_cloud_filtered, 'B3')
        if s2_total_pixel_count > 0:
            # Calculate inundation binary using Sentinel-2
            s2_inundation, ndwi_raw, ndwi_threshold = calculate_s2_inundation(s2_cloud_filtered)
            s2_x, s2_y = compute_histogram(s2_inundation.select('S2 Surface Water Binary'),
                                           buffer,
                                           base_dir,
                                           site_name)
            s2_total_pixel_count, s2_inundation_count, water_flag = get_pixel_counts_from_hist(s2_x, s2_y)
            if water_flag is True:
                pass
            else:
                s2_inundation_count = 0

            # Plot histograms if running tests
            if site_name:
                compute_histogram(s2_cloud_filtered.select('B3'), buffer, base_dir, site_name)
                compute_histogram(s2_cloud_filtered.select('B8'), buffer, base_dir, site_name)
                compute_histogram(s2_inundation.select('S2 Surface Water Binary'), buffer, base_dir, site_name)
                compute_histogram(ndwi_raw.select('NDWI'), buffer, base_dir, site_name)
            else:
                pass
        else:
            s2_inundation_count = np.nan
            s2_total_pixel_count = np.nan
    else:
        s2_inundation_count = np.nan
        s2_total_pixel_count = np.nan

    result = {
        'S1 Soil Moisture': soil_moisture_value_out,
        'S1 total pixel count': s1_total_pixel_count,
        'S1 Inundation count': s1_inundation_count,
        'S2 total pixel count': s2_total_pixel_count,
        'S2 Inundation count': s2_inundation_count
    }
    pprint(result)
    return result


def get_month_dates(year):
    month_dates = []
    current_date = date(year, 1, 1)
    while current_date.year == year:
        next_month = current_date.replace(day=28) + timedelta(days=4)  # Ensure we move to the next month
        month_end = next_month - timedelta(days=next_month.day)
        month_dates.append((current_date.strftime('%Y-%m-%d'), month_end.strftime('%Y-%m-%d')))
        current_date = next_month
    return month_dates


def get_averages_for_nests(latitude, longitude, year, home_range, index_in, fp_out_nest):
    """
    Assumes a year is given as a single YYYY int, returns monthly means for that year.
    https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06#bands
    https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_007#description
    """
    # Load required datasets
    rainfall = ee.ImageCollection('NASA/GPM_L3/IMERG_V06').select('precipitationCal')
    soil_moisture = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007').select(['sm_rootzone_wetness'])

    # Create a point from the input coordinates
    point = ee.Geometry.Point(longitude, latitude)

    # Create a buffer around the point
    buffer_radius = home_range
    buffer = point.buffer(buffer_radius)

    # Get monthly date ranges for the year and run them
    dates_list = get_month_dates(int(year))
    for date_pair in dates_list:
        index_out = index_in + '_' + date_pair[0].split('-')[1]
        start_date = ee.Date(date_pair[0])
        end_date = ee.Date(date_pair[1])
        print('Getting nest data for:', index_out)

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

        nests_new_row_dict = {'index': index_out,
                              'SM rootzone wetness': soil_moisture_mean,
                              'Precipitation': rainfall_mean}
        pprint(nests_new_row_dict)
        save_dict_to_csv(nests_new_row_dict, fp_out_nest)
    return


def save_dict_to_csv(data_dict, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)
