import csv
import calendar
import ee
import datetime
from geemap import geemap
import pandas as pd


def generate_days(years):
    day_list = []
    for year in years:
        start_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 12, 31)
        delta = datetime.timedelta(days=1)
        current_date = start_date
        while current_date <= end_date:
            day_list.append(current_date.strftime("%Y-%m-%d"))
            current_date += delta
    return day_list


def generate_month_tuples(years):
    month_tuples = []
    for year in years:
        for month in range(1, 13):
            _, last_day = calendar.monthrange(year, month)
            first_day = f"{year}-{month:02d}-01"
            last_day = f"{year}-{month:02d}-{last_day:02d}"
            month_tuple = (first_day, last_day)
            month_tuples.append(month_tuple)
    return month_tuples


def prepare_csv_for_gee(csv_path, longitude_column, latitude_column):
    # Read the CSV file
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Get the header row
        data = [row for row in reader]  # Get the data rows

    # Prepare the features
    features = []
    for row in data:
        longitude = float(row[longitude_column])
        latitude = float(row[latitude_column])
        properties = {header: row[i] for i, header in enumerate(headers) if i != longitude_column and i != latitude_column}
        point = ee.Geometry.Point(longitude, latitude)
        feature = ee.Feature(point, properties)
        features.append(feature)

    # Create an Earth Engine feature collection
    feature_collection = ee.FeatureCollection(features)
    return feature_collection, features


def compute_daily_wetness(image_collection, start_date, end_date, region_of_interest):
    # Filter Sentinel-1 GRD collection
    collection = ee.ImageCollection(image_collection) \
        .filterBounds(region_of_interest) \
        .filterDate(start_date, end_date) \
        .select('VV')

    # Compute daily average backscatter
    def compute_wetness(image):
        wetness = image.multiply(-1).exp()
        return wetness.rename('wetness').copyProperties(image, ['system:time_start'])

    daily_wetness = collection.map(compute_wetness)
    return daily_wetness


def compute_weekly_wetness(image_collection, start_date, end_date, region_of_interest):
    # Filter Sentinel-1 GRD collection
    collection = ee.ImageCollection(image_collection) \
        .filterBounds(region_of_interest) \
        .filterDate(start_date, end_date) \
        .select('VV')

    # Compute weekly average backscatter
    def compute_wetness(image):
        start = image.date().advance(-3, 'day')
        end = image.date().advance(3, 'day')
        wetness = collection \
            .filterDate(start, end) \
            .mean() \
            .multiply(-1).exp()
        return wetness.rename('wetness').copyProperties(image, ['system:time_start'])

    weekly_wetness = collection.map(compute_wetness)
    return weekly_wetness


def visualize_image_collection(image_collection):
    # Create a Map instance
    Map = geemap.Map()

    # Add the image collection to the map
    Map.addLayer(image_collection, {}, 'Image Collection')

    # Center the map display on the extent of the image collection
    bounds = image_collection.geometry().bounds().getInfo()['coordinates']
    Map.fit_bounds(bounds)

    # Display the map
    Map
    return


def visualize_feature_collection(feature_collection):
    # Convert feature collection to JSON
    fc_json = geemap.ee_to_json(feature_collection)
    # Show the feature collection in a new window
    geemap.show_json(fc_json)
    return
