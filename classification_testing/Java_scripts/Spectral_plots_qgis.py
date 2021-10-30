import ee 
from ee_plugin import Map 

gcps = ee.FeatureCollection("users/ujavalgandhi/e2e/bangalore_gcps")
composite = ee.Image('users/ujavalgandhi/e2e/bangalore_composite')

# Overlay the point on the image to get bands data.
training = composite.sampleRegions({
  'collection': gcps,
  'properties': ['landcover'],
  'scale': 10
})


# We will create a chart of spectral signature for all classes

# We have multiple GCPs for each class
# Use a grouped reducer to calculate the average reflectance
# for each band for each class

# We have 12 bands so need to repeat the reducer 12 times
# We also need to group the results by class
# So we find the index of the landcover property and use it
# to group the results
bands = composite.bandNames()
numBands = bands.length()
bandsWithClass = bands.add('landcover')
classIndex = bandsWithClass.indexOf('landcover')

# Use .combine() to get a reducer capable of
# computing multiple stats on the input
combinedReducer = ee.Reducer.mean().combine({
  'reducer2': ee.Reducer.stdDev(),
  'sharedInputs': True})

# Use .repeat() to get a reducer for each band
# We then use .group() to get stats by class
repeatedReducer = combinedReducer.repeat(numBands).group(classIndex)

gcpStats = training.reduceColumns({
    'selectors': bands.add('landcover'),
    'reducer': repeatedReducer,
})

# Result is a dictionary, we do some post-processing to
# extract the results
groups = ee.List(gcpStats.get('groups'))

classNames = ee.List(['urban', 'bare', 'water', 'vegetation'])


def func_aad(item):
  # Extract the means
  values = ee.Dictionary(item).get('mean')
  groupNumber = ee.Dictionary(item).get('group')
  properties = ee.Dictionary.fromLists(bands, values)
  withClass = properties.set('class', classNames.get(groupNumber))
  return ee.Feature({}, withClass)

fc = ee.FeatureCollection(groups.map(func_aad
))






))

# Chart spectral signatures of training data
options = {
  'title': 'Average Spectral Signatures',
  'hAxis': '{title': 'Bands'},
  'vAxis': '{title': 'Reflectance',
    'viewWindowMode':'explicit',
    'viewWindow': {
        'max':0.6,
        'min':0
    }},
  'lineWidth': 1,
  'pointSize': 4,
  'series': {
    '0': '{color': 'grey'},
    '1': '{color': 'brown'},
    '2': '{color': 'blue'},
    '3': '{color': 'green'},
}}

# Default band names don't sort propertly
# Instead, we can give a dictionary with
# labels for each band in the X-Axis
bandDescriptions = {
  'B2': 'B02/Blue',
  'B3': 'B03/Green',
  'B4': 'B04/Red',
  'B8': 'B08/NIR',
  'B11': 'B11/SWIR-1',
  'B12': 'B12/SWIR-2'
}
# Create the chart and set options.
chart = ui.Chart.feature.byProperty({
  'features': fc,
  'xProperties': bandDescriptions,
  'seriesProperty': 'class'
}) \
.setChartType('ScatterChart') \
.setOptions(options)

print(chart)

def classChart(landcover, label, color):
  options = {
  'title': 'Spectral Signatures for ' + label + ' Class',
  'hAxis': '{title': 'Bands'},
  'vAxis': '{title': 'Reflectance',
    'viewWindowMode':'explicit',
    'viewWindow': {
        'max':0.6,
        'min':0
    }},
  'lineWidth': 1,
  'pointSize': 4,
  }

  fc = training.filter(ee.Filter.eq('landcover', landcover))
  chart = ui.Chart.feature.byProperty({
  'features': fc,
  'xProperties': bandDescriptions,
  }) \
.setChartType('ScatterChart') \
.setOptions(options)

print(chart)

classChart(0, 'Urban')
classChart(1, 'Bare')
classChart(2, 'Water')
classChart(3, 'Vegetation')
