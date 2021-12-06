"""
This script contains the functions related to handling, moving, stacking and dealing with all forms of
the satellite data.
"""
import ee
import pandas as pd
ee.Initialize()


def create_data_stack(aoi, date_list, year, max_min_values=None):
    """
    Convenience function to compile and min-max scale all distinct dataset sub-stacks:
    * Sentinel 1 flood frequency index
    * Sentinel 2 data bands: variable from 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'
    * ** NO LONGER USED ** Sentinel 1 data bands: 'VV', 'VH'
    * ** NO LONGER USED ** topography data bands: slope, aspect & elevation [static, 30m SRTM derived]
    Also generates EVI & AVI indices from S2

    :param aoi: ee.featurecollection.FeatureCollection, used to indicate AOI extent
    :param date_list: list of tuples of strings (i.e. [('a','b'),('c','d')]), used to define start & end of each
    compositing period, expects 'YYYY-MM-DD' format
    :param year: int, current year to generate lc map for
    : max_min_values:, None (default) or tuple of max/min dicts. if None, computes min max values for each band for the
    training region. otherwise, computes max/min values
    :return: ee.image.Image, tuple of max min value dictionaries (max_dict, min_dict)
    """
    print('create_data_stack(): hello!')

    # parameters used for sentinel-2 imagery analysis
    s2_params = {
        'CLOUD_FILTER': 60,  # int, max cloud coverage (%) permitted in a scene
        'CLD_PRB_THRESH': 40,  # int, 's2cloudless' 'probability' band value > thresh = cloud
        'NIR_DRK_THRESH': 0.15,  # float, if Band 8 (NIR) < NIR_DRK_THRESH = possible shadow
        'CLD_PRJ_DIST': 1,  # int, max distance  from cloud edge for possible shadow (100s of m)
        'BUFFER': 50,  # int, distance (m) used to buffer cloud edges
        # 'S2BANDS': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        'S2BANDS': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']  # list of str, which S2 bands to return?
    }

    print('create_data_stack(): Getting satellite data...')
    #s1_stack = fetch_sentinel1(aoi, date_list)
    s2_stack = fetch_sentinel2(aoi, date_list, s2_params, fill_mask_bkwd=True, fill_mask_fwd=True)
    flood_index = fetch_sentinel1_flood_index(aoi, str((int(year)-1))+'-01-01',
                                                   year+'-12-01',
                                                   smoothing_radius=100.0,
                                                   flood_thresh=-13.0)
    # topo_stack = fetch_topography(aoi)  # not neccesary for Polesia region

    # stack all the datasets from various data sources
    # combined_stack = s1_stack.addBands(s2_stack)  # not neccesary for Polesia region
    # combined_stack = combined_stack.addBands(topo_stack)  # not neccesary for Polesia region
    combined_stack = s2_stack.addBands(flood_index)

    # Calculate indices on raw data
    print('create_data_stack(): Calculating indices...')
    combined_stack = compute_indices(combined_stack, date_list)

    if not max_min_values:
        # Normalise to between 0 and 1 using min-max.
        print('create_data_stack(): Calculating min-max values for each band...')
        max_dict, min_dict = get_min_max(combined_stack)
    else:
        # use previously calculated values
        print('create_data_stack(): using precalculated min-max values for each band...')
        max_dict = max_min_values[0]
        min_dict = max_min_values[1]

    print('create_data_stack(): Normalising all bands with min-max...')
    counter = 0
    band_names = combined_stack.bandNames().getInfo()

    for i in band_names:
        counter = counter + 1
        norm_band = combined_stack.select(i).unitScale(min_dict.get(i), max_dict.get(i))
        if counter == 1:
            normed_combined_stack = norm_band
        else:
            normed_combined_stack = normed_combined_stack.addBands(norm_band)
    print('create_data_stack(): Normalisation complete!')
    print('create_data_stack(): bye!')
    return normed_combined_stack, (max_dict, min_dict)


def get_min_max(stack):
    max_dict = stack.reduceRegion(**{
        'reducer': ee.Reducer.max(),
        'geometry': stack.geometry(),
        'scale': 20,
        'maxPixels': 1e9,
        'bestEffort': True}).getInfo()

    min_dict = stack.reduceRegion(**{
        'reducer': ee.Reducer.min(),
        'geometry': stack.geometry(),
        'scale': 20,
        'maxPixels': 1e9,
        'bestEffort': True}).getInfo()
    return max_dict, min_dict


def calc_EVI(combined_stack, band_date_ID_B8, band_date_ID_B4,  band_date_ID_B2, month):
    new_name = 'EVI_' + month
    EVI = combined_stack.expression('2.5 * ((B8-B4) / (B8 + 6 * B4-7.5 * B2 + 1))', {
        'B8': combined_stack.select(band_date_ID_B8).multiply(0.0001),
        'B4': combined_stack.select(band_date_ID_B4).multiply(0.0001),
        'B2': combined_stack.select(band_date_ID_B2).multiply(0.0001)}).rename(new_name)
    return EVI


def calc_AVI(combined_stack, band_date_ID_B8, band_date_ID_B4, month):
    new_name = 'AVI_'+month
    AVI = combined_stack.expression('(B8 * (1-B4)*(B8-B4))', {
        'B8': combined_stack.select(band_date_ID_B8).multiply(0.0001),
        'B4': combined_stack.select(band_date_ID_B4).multiply(0.0001)}).rename(new_name)
    return AVI


def compute_indices(combined_stack, date_list):
    """
    Computes the EVI and AVI indices for all months in the stack, except for those listed in the skip state.
    Currently only skips October.
    """
    print('compute_indices(): hello!')
    for i in date_list:
        month = i[0].split('-')[1]
        if month == '10':
            continue
        else:
            band_date_ID_B8 = 'S2_B8_'+i[0]+'_'+i[1]
            band_date_ID_B4 = 'S2_B4_'+i[0]+'_'+i[1]
            band_date_ID_B2 = 'S2_B2_'+i[0]+'_'+i[1]

            EVI = calc_EVI(combined_stack, band_date_ID_B8, band_date_ID_B4, band_date_ID_B2, month)
            AVI = calc_AVI(combined_stack, band_date_ID_B8, band_date_ID_B4, month)

            combined_stack = combined_stack.addBands(EVI)
            combined_stack = combined_stack.addBands(AVI)

            nameOfBands = combined_stack.bandNames().getInfo()
            nameOfBands.remove(band_date_ID_B8)
            nameOfBands.remove(band_date_ID_B4)
            nameOfBands.remove(band_date_ID_B2)
            combined_stack = combined_stack.select(nameOfBands)
    print('compute_indices(): bye!')
    return combined_stack


def fetch_sentinel1_flood_index(aoi, start_date_str, end_date_str, smoothing_radius=100.0, flood_thresh=-13.0):
    """
    Create a simple flood frequency layer from Sentinel-1 data (IW mode & VV)
    1) preprocess median non-winter (Mar-Oct) monthly composites using start/end dates
    2) apply spatial smoother (focal median) to get rid of the backscatter
    3) threshold monthly composite as binary flooded/not flooded map
    4) accumulate binary monthly flood maps, and normalise by dividing by number of months

    NOTE: this func could be modified to make use of 'VH' polorisation.Needs further exploration.
    This paper however suggests VV might be better for flood detection:
    https://iopscience.iop.org/article/10.1088/1755-1315/357/1/012034/pdf

    :: params ::
    :param aoi: ee.featurecollection.FeatureCollection, used to indicate AOI extent
    : param start_date_str: str, should be YYYY-MM-01 for start month (inclusive)
    : param end_date_str: str, should be YYYY-MM-01 for end month (inclusive)
    : param smoothing_radius: float, radius of smoothing kernel (metres)
    : param flood_thresh: float, VV values < flood_thresh are considered flooded.
    : returns : flood frequency map [0-1]
    """
    print('fetch_sentinel1_flood_index(): hello!')

    def s1_mask_border_noise(img):
        """
        Sentinel-1 data on GEE sometimes suffers from 'border noise' prior to May 2018 at swath edges.
        This needs to be masked in individual images via mapping before compositing images.
        This func is derived from the example code provided on the
        GEE ee.ImageCollection("COPERNICUS/S1_GRD") pages
        """
        edge = img.lt(-30.0)
        masked_img = img.mask().And(edge.Not())
        return img.updateMask(masked_img)

    band = 'VV'  # threshold needs changing if using VH and further testing

    # generate list of months
    start_date_list = [d.strftime('%Y-%m-%d') for d in pd.date_range(start=start_date_str,
                                                                     end=end_date_str, freq='MS')]
    # omit the winter months (jan, feb, nov, dec)
    start_date_list = [k for k in start_date_list if (int(k[5:7]) > 2) and (int(k[5:7]) < 11)]
    n_months = len(start_date_list)  # used for standardising

    # specify filters to apply to the GEE Sentinel-1 collection
    filters = [ee.Filter.listContains("transmitterReceiverPolarisation", band),
               ee.Filter.equals("instrumentMode", "IW"),
               ee.Filter.geometry(aoi)]

    # iteratively generate a monthly flood map and aggregate them
    for i, start_date in enumerate(start_date_list):
        end_date = ee.Date(start_date).advance(1, 'month')

        # load, preprocess and make composite
        s1_median = (ee.ImageCollection('COPERNICUS/S1_GRD')
                     .filterDate(start_date, end_date)
                     .filter(filters)
                     .select(band)
                     .map(s1_mask_border_noise)  # remove border noise
                     .median()
                     .clip(aoi.geometry()))

        # smooth image to get rid of backscatter noise
        s1_smoothed = s1_median.focal_median(smoothing_radius, 'circle', 'meters')

        # apply a flood masking threshold
        flood_map = s1_smoothed.lt(flood_thresh)

        # sum months
        if i == 0:
            flood_aggr = flood_map
        else:
            flood_aggr = flood_aggr.add(flood_map)

    # Standardise & rename
    flood_aggr = flood_aggr.divide(n_months).rename("s1_floodfreq")
    print('fetch_sentinel1_flood_index(): bye!')
    return flood_aggr


def fetch_sentinel2(aoi, date_list, s2_params, fill_mask_bkwd=True, fill_mask_fwd=True):
    """
    fetch a datastack of Sentinel-2 composites, with cloud/shadow masking applied.
    most of the code to do this is derived from here:
    https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
    Cloud masking filling attempts to fill with data from the month in the previous year, then any remaining gaps from
    the following year.

    :param aoi: ee.featurecollection.FeatureCollection, used to indicate AOI extent
    :param date_list: list of tuples of strings (i.e. [('a','b'),('c','d')]), used to define start & end of each
                      compositing period, expects 'YYYY-MM-DD' format
    :param s2_params: dict, contains parameters used for cloud & shadow masking
    :param fill_mask_bkwd: bool, if true, attempts to fill cloud gaps using last year's data
    :param fill_mask_fwd: bool, if true, attempts to fill cloud gaps using next year's data
    :return: ee.image.Image, stack of monthly composite images of bands specified in s2_params
    """
    print('fetch_sentinel2: hello!')

    def get_s2_sr_cld_col(aoi, start_date, end_date):
        """
        get & join the S2_SR and S2_CLOUD_PROBABILITY collections

        uses globals:
            CLOUD_FILTER: max cloud coverage (%) permitted in a scene

        :returns: ee.ImageCollection
        """
        # Import and filter S2 SR.
        s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

        # Import and filter s2cloudless.
        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                            .filterBounds(aoi)
                            .filterDate(start_date, end_date))

        # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
        return ee.ImageCollection(
            ee.Join.saveFirst('s2cloudless').apply(**{
                'primary': s2_sr_col,
                'secondary': s2_cloudless_col,
                'condition': ee.Filter.equals(**{
                    'leftField': 'system:index',
                    'rightField': 'system:index'
                })}))

    def add_cld_shdw_mask(img):
        """
        generate a cloud and shadow mask band
        uses globals:
            BUFFER: distance (m) used to buffer cloud edges
        :returns: img with added cloud mask, shadow mask, and cloud-shadow mask
        """

        # Add cloud component bands.
        img_cloud = add_cloud_bands(img)

        # Add cloud shadow component bands.
        img_cloud_shadow = add_shadow_bands(img_cloud)

        # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
        is_cld_shdw = (img_cloud_shadow.select('clouds')
                       .add(img_cloud_shadow.select('shadows')).gt(0)
                       )

        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
        # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
        is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER * 2 / 20)
                       .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                       .rename('cloudshadowmask'))

        # Add the final cloud-shadow mask to the image.
        return img_cloud_shadow.addBands(is_cld_shdw)

    def add_cloud_bands(img):
        """
        identify cloudy pixels using s2cloudless product probabilty band

        uses globals:
            CLD_PRB_THRESH: s2cloudless 'probability' band value > thresh = cloud

        :returns: img
        """
        # Get s2cloudless image, subset the probability band.
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

        # Condition s2cloudless by the probability threshold value.
        is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

        # Add the cloud probability layer and cloud mask as image bands.
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def add_shadow_bands(img):
        """
        identify cloud shadows from intersection of:
            (1) darkest NIR scene pixels below NIR_DRK_THRESH that are not water
            (2) projected location of cloud shadows based on CLD_PRJ_DIST*10

        uses globals:
            NIR_DRK_THRESH: if Band 8 (NIR) < NIR_DRK_THRESH = possible shadow
            CLD_PRJ_DIST:   max distnce [km or 100m?] from cloud edge for possible shadow

        :returns: img
        """
        # Identify water pixels from the SCL band.
        not_water = img.select('SCL').neq(6)

        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        SR_BAND_SCALE = 1e4
        dark_pixels = (img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE)
                       .multiply(not_water)
                       .rename('dark_pixels')
                       )

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (img.select('clouds')
                    .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
                    .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                    .select('distance')
                    .mask()
                    .rename('cloud_transform'))

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')

        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def apply_cld_shdw_mask(img):
        """
        apply the cloud & shadow mask
        :returns: img after application of cloud-shadow mask
        """

        # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
        not_cld_shdw = img.select('cloudshadowmask').Not()

        # Subset reflectance bands and update their masks, return the result.
        return img.select('B.*').updateMask(not_cld_shdw)

    def fill_cloud_gaps(img_orig, img_fill):
        """
        Where img_orig is masked (i.e. transparent null values) due to e.g. cloud masking,
        fill those gaps where possible using data from img_fill. any remaining gaps (i.e. cloudy in both images)
        are re-masked.

        :param img_orig: Image, to be filled
        :param img_fill: Image, used for filling
        :returns: img_new, img_orig after gap filling and remasking
        """
        img_new = img_orig.unmask(-99999)  # masked locations
        fill_pixels = img_new.eq(-99999)  # binary mask with value = 1 where we want to fill
        img_new = img_new.where(fill_pixels, img_fill)  # fill img_new with img_fill where fill_pixels==1
        mask = img_new.neq(
            -99999)  # -99999 will remain where no valid pixels in img_fill (i.e. cloudy in both), so remask
        img_new = img_new.mask(mask)
        return img_new

    # get individual variables from param dict
    CLOUD_FILTER = s2_params.get('CLOUD_FILTER')
    NIR_DRK_THRESH = s2_params.get('NIR_DRK_THRESH')
    CLD_PRJ_DIST = s2_params.get('CLD_PRJ_DIST')
    CLD_PRB_THRESH = s2_params.get('CLD_PRB_THRESH')
    BUFFER = s2_params.get('BUFFER')
    S2BANDS = s2_params.get('S2BANDS')

    # iteratively fetch each month of Sentinel-2 imagery and generate a median composite for the AOI
    for i, date_tuple in enumerate(date_list):
        new_band_names = [f'S2_{x}_{date_tuple[0]}_{date_tuple[1]}' for x in S2BANDS]
        start_date = ee.Date(date_tuple[0])
        end_date = ee.Date(date_tuple[1])

        # load and process collection for current year
        s2_sr_cld_col = get_s2_sr_cld_col(aoi, start_date, end_date)
        s2cldless_median = (s2_sr_cld_col.map(add_cld_shdw_mask)
                            .map(apply_cld_shdw_mask)
                            .select(S2BANDS)
                            .median()
                            .clip(aoi.geometry()))

        # try to cloud gap fill backwards in time
        if fill_mask_bkwd:
            s2_sr_cld_col_fill_bkwd = get_s2_sr_cld_col(aoi, start_date.advance(-1, 'year'),
                                                        end_date.advance(-1, 'year'))
            s2cldless_median_fill_bkwd = (s2_sr_cld_col_fill_bkwd.map(add_cld_shdw_mask)
                                          .map(apply_cld_shdw_mask)
                                          .select(S2BANDS)
                                          .median()
                                          .clip(aoi.geometry()))
            # sometimes S2 data used for bkwd filling is missing; in these cases, we cannot gap fill
            band_test = len(s2cldless_median_fill_bkwd.bandNames().getInfo())
            if band_test > 0:
                print(
                    f"fetch_sentinel2(): apply backward cloud gap filling for {date_tuple[0]} - {date_tuple[1]}...")
                # if bands are present apply cloud gap filling
                s2cldless_median = fill_cloud_gaps(img_orig=s2cldless_median,
                                                   img_fill=s2cldless_median_fill_bkwd)
            else:
                print(
                    f"fetch_sentinel2(): Skipping backward cloud gap filling for {date_tuple[0]} - {date_tuple[1]}; missing S2 data one year earlier")

        # try to cloud gap fill forwards in time
        if fill_mask_fwd:
            s2_sr_cld_col_fill_fwd = get_s2_sr_cld_col(aoi, start_date.advance(1, 'year'), end_date.advance(1, 'year'))
            s2cldless_median_fill_fwd = (s2_sr_cld_col_fill_fwd.map(add_cld_shdw_mask)
                                         .map(apply_cld_shdw_mask)
                                         .select(S2BANDS)
                                         .median()
                                         .clip(aoi.geometry()))
            # sometimes S2 data used for fwd filling is missing; in these cases, we cannot gap fill
            band_test = len(s2cldless_median_fill_fwd.bandNames().getInfo())
            if band_test > 0:
                print(f"fetch_sentinel2(): apply forward cloud gap filling for {date_tuple[0]} - {date_tuple[1]}...")
                # if bands are present apply cloud gap filling
                s2cldless_median = fill_cloud_gaps(img_orig=s2cldless_median,
                                                   img_fill=s2cldless_median_fill_fwd)
            else:
                print(
                    f"fetch_sentinel2(): Skipping forward cloud gap filling for {date_tuple[0]} - {date_tuple[1]}; missing S2 data one year later")

        # rename bands
        s2cldless_median = s2cldless_median.rename(new_band_names)
        # append to stack
        if i == 0:
            median_stack = s2cldless_median
        else:
            median_stack = median_stack.addBands(s2cldless_median)
    print('fetch_sentinel2(): bye!')
    return median_stack


def fetch_sentinel1(aoi, date_list):
    """

    ** NOT CURRENTLY IN USE! **

    fetch a datastack of Sentinel-1 composites.

    :param aoi: ee.featurecollection.FeatureCollection, used to indicate AOI extent
    :param date_list: list of tuples of strings (i.e. [('a','b'),('c','d')]), used to define start & end of each compositing period, expects 'YYYY-MM-DD' format
    :return: ee.image.Image, stack of composite images
    """
    print('fetch_sentinel1(): hello!')
    S1BANDS = ['VV', 'VH']

    # specify filters to apply to the GEE Sentinel-1 collection
    filters = [ee.Filter.listContains("transmitterReceiverPolarisation", "VV"),
               ee.Filter.listContains("transmitterReceiverPolarisation", "VH"),
               ee.Filter.equals("instrumentMode", "IW"),
               ee.Filter.geometry(aoi)]

    # iteratively fetch each month of Sentinel-1 imagery and generate a median composite for the AOI
    for i, date_tuple in enumerate(date_list):
        new_band_names = [f'S1_{x}_{date_tuple[0]}_{date_tuple[1]}' for x in S1BANDS]
        start_date = ee.Date(date_tuple[0])
        end_date = ee.Date(date_tuple[1])

        # load and filter collection
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterDate(start_date, end_date)
        s1 = s1.filter(filters)

        # make composite, clip and give sensible name
        s1_median = (s1.select(S1BANDS)
                     .median()
                     .clip(aoi.geometry())
                     .rename(new_band_names))

        # append to stack
        if i == 0:
            median_stack = s1_median
        else:
            median_stack = median_stack.addBands(s1_median)
    print('fetch_sentinel1(): bye!')
    return median_stack


def fetch_topography(aoi):
    """

    ** NOT CURRENTLY IN USE **

    Get static topographic layers
    elevation, slope and aspect from the NASA SRTM product 30m ('USGS/SRTMGL1_003').
    Also gets 'Global SRTM Topographic Diversity' ('CSP/ERGo/1_0/Global/SRTM_topoDiversity'),
    but I'm not convinced this is useful - 270m, looks a lot like coarse res slope/elevation

    :param aoi: ee.featurecollection.FeatureCollection, used to indicate AOI extent
    :return: ee.image.Image, stack of composite images with bands: ['elevation', 'aspect', 'slope', 'topographic_diversity']
    """
    print('fetch_topography(): hi!')
    # datasets & bands
    srtm = ee.Image('USGS/SRTMGL1_003')
    ds_topodiv = ee.Image('CSP/ERGo/1_0/Global/SRTM_topoDiversity')
    elevation = srtm.select('elevation').clip(aoi.geometry())
    topodiversity = ds_topodiv.select('constant').clip(aoi.geometry()).rename('topographic_diversity')

    # derive slope & aspect
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)

    # compile
    # stack = elevation.addBands(slope)
    stack = slope.addBands(aspect)
    # stack = stack.addBands(topodiversity)
    print('fetch_topography(): bye!')
    return stack