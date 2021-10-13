import ee
from geemap import geemap

def fetch_sentinel1(aoi, start_date_list):
    """
    fetch a datastack of Sentinel-1 monthly composites.
    
    :param aoi: ee.featurecollection.FeatureCollection, used to indicate AOI extent 
    :param start_date_list: list, strings used to define start of each month, expects 'YYYY-MM-01' format
    :return: ee.image.Image, stack of monthly composite images
    """
    print('fetch_sentinel1(): hello!')
    
    # specify filters to apply to the GEE Sentinel-1 collection
    filters = [ee.Filter.listContains("transmitterReceiverPolarisation", "VV"),
           ee.Filter.listContains("transmitterReceiverPolarisation", "VH"),
           ee.Filter.equals("instrumentMode", "IW"),
           ee.Filter.geometry(aoi)]
    
    # iteratively fetch each month of Sentinel-1 imagery and generate a median composite for the AOI
    for i, start_date in enumerate(start_date_list):
        print(f'fetch_sentinel1(): processing month {start_date}')
        end_date = ee.Date(start_date).advance(1, 'month')

        # load and filter collection
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
             .filterDate(start_date, end_date)   
        s1 = s1.filter(filters)

        # make composite, clip and give sensible name
        s1_median = (s1.select('VV', 'VH')
                      .median()
                      .clip(aoi.geometry())
                      .rename(f'S1_VV_{start_date[0:7]}', 
                              f'S1_VH_{start_date[0:7]}'))

        # append to stack
        if i == 0:
            median_stack = s1_median
        else:
            median_stack = median_stack.addBands(s1_median)
    
    print('fetch_sentinel1(): bye!')    
    return median_stack


def fetch_sentinel2(aoi, start_date_list, s2_params):

    """
    fetch a datastack of Sentinel-2 monthly composites, with cloud/shadow masking applied.
    most of the code to do this is derived from here:
    https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
    
    :param aoi: ee.featurecollection.FeatureCollection, used to indicate AOI extent 
    :param start_date_list: list, strings used to define start of each month, expects 'YYYY-MM-01' format
    :param s2_params: dict, contains parameters used for cloud & shadow masking   
    :return: ee.image.Image, stack of monthly composite images with bands: 
             ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    """ 
    
    print('fetch_sentinel2(): hello!')
    
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
        # mdj TODO: confirmation that BUFFER is in [m]
        #           focal_max() default units = pixels (and pix res is 10m)
        #           so if BUFFER = 100
        #           100 * 0.1 = 10 pixels
        #           10 pix * 10 [pix res] = 100m
        is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER*2/20)
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
        dark_pixels = (img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE)
                       .multiply(not_water)
                       .rename('dark_pixels')
                      )

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        # mdj TODO: check why CLD_PRJ_DIST*10? i'm not convinced CLD_PRJ_DIST is in km.. 
        #           'clouds' is 10m res. 
        #           directionalDistanceTransform 2nd arg 'maxDistance' is in pixels
        #           so actually CLD_PRJ_DIST units = 100s of m?
        cld_proj = (img.select('clouds')
                    .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
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
    
    
    # get individual variables from param dict
    CLOUD_FILTER   = s2_params.get('CLOUD_FILTER')
    NIR_DRK_THRESH = s2_params.get('NIR_DRK_THRESH')
    CLD_PRJ_DIST   = s2_params.get('CLD_PRJ_DIST')
    CLD_PRB_THRESH = s2_params.get('CLD_PRB_THRESH')
    BUFFER         = s2_params.get('BUFFER')
    # mdj: S2BANDS is currently hard-coded here as not sure how to dynamically rename bands
    #S2BANDS        = s2_params.get('S2BANDS')
    S2BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    
    # iteratively fetch each month of Sentinel-2 imagery and generate a median composite for the AOI
    for i, start_date in enumerate(start_date_list):
        #mnth=i+1
        print(f'fetch_sentinel2(): processing month {start_date}')
        end_date = ee.Date(start_date).advance(1, 'month')

        # load and filter collection
        s2_sr_cld_col = get_s2_sr_cld_col(aoi, start_date, end_date)

        # do cloud processing, make composite, clip and give sensible names
        s2cldless_median = (s2_sr_cld_col.map(add_cld_shdw_mask)
                                     .map(apply_cld_shdw_mask)
                                     .select(S2BANDS) 
                                     .median()
                                     .clip(aoi.geometry())
                                     .rename(f'S2_B2_{start_date[0:7]}', 
                                             f'S2_B3_{start_date[0:7]}', 
                                             f'S2_B4_{start_date[0:7]}', 
                                             f'S2_B5_{start_date[0:7]}',
                                             f'S2_B6_{start_date[0:7]}', 
                                             f'S2_B7_{start_date[0:7]}', 
                                             f'S2_B8_{start_date[0:7]}', 
                                             f'S2_B8A_{start_date[0:7]}',
                                             f'S2_B11_{start_date[0:7]}', 
                                             f'S2_B12_{start_date[0:7]}'))    
        # append to stack
        if i == 0:
            median_stack = s2cldless_median
        else:
            median_stack = median_stack.addBands(s2cldless_median)
    
    print('fetch_sentinel2(): bye!')    
    return median_stack


def fetch_topography(aoi):
    """
    Get static topographic layers 
    elevation, slope and aspect from the NASA SRTM product 30m ('USGS/SRTMGL1_003').
    Also gets 'Global SRTM Topographic Diversity' ('CSP/ERGo/1_0/Global/SRTM_topoDiversity'), 
    but I'm not convinced this is useful - 270m, looks a lot like coarse res slope/elevation

    :param aoi: ee.featurecollection.FeatureCollection, used to indicate AOI extent 
    :return: ee.image.Image, stack of composite images with bands: ['elevation', 'aspect', 'slope', 'topographic_diversity']     
    """
    print('fetch_topography(): hi!')
    #datasets & bands
    srtm = ee.Image('USGS/SRTMGL1_003')
    ds_topodiv = ee.Image('CSP/ERGo/1_0/Global/SRTM_topoDiversity')
    elevation = srtm.select('elevation').clip(aoi.geometry())
    topodiversity = ds_topodiv.select('constant').clip(aoi.geometry()).rename('topographic_diversity')
    
    # derive slope & aspect
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)

    # compile
    stack = elevation.addBands(slope)
    stack = stack.addBands(aspect)
    stack = stack.addBands(topodiversity)
    print('fetch_topography(): bye!')
    return stack


def map_topography(stack, lat=51.85, lon=27.8, elv_rng=(100.0,200.0), asp_rng=(0.0,360.0), slp_rng=(0.0, 10.0), topdiv_rng=(0,0.15)):
    """
    Quick mapping function for debugging topo data 
    : param stack: Image, stack containing topographic composite images for AOI
    : param lat: float, map central latitude
    : param lon: float, map central longitude
    : params elv_rng, asp_rng, slp_rng, topdiv_rng: tuples of floats, min/max z-values for elevation, aspect, slope, and topo diversity, respectively. 
                                                    Defaults are suited to the Polesia training AOI    
    : return : 
    """
    Map = geemap.Map(center=(lat, lon), zoom=9)
    Map.add_basemap('SATELLITE')
    Map.addLayer(stack, {'min': elv_rng[0],'max': elv_rng[1], 'bands': ['elevation']}, 'elevation')
    Map.addLayer(stack, {'min': asp_rng[0],'max': asp_rng[1], 'bands': ['aspect']}, 'aspect')
    Map.addLayer(stack, {'min': slp_rng[0],'max': slp_rng[1], 'bands': ['slope']}, 'slope')
    Map.addLayer(stack, {'min': topdiv_rng[0],'max': topdiv_rng[1], 'bands': ['topographic_diversity']}, 'topographic_diversity')
    return Map


def map_sentinel1(stack, start_date_list, lat=51.85, lon=27.8):
    """
    Quick mapping function for debugging S1 data (NOTE: VH duplicated in Green & Blue)
    : param stack: Image, stack of S1 composite images for AOI
    : param start_date_list: list, strings used to define start of each month, expects 'YYYY-MM-01' format
    : param lat: float, map central latitude
    : param lon: float, map central longitude
    : return : 
    """
    Map = geemap.Map(center=(lat, lon), zoom=9)
    Map.add_basemap('SATELLITE')
    for i, start_date in enumerate(start_date_list):
        vis = {'min': -50,'max': 1, 'bands': [f'S1_VV_{start_date[0:7]}', 
                                              f'S1_VH_{start_date[0:7]}', 
                                              f'S1_VH_{start_date[0:7]}']}
        Map.addLayer(stack, vis, f'S1_{start_date}')
    return Map


def map_sentinel2(stack, start_date_list, lat=51.85, lon=27.8):
    """
    Quick mapping function for debugging S2 data - RGB only
    : param stack: Image, stack of S2 composite images for AOI
    : param start_date_list: list, strings used to define start of each month, expects 'YYYY-MM-01' format
    : param lat: float, map central latitude
    : param lon: float, map central longitude
    : return : 
    """
    Map = geemap.Map(center=(lat, lon), zoom=9)
    Map.add_basemap('SATELLITE')
    for i, start_date in enumerate(start_date_list):
        vis = {'min': -0.0,'max': 3000, 'bands': [f'S2_B4_{start_date[0:7]}', 
                                                  f'S2_B3_{start_date[0:7]}', 
                                                  f'S2_B2_{start_date[0:7]}']}
        Map.addLayer(stack, vis, f'S2_{start_date}')
    return Map


def create_data_stack(aoi, start_date_list, s2_params):
    """ 
    convience function to compile and combine all distinct dataset sub-stacks
    * Sentinel 1 data bands: 'VV', 'VH' [monthly]
    * Sentinel 2 data bands: 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12' [monthly]
    * topography data bands: slope, aspect & elevation [static, 30m SRTM derived], topographic diversity [static, 270m]

    :param aoi: ee.featurecollection.FeatureCollection, used to indicate AOI extent 
    :param start_date_list: list, strings used to define start of each month, expects 'YYYY-MM-01' format
    :param s2_params: dict, contains parameters used for cloud & shadow masking   
    :return: ee.image.Image

    """
    s1_stack = fetch_sentinel1(aoi, start_date_list)
    s2_stack = fetch_sentinel2(aoi, start_date_list, s2_params)
    topo_stack = fetch_topography(aoi)

    combined_stack = s1_stack.addBands(s2_stack)
    combined_stack = combined_stack.addBands(topo_stack)

    return combined_stack

