# Polesia-Landcover

[![DOI](https://zenodo.org/badge/391597735.svg)](https://zenodo.org/badge/latestdoi/391597735)

We use Google Earth Engine to carry out annual automated mapping of landcover in Polesia. This is achieved with an optimised Random Forest algorithm that is applied over a set of ‘complex’ and ‘simple’ classes that draw their classification characteristics from a customised annualised satellite data stack. The complex classes split out forest types and swamps into a finer degree as possible. The simple classes bin these finer ecological gradations into coniferous forests, deciduous forests, and swamps. We use a mixture of Sentinel 1 and 2 bands, some of which are turned into indices in order to reduce input bands

Install Google Earth Engine Python API: https://developers.google.com/earth-engine/guides/python_install-conda

Install of GeeMap and GeoPandas important information: https://geemap.org/installation/

Please note this is currently being tested as a beta version. If you have any issues feel free to raise them. Or even better, fix them and submit a pull request! 

The code is written in Python 3.9, with the following libraries required in addition to a standard Python install:
- Geemap
- Geopandas
- Pandas
- Numpy
- earthengine-api (Google Earth Engine)
- Shapely

See the .yml file for a full list of all libraries and dependencies.

## Overview of methods
The landcover data used in this work was provided by Dmitri Grummo and Valery Dombrovski from the National academy of Sciences, Belarus. The raw training data consisted of 30,410 polygons over the Pripyat Polesie region, divided into 77 classes. In addition to this original data, further mapping and class interpretation assistance was contributed by Iurii Strus of the Frankfurt Zoological Society. These data were transformed into a set of ‘Simple’ and ‘Complex’ classes that are possible to detect from orbit, using free data in a workflow that does not incur cloud computation costs.

Once the reduction to the complex classes was carried out, the polygons of the entire region were then randomly sampled, with a maximum of 200 polygons from each of the 13 classes selected, balanced by polygon area. This reduced the data load and allowed us to sample the Pripyat Polesie region in a representative manner and carry out manual correction of each polygon. Some of the issues encountered during the manual data cleaning process included: i) borders of regions mislocated by a significant degree over other classes, ii) miss-classification of classes and ii) invalid geometries and holes. Once the manual resolution of these issues had been completed, we were left with 2254 polygons distributed across the whole region. These cleaned, Complex classes, polygons were then transformed once again in order to give the Simple class polygons. The cleaned polygons of both classes were then sampled with random-points-in-polygons at 20 m spacing, with a maximum of 200 points per polygon. This is a required step in order to reduce the processing load within GEE when sampling the satellite data stack.

Prior to entry into the classification pipeline, the randomly generated points are resampled from within themselves to balance the number of samples across all classes to whatever the desired number of training data samples is (Training_points_balance.py).

The satellite data stack is built for the year on which training is to be carried out (2018), and is used to generate a classified landcover map for the same year (2018). For all subsequent years that are desired to be mapped, the datastack should be rebuilt for the year of interest. Landcover maps for years beyond 2018 can be mapped using this workflow, however it is important to note that the final classification maps produced will likely considerably decrease in accuracy unless the underlying training data is updated using new survey information that accounts for land use change.

The temporal element of the classification is achieved by building monthly median composites for each sensor used. These monthly composites are then used to generate the additional indices and analytical layers that contribute to the classification. We use the Sentinel 1 (S1) and Sentinel 2 (S2) platforms throughout, utilizing both atmospherically corrected (in the case of S2) bands ‘as is’ from GEE and three derived metrics:
1) Enhanced Vegetation Index (EVI) (Liu & Huete 1995): 2.5 * ((B8 – B4) / (B8 + 6 * B4 – 7.5 * B2 + 1))
2) Advanced Vegetation Index (AVI) (Cuizhen and Jiaguo 2005): (B8 * (1 – B4) * (B8 – B4))
3) Simple flood frequency (SFF): Monthly flood occurrence / Number of months

‘B’ refers to the S2 band number; those bands that are incorporated into a metric are removed from the overall data stack, in order to reduce processing load. The exception being October, in which indices are not used as they resulted in too much data loss.

SFF is derived from Sentinel 1 and evaluates all non-winter (March-October) months for the mapped year and the previous year (the number of prior years that are evaluated is adjustable). The prior year is included in order to capture the signature of flood meadows and forests that may not be inundated every year, but which are still considered ‘wet’. It uses a simple surface smoothness threshold approach to detect the presence of water (Aldhshan et al. 2019). The current surface smoothness threshold (-13) was determined using expert judgement and is specific to the Polesia training region; this should be adapted if the landcover mapping workflow is applied in a different geographical context. If water is present in a month, then the index of that pixel is advanced by 1, with the final value given normalised by the number of months in the record. Therefore, pixels that experience a flood every non-winter month will be 1, whilst those pixels that never flood will be 0. Similar to the S2 indices, by including the flood index we were able to reduce the number of input S1 bands and reduce data stack size with minimal information loss (~1 % accuracy).

Each input swath to the monthly composite is masked for cloud cover and cloud shadows during the compositing process. The final composite pixel for a month is the median of all contributing pixels values. If any data gaps are present in the composite, i.e., no pixel was cloud or shadow free for the month, then we attempt to fill that gap with data using a twostep gap filling process: first, we attempt to fill the gaps with data from the same month in the previous year (‘backward filling’), then any remaining data gaps (e.g. locations where cloud/cloud shadow gaps are present in both the current and previous years) are filled with data from the next year (‘forward filling’). Missing imagery in the GEE collections can also result in data gaps that the gap filling algorithm will attempt to fill. Unavoidable gaps will still remain in the final landcover map where no valid data are available for the entire month in the current, previous, and next year, as no data are available for filling.

We found that including topography and elevation into the data stack did not improve classification performance. This is likely due to the generally flat nature of the study region. In mountainous regions, topography would very likely be a beneficial inclusion in the data stack. Functions to include topography can be found in the “Development_scripts” sub-folder.

Testing found that of the freely available GEE classification algorithms, Random Forest (RF) performed the best. Functions to apply the other three available algorithms (CART, Gradient Boost and SVM) can be found in the “Development_scripts” sub-folder. By ‘freely available’ it is meant that you do not need to set up a Google Compute Engine instance to run a custom classification algorithm implementation, in order to apply it to the full data stack or and mapping tile.

The training data is loaded into GEE and samples every band of the provided data stack. This dataset is then randomly split into train (70%) and test (30%) elements. As suggested by the names, the RF classification model is trained on the train element and then tested against the test element. The parameters of the RF model are as follows: number of trees = as set by the optimization process, variables per split = none, minimum leaf population = 1, bag fraction = 0.5, maximum nodes = none and seed = 0.
In order to achieve best performance for the RF three optimization steps were taken: i) iteration of the original desired 19 classes to the final Complex and Simple classes in order to reduce class confusion and increase overall accuracy as per Appendix 1, ii) tree size experimentation and iii) training data size experimentation.

In general, increasing both tree size and training data size lead to an increase in RF performance. Tree size increases tended to give a relatively small boost (~0.5 to 1%) in performance, particularly once past the 100 trees mark. However, each step (250 more samples per class) in training data size led to a significant increase in performance, roughly in the order of 2 to 3%. As highlighted previously, processing load is a major constraint for free users of GEE and prevents us from simply maximising the tree size, training data size, and number of bands simultaneously. The substantial increase in RF performance with training data is the primary motivation behind attempting to reduce the number of bands being used whilst retaining as much information as possible (e.g. through indices, and use of limited time periods). Similarly, we prioritised training data size over tree size, and so it is always preferred to stay at the best training data size value and decrease the number of trees to the next optimal number. Note that the next optimal number of trees is not always the immediate next lower value of trees.

In order to allow processing over the very large Polesian region we break any provided target mapping extent (“fp_target_ext”) down into sub-processing regions, which are in turn divided into mapping tiles. Sub-processing regions are 1 degree in size and mapping tiles 0.15 degrees (of latitude-longitude). Any size of target extent can be provided, and the code will break it down into these tiles if the region is larger than the thresholds. One mapping tile takes approximately 5 minutes to classify and download, with a file size of approximately 75 kb. The tiling is carried out on your local machine and is not a functionality within GEE. The mapping function takes these tiles and sends them one at a time to the GEE server in order for the classification to be carried out, using the model generated from the training data. It may be possible to increase the size of the map-tiles if the processing load of the pipeline can be reduced. These tiles are temporarily generated in your export folder and are deleted once all downloads are complete. We do not send multiple tiles to GEE as there is a limit on how many concurrent operations and size of operations GEE will allow you to perform. Given that we are already pushing the limits of operation size, it is considered best to keep concurrent operations to 1.

For the Complex classes, with optimal settings of 150 trees and 2800 training data size, the resulting accuracy is: 76.4%. For the Simple classes, with optimal settings of 75 trees and 5000 training data size, the resulting accuracy is: 87.2%.

## User guide
Within “Classifcation_MAIN.py” there are the following toggles and paths to be aware of that you will need to customise to your own set up. All file paths are set up to build on top of your specified base directory, therefore the amount of files and folders you need to name and set up should be fairly minimal. For example, in the case of fp_train_ext, so long as you place the shapefile in your base directory all you need to do is make sure that name of the shapefile is the same as the name of the variable given. e.g. ‘Project_area.shp’.

![image](https://user-images.githubusercontent.com/19290575/144901724-20f534c2-b8eb-46fb-b6b0-03110f8cce88.png)

With all of the above settings and shapefiles in place, run “Optimization_MAIN.py” in order to start mapping. The script will first generate the data-stack and classification models, it will then carry out an accuracy assessment if so specified, before then tiling the specified mapping region and carrying out a classification of that region for the years desired. It will do all of this automatically with no user input. You just need to make sure that your internet connection remains up and that you have enough space to store the downloaded mapped tiles. There is no need to have the accuracy assessment toggled on if you are already happy with the classification model set-up.

Advanced users may wish to run their own optimization experiments, to do so use the script provided and set the various paths in the same manner as described for the main classification script. You will need to have run the training data class balancing script in order for the optimization of training data class sizes to function (or have all the class size shape files already available).

An advanced user may also wish to use the full accuracy assessment function. This can be easily implemented by switching in the relevant function name in the main classification script. The full assessment can be used to examine each class's performance and generates a range of the typical accuracy metrics in which you may be interested. This is particularly useful when considering which classes can be merged and separated.

Other options and further work that you may wish to consider are:

● Increasing the number of years used in the Sentinel 1 flood analysis, this is currently set to current year and year – 1. More years may provide interesting results and generate a flood layer that is of use in further downstream work. [Satellite_data_handling.py, line 44)

● Trying out different vegetation indices to those provided here. We selected two that provide the required performance stability with band reduction, but there are likely others that could do a better job with sufficient experimentation.

● Use a principal component analysis to analyse the band stack and carry out further dimensionality reduction, thereby allowing the use of more training data.

● Make more intelligent use of the S1 bands to generate surface roughness indices in order to add that information back into the stack (raw S1 bands had some use, but for the most part did not differentiate classes that well).


## Acknowledgements
Thanks to Dmitri Grummo and Valery Dombrovski of the National academy of Sciences (Belarus), and Iurii Strus of the Frankfurt Zoological Society. Their data, input and local knowledge has been invaluable in the preparation of ground data and the general approach to mapping. Thanks also to Adham Ashton-Butt (British Trust for Ornithology) for co-ordinating the work, providing data and constant feedback as to what a useful end-product needs. Sentinel 1 and 2 data is provided by the European Space Agency, with processing carried out on the Google Earth Engine platform. Finally, thanks to the Frankfurt Zoological Society and the Endangered Landscapes Programme for funding this work.

## References
Aldhshan, S.R., Mohammed, O.Z. and Shafri, H.M., (2019), November. Flash flood area mapping using sentinel-1 SAR data: a case study of eight upazilas in Sunamganj district, Bangladesh. In IOP Conference Series: Earth and Environmental Science, Vol. 357, No. 1, p. 012034. IOP Publishing.

Cuizhen W. and Jiaguo Q. (2005) Assessment of tropical forest degradation with canopy fractional cover from landsat ETM+ and IKONOS imagery, Earth Interactions, vol. 9, no. 22, pp.1-17.

Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. Remote Sensing of Environment.

Liu H. Q. & Huete A. (1995) Feedback based modification of the NDVI to minimize canopy background and atmospheric noise, IEEE Transactions on Geoscience and Remote Sensing, vol. 33, no. 2, pp. 457–465, 1995.
