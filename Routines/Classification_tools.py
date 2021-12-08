"""
This script contains all the classification functions to carry out landcover mapping in Polesia.

Uses random forest for simplicity, free data processing on GEE and ease of adaption to other targets.
"""
import numpy as np
from geemap import geemap
import ee
import os
import csv
import shutil
from Utilities import table_writer, get_max_acc, match_result_lengths, get_list_of_files, line_plot
from Training_data_handling import load_sample_training_data
from Satellite_data_handling import create_data_stack
from Processing_tools import tile_polygon

ee.Initialize()


def map_target_area(fp_target_ext, fp_export_dir, years_to_map, scale, clf_complex, clf_simple,
                    max_min_values_complex, max_min_values_simple):
    """
    Primary operational function that takes in a target area polygon and returns a land-cover classification
    for said polygon. Using both complex and simple classes.

    @param max_min_values_simple: Two dictionaries from the find_min_max function for simple classes
    @type max_min_values_simple: dict
    @param max_min_values_complex: Two dictionaries from the find_min_max function for complex classes
    @type max_min_values_complex: dict
    @param clf_simple: classification object for the simple classes
    @type clf_simple: object
    @param clf_complex: classification object for the complex classes
    @type clf_complex: object
    @param scale: Desired output mapping scale
    @type scale: int
    @param years_to_map: List of integers that are the years to classify and return a map for
    @type years_to_map: list
    @param fp_export_dir: Destination directory for geotifs
    @type fp_export_dir: str
    @param fp_target_ext: Full path to the polygon that describes the desired region for mapping
    @type fp_target_ext: str

    """
    print('map_target_area(): hello!')
    print('map_target_area(): Generating processing areas...')
    process_size = 1.0
    process_dir = tile_polygon(fp_target_ext, process_size, fp_export_dir, 'processing_areas/')
    process_list = get_list_of_files(process_dir, ".shp")

    # Tile processing areas to allow GEE extraction
    tile_size = 0.15
    for i in process_list:
        print('map_target_area(): Generating tiles for processing area', str(i) + '...')
        process_num = i.split('/')[-1].split('.')[0]
        tile_dir = tile_polygon(i, tile_size, fp_export_dir, process_num + '_tiles/')

        # Run the classification for the desired years over the processing areas and tiles
        tile_list = get_list_of_files(fp_export_dir + process_num + '_tiles/', ".shp")
        for j in years_to_map:
            print('\nmap_target_area(): mapping year', str(j) + '...')
            for k in tile_list:
                print('\nmap_target_area(): mapping tile', str(k) + '...')
                try:
                    yearly_classifier_function(j, k, process_num, scale,
                                               fp_export_dir, clf_complex, 'Complex', max_min_values_complex)
                except Exception as e:
                    print('map_target_area(): Tile ' + k + ' failed to process Complex with exception:')
                    print(e)
                    continue

                try:
                    yearly_classifier_function(j, k, process_num, scale,
                                               fp_export_dir, clf_simple, 'Simple', max_min_values_simple)
                except Exception as e:
                    print('map_target_area(): Tile ' + k + ' failed to process Simple with exception:')
                    print(e)
                    continue

    # Clean up
    try:
        shutil.rmtree(process_dir)
    except Exception as e:
        print('Failed to clear processing area directory (try doing it manually) with error:')
        print(e)

    for i in process_list:
        try:
            shutil.rmtree(i)
        except Exception as e:
            print('Failed to clear tile directory ' + i + ' (try doing it manually) with error:')
            print(e)
    print('map_target_area(): bye!')


def yearly_classifier_function(year, k, process_num, scale,
                               fp_export_dir, clf, run_type, max_min_values):
    """
    @param year: Year to be mapped
    @type year: int
    @param k: File path for the tile being mapped
    @type k: str
    @param process_num: The number of the processing area being mapped
    @type process_num: str
    @param scale: Pixel size of mapping in meters
    @type scale: int
    @param fp_export_dir: Filepath to export directory
    @type fp_export_dir: str
    @param clf: Earth engine classification model
    @type clf: object
    @param run_type: Is it being run on complex or simple classes?
    @type run_type: str
    @param max_min_values: The max-min values used in normalisation, from the whole dataset not just this tile.
    @type max_min_values: dict
    @return: Does not return anything, carries out export via apply_random_forest
    @rtype: None
    """
    print('yearly_classifier_function(): hello!')
    year = str(year)
    tile_num = k.split('/')[-1].split('.')[0]
    export_name = 'PArea' + process_num + '_tile' + tile_num + '_RF_' + year + '_' + run_type
    check_name = export_name + '.tif'

    if not os.path.exists(fp_export_dir+check_name):
        aoi = geemap.shp_to_ee(k)
        date_list = [(year + '-03-01', year + '-03-30'),
                     (year + '-04-01', year + '-04-30'), (year + '-05-01', year + '-05-31'),
                     (year + '-06-01', year + '-06-30'), (year + '-07-01', year + '-07-30'),
                     (year + '-10-01', year + '-10-30')]

        tile_stack, max_min_values_output = create_data_stack(aoi, date_list, year, max_min_values)
        training_bands = tile_stack.bandNames().getInfo()

        apply_random_forest(export_name, training_bands, k, tile_stack, scale, fp_export_dir, clf)
    else:
        pass
    print('yearly_classifier_function(): bye!')


def RF_model_and_train(year, scale, label, aoi, fp_train_points, trees):
    """
    @param year: Year on which model is to be trained
    @type year: int
    @param scale: Pixel size on which to train and classify
    @type scale: int
    @param label: Name of the column that contains the classification labels
    @type label: str
    @param aoi: GEE polygon object that describes the target area
    @type aoi: object
    @param fp_train_points: Full file path to the training points shapefile
    @type fp_train_points: str
    @param trees: Number of trees to use in the random forest model
    @type trees: int
    @return: The classification model, the test dataset, max min values for later use in normalisation
    @rtype: object, object, dict
    """
    print('RF_model_and_train(): hello!')
    year = str(year)
    date_list = [(year + '-03-01', year + '-03-30'),
                 (year + '-04-01', year + '-04-30'), (year + '-05-01', year + '-05-31'),
                 (year + '-06-01', year + '-06-30'), (year + '-07-01', year + '-07-30'),
                 (year + '-10-01', year + '-10-30')]
    stack, max_min_values_output = create_data_stack(aoi, date_list, year, None)
    band_names = stack.bandNames()
    trainingbands = band_names.getInfo()

    train, test = load_sample_training_data(fp_train_points, trainingbands, stack, scale, label)
    clf = generate_RF_model(trees, train,  label, trainingbands)
    print('RF_model_and_train(): bye!')
    return clf, test, max_min_values_output


def generate_RF_model(trees_num, train, class_col_name, training_bands):
    """
    @param trees_num: Number of tress to be used in the random forest model
    @type trees_num: int
    @param train: The training data on which to build the model
    @type train: object
    @param class_col_name: The column in the training data that contains the classes
    @type class_col_name: str
    @param training_bands: The prepped bands on which training/classification will be carried out
    @type training_bands: list
    @return: The trained classification model
    @rtype: object
    """
    print('generate_RF_model(): hello!')
    init_params = {"numberOfTrees": trees_num,
                   "variablesPerSplit": None,
                   "minLeafPopulation": 1,
                   "bagFraction": 0.5,
                   "maxNodes": None,
                   "seed": 0}
    clf = ee.Classifier.smileRandomForest(**init_params).train(train, class_col_name, training_bands)
    print('generate_RF_model(): bye!')
    return clf


def apply_random_forest(export_name, training_bands, fp_target_ext, stack, scale, fp_export_dir, clf):
    """
    https://developers.google.com/earth-engine/apidocs/ee-classifier-smilerandomforest
    It downloads to the local export directory the applied RF classification.

    @param export_name: Save name of the tile to be (no extension)
    @type export_name: str
    @param training_bands: The prepped bands on which training/classification will be carried out
    @type training_bands: list
    @param fp_target_ext: Full filepath to the shapefile which outlines the training data extent
    @type fp_target_ext: str
    @param stack: The prepped and analysis ready satellite bands in a stack
    @type stack: object
    @param scale: The pixel size at which to carry out the analysis
    @type scale: int
    @param fp_export_dir: Filepath to the desired output location of the mapped tiles
    @type fp_export_dir: str
    @param clf: The prepared classification model
    @type clf: object
    @return: Carries out the export to the specified location
    @rtype: None
    """
    print('apply_random_forest(): hello!')
    # Carry out the Random Forest
    print('apply_random_forest(): Using random forest to classify region', export_name+'...')
    target_area = geemap.shp_to_ee(fp_target_ext)
    target_stack = stack.clip(target_area)
    classified = target_stack.select(training_bands).classify(clf)

    # Export results to local
    file_out = fp_export_dir+export_name+'.tif'
    roi = target_area.geometry()
    geemap.ee_export_image(classified, filename=file_out, scale=scale, file_per_band=False, region=roi)
    print('apply_random_forest(): Random forest classification complete for', export_name+'!')
    print('apply_random_forest(): bye!')


def accuracy_assessment_basic(clf, test, export_name, label):
    print('Carrying out basic accuracy assessment for', export_name+'...')
    tested = test.classify(clf)
    test_accuracy = tested.errorMatrix(label, 'classification')
    test_overall_accuracy = np.around(test_accuracy.accuracy().getInfo(), decimals=4)
    print('Test data performance overall accuracy:', test_overall_accuracy)
    return test_overall_accuracy


def accuracy_assessment_full(clf, test, export_name, label, fp_export_dir):
    print('Carrying out full accuracy assessment for', export_name+'...')
    # Run training data assessment
    # Confusion matrix representing re-substitution accuracy
    trainAccuracy = clf.confusionMatrix()
    resub_error_matrix = trainAccuracy.getInfo()

    training_overall_accuracy = np.around(trainAccuracy.accuracy().getInfo(), decimals=4)
    print('Training data performance overall accuracy:', training_overall_accuracy)

    kappa_train = np.around(trainAccuracy.kappa().getInfo(), decimals=4)
    print('Kappa coefficient for training data (-1 to 1) =', kappa_train)

    producers_train = trainAccuracy.producersAccuracy().getInfo()
    table_writer(producers_train, 'Producers_acc_train_', export_name)

    consumers_train = trainAccuracy.consumersAccuracy().getInfo()
    table_writer(consumers_train, 'Consumers_acc_train_', export_name)

    training_csv = os.path.join(fp_export_dir, 'Training_confusion_matrix'+export_name+'.csv')
    with open(training_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(resub_error_matrix)

    # Run test data assessment
    tested = test.classify(clf)
    test_accuracy = tested.errorMatrix(label, 'classification')

    test_error_matrix = test_accuracy.getInfo()
    table_writer(test_error_matrix, 'Error_matrix_test_', export_name)

    test_overall_accuracy = np.around(test_accuracy.accuracy().getInfo(), decimals=4)
    print('Test data performance overall accuracy:', test_overall_accuracy)

    kappa_test = np.around(test_accuracy.kappa().getInfo(), decimals=4)
    print('Kappa coefficient for test data (-1 to 1) =', kappa_test)

    producers_test = test_accuracy.producersAccuracy().getInfo()
    table_writer(producers_test, 'Producers_acc_test_', export_name)

    consumers_test = test_accuracy.consumersAccuracy().getInfo()
    table_writer(consumers_test, 'Consumers_acc_test_', export_name)
    return test_overall_accuracy


def training_data_size_optimize(root_train_fpath, aoi, training_type, label, year, plots_out_dir, scale):
    """
    @param root_train_fpath: The top directory path that will contain the output training data
    @type root_train_fpath: str
    @param aoi: GEE polygon object of the training data area
    @type aoi: object
    @param training_type: Name of the test, usually simple or complex (classes)
    @type training_type: str
    @param label: Name of the column that contains your classes in the training data attribute table
    @type label: str
    @param year: Year on which the training data is being used
    @type year: int
    @param plots_out_dir: Filepath to the directory in which your want to plots to output
    @type plots_out_dir: str
    @param scale: Pixel size at which the classification is being carried out in meters
    @type scale: int
    @return: None
    @rtype: None
    """
    year = str(year)
    date_list = [(year + '-03-01', year + '-03-30'),
                 (year + '-04-01', year + '-04-30'), (year + '-05-01', year + '-05-31'),
                 (year + '-06-01', year + '-06-30'), (year + '-07-01', year + '-07-30'),
                 (year + '-10-01', year + '-10-30')]
    stack, max_min_values_output = create_data_stack(aoi, date_list, year, None)
    band_names = stack.bandNames()
    trainingbands = band_names.getInfo()

    training_test_vals = [250, 500, 750, 1000, 1250, 1500, 1750, 2000,
                          2250, 2500, 2750, 2800, 2850, 2900, 3000,
                          3250, 3500, 4000, 4250, 4500, 4750, 5000]
    result_trainsize_vals = []
    for i in training_test_vals:
        try:
            fp = root_train_fpath + training_type + '_points_' + str(i) + ".shp"
            train, test = load_sample_training_data(fp, trainingbands, stack, scale, label)
            clf = generate_RF_model(150, train, label, trainingbands)
            val = accuracy_assessment_basic(clf,
                                            test,
                                            training_type + '_' + str(i),
                                            label)
            result_trainsize_vals.append(val)
        except Exception as e:
            print('Training size limit hit with exception: ', e)
            print('Nan returned')
            val = np.nan
            result_trainsize_vals.append(val)
            break

    training_test_vals = match_result_lengths(training_test_vals, result_trainsize_vals)
    max_acc, training_size = get_max_acc(training_test_vals, result_trainsize_vals)

    x_label = 'Training data size ' + training_type
    line_plot(training_test_vals, result_trainsize_vals, x_label, plots_out_dir)
    print('Best training size for '+training_type + ' = ' + str(training_size))


def trees_size_optimize(fp_train_points, aoi, training_type, label, plots_out_dir, scale):
    """
    @param fp_train_points: Filepath to the training data points of the size you want (usually 2000)
    @type fp_train_points: str
    @param aoi: GEE polygon object of the training data area
    @type aoi: object
    @param training_type: Name of the test, usually simple or complex (classes)
    @type training_type: str
    @param label: Name of the column that contains your classes in the training data attribute table
    @type label: str
    @param plots_out_dir: Filepath to the directory in which your want to plots to output
    @type plots_out_dir: str
    @param scale: Pixel size at which the classification is being carried out in meters
    @type scale: int
    @return: None
    @rtype: None
    """
    year = str(2018)
    date_list = [(year + '-03-01', year + '-03-30'),
                 (year + '-04-01', year + '-04-30'), (year + '-05-01', year + '-05-31'),
                 (year + '-06-01', year + '-06-30'), (year + '-07-01', year + '-07-30'),
                 (year + '-10-01', year + '-10-30')]
    stack, max_min_values_output = create_data_stack(aoi, date_list, year, None)
    band_names = stack.bandNames()
    trainingbands = band_names.getInfo()

    train, test = load_sample_training_data(fp_train_points, trainingbands, stack, scale, label)

    trees_test_vals = [25, 50, 75, 100, 125, 150, 175, 200, 225,
                       250, 275, 300, 325, 350, 375, 400, 425, 450]
    result_trees_vals = []
    for i in trees_test_vals:
        try:
            clf = generate_RF_model(i, train, label, trainingbands)
            val = accuracy_assessment_basic(clf,
                                            test,
                                            training_type + '_' + str(i),
                                            label)
            result_trees_vals.append(val)

        except Exception as e:
            print('Tree size limit hit with exception: ', e)
            print('Nan returned')
            val = np.nan
            result_trees_vals.append(val)
            break

    trees_test_vals = match_result_lengths(trees_test_vals, result_trees_vals)
    max_acc, tree_size = get_max_acc(trees_test_vals, result_trees_vals)

    x_label = 'Trees ' + training_type
    line_plot(trees_test_vals, result_trees_vals, x_label, plots_out_dir)
    print('Best tree size for '+training_type + ' = ' + str(tree_size))