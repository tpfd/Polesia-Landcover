"""
Functions to run the v2 of a classifier pipeline
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
# sys.path.append("/home/markdj/repos/Polesia-Landcover/data_stack_testing/")
sys.path.append("C:/Users/tpfdo/OneDrive/Documents/GitHub/Polesia-Landcover/data_stack_testing/")
import ee
import numpy as np
import os
import csv
from geemap import geemap
from data_stack_tools_v1 import fetch_sentinel1_v2, fetch_sentinel2_v2, map_topography, create_data_stack_v2

ee.Initialize()

"""
Functions

Note that the accuracy assessment is independent of the apply_random_forest function, and is carried out over the whole 
training/test set. So no need to test accuracy for every tile.
"""


def load_sample_training_data(fp_train_points, target_bands):
    """
    This function takes a filepath to a point shape file of your training data.
    It returns a training and test sampled dataset with a 70-30 split.
    It operates over the global raster data stack.
    """
    print('Loading training data:', fp_train_points+'...')
    training_points = geemap.shp_to_ee(fp_train_points)
    data = stack.select(target_bands).sampleRegions(collection=training_points,
                                                    properties=[label],
                                                    scale=scale)

    # Split into train and test
    split = 0.7
    data = data.randomColumn(seed=0)
    train = data.filter(ee.Filter.lt('random', split))
    test = data.filter(ee.Filter.gte('random', split))
    print('Training data loaded and ready!')
    return train, test


def apply_random_forest(train, export_name, trees_num, training_bands):
    """
    https://developers.google.com/earth-engine/apidocs/ee-classifier-smilerandomforest

    This function takes a train dataset from load_sample_training_data, and a simple string name e.g. 'RF_simple'.
    It returns the prepped classification EE function.
    It downloads to the local export directory the applied RF classification.
    """
    print('Setting up random forest classifier for', export_name+'...')
    init_params = {"numberOfTrees": trees_num,
                   "variablesPerSplit": None,
                   "minLeafPopulation": 1,
                   "bagFraction": 0.5,
                   "maxNodes": None,
                   "seed": 0}

    clf = ee.Classifier.smileRandomForest(**init_params).train(train, label, training_bands)

    # Carry out the Random Forest
    #print('Using random forest to classify region...')
    #target_area = geemap.shp_to_ee(fp_target_ext)
    #target_stack = stack.clip(target_area)
    #classified = target_stack.select(trainingbands).classify(clf)

    # Export results to local
    #file_out = fp_export_dir+export_name+'.tif'
    #roi = target_area.geometry()
    #geemap.ee_export_image(classified, filename=file_out, scale=scale, file_per_band=False, region=roi)
    #print('Random forest classification complete for', export_name+'!')
    return clf


def apply_CART(train, export_name):
    """
    https://developers.google.com/earth-engine/apidocs/ee-classifier-smilecart
    """
    print('Setting up CART classifier for', export_name + '...')
    init_params = {"maxNodes": None,
                   "minLeafPopulation": None}

    clf = ee.Classifier.smileCart(**init_params).train(train, label, trainingbands)

    # Carry out the Random Forest
    print('Using CART to classify region...')
    target_area = geemap.shp_to_ee(fp_target_ext)
    target_stack = stack.clip(target_area)
    classified = target_stack.select(trainingbands).classify(clf)

    # Export results to local
    file_out = fp_export_dir + export_name + '.tif'
    roi = target_area.geometry()
    geemap.ee_export_image(classified, filename=file_out, scale=scale, file_per_band=False, region=roi)
    print('CART classification complete for', export_name + '!')
    return clf


def apply_SVM(train, export_name):
    """
    https://developers.google.com/earth-engine/apidocs/ee-classifier-libsvm
    """
    print('Setting up SVM classifier for', export_name + '...')
    init_params = {"decisionProcedure": "Voting",
                   "svmType": "C_SVC",
                   "kernelType": "LINEAR",
                   "shrinking": True,
                   "degree": None,
                   "gamma": None,
                   "coef0": None,
                   "cost": None,
                   "nu": None,
                   "terminationEpsilon": None,
                   "lossEpsilon": None,
                   "oneClass": None}

    clf = ee.Classifier.libsvm(**init_params).train(train, label, trainingbands)

    # Carry out the Random Forest
    print('Using SVM to classify region...')
    target_area = geemap.shp_to_ee(fp_target_ext)
    target_stack = stack.clip(target_area)
    classified = target_stack.select(trainingbands).classify(clf)

    # Export results to local
    file_out = fp_export_dir + export_name + '.tif'
    roi = target_area.geometry()
    geemap.ee_export_image(classified, filename=file_out, scale=scale, file_per_band=False, region=roi)
    print('SVM classification complete for', export_name + '!')
    return clf


def apply_gradient_tree_boost(train, export_name):
    """
    https://developers.google.com/earth-engine/apidocs/ee-classifier-smilegradienttreeboost

    This function takes a train dataset from load_sample_training_data, and a simple string name e.g. 'RF_simple'.
    It returns the prepped classification EE function.
    It downloads to the local export directory the applied RF classification.
    """
    print('Setting up gradient tree boost classifier for', export_name+'...')
    init_params = {"numberOfTrees": 100,
                   "shrinkage": 0.005,
                   "samplingRate": 0.7,
                   "maxNodes": None,
                   "loss": "LeastAbsoluteDeviation",
                   "seed": 0}

    clf = ee.Classifier.smileGradientTreeBoost(**init_params).train(train, label, trainingbands)

    # Carry out the Random Forest
    print('Using gradient tree boost to classify region...')
    target_area = geemap.shp_to_ee(fp_target_ext)
    target_stack = stack.clip(target_area)
    classified = target_stack.select(trainingbands).classify(clf)

    # Export results to local
    file_out = fp_export_dir+export_name+'.tif'
    roi = target_area.geometry()
    geemap.ee_export_image(classified, filename=file_out, scale=scale, file_per_band=False, region=roi)
    print('GTB classification complete for', export_name+'!')
    return clf


def table_writer(table, table_name, export_name):
    out_csv = os.path.join(fp_export_dir, table_name + export_name + '.csv')
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)
    print(table_name+export_name, 'written out.')


def accuracy_assessment(clf, test, export_name):
    print('Carrying out accuracy assessment for', export_name+'...')
    # Run training data assessment
    # Confusion matrix representing re-substitution accuracy
    trainAccuracy = clf.confusionMatrix()
    #resub_error_matrix = trainAccuracy.getInfo()

    #training_overall_accuracy = np.around(trainAccuracy.accuracy().getInfo(), decimals=4)
    #print('Training data performance overall accuracy:', training_overall_accuracy)

    #kappa_train = np.around(trainAccuracy.kappa().getInfo(), decimals=4)
    #print('Kappa coefficient for training data (-1 to 1) =', kappa_train)

    #producers_train = trainAccuracy.producersAccuracy().getInfo()
    #table_writer(producers_train, 'Producers_acc_train_', export_name)

    #consumers_train = trainAccuracy.consumersAccuracy().getInfo()
    #table_writer(consumers_train, 'Consumers_acc_train_', export_name)

    #training_csv = os.path.join(fp_export_dir, 'Training_confusion_matrix'+export_name+'.csv')
    #with open(training_csv, "w", newline="") as f:
    #    writer = csv.writer(f)
    #    writer.writerows(resub_error_matrix)

    # Run test data assessment
    tested = test.classify(clf)
    test_accuracy = tested.errorMatrix('VALUE', 'classification')

    #test_error_matrix = test_accuracy.getInfo()
    #table_writer(test_error_matrix, 'Error_matrix_test_', export_name)

    test_overall_accuracy = np.around(test_accuracy.accuracy().getInfo(), decimals=4)
    print('Test data performance overall accuracy:', test_overall_accuracy)

    #kappa_test = np.around(test_accuracy.kappa().getInfo(), decimals=4)
    #print('Kappa coefficient for test data (-1 to 1) =', kappa_test)

    #producers_test = test_accuracy.producersAccuracy().getInfo()
    #table_writer(producers_test, 'Producers_acc_test_', export_name)

    #consumers_test = test_accuracy.consumersAccuracy().getInfo()
    #table_writer(consumers_test, 'Consumers_acc_test_', export_name)
    return test_overall_accuracy


def feature_importance_analysis(clf, export_name):
    dict_featImportance = clf.explain().getInfo()
    importance = dict_featImportance.get('importance')

    importance_csv = os.path.join(fp_export_dir, 'Feature_importance'+export_name+'.csv')
    with open(importance_csv, 'w') as f:
        writer = csv.writer(f)
        for row in importance.items():
            writer.writerow(row)


def spectral_stats(band_names_in, training_data, export_name):
    print('Generating spectral means...')
    band_names_list = band_names_in.getInfo()
    numBands = len(band_names_list)
    bandsWithClass = band_names_in.add('VALUE')
    classIndex = bandsWithClass.indexOf('VALUE')

    gcpStats = training_data.reduceColumns(**{
        'selectors': bandsWithClass,
        'reducer': ee.Reducer.mean().repeat(numBands).group(classIndex)}).getInfo()

    print('Exporting means...')
    spectral_csv = os.path.join(fp_export_dir, 'Class_spectral_mean'+export_name+'.csv')
    pd.DataFrame(gcpStats).to_csv(spectral_csv, index=False)
    print('Done!')


def line_plot(x, y, x_label):
    plt.plot(x, y, color='red', marker='x')
    plt.xlabel(x_label)
    plt.ylabel('Accuracy vs test (%)')
    plt.savefig(fp_export_dir+x_label+'.png')


def match_result_lengths(x, y):
    len_test = len(x)
    len_res = len(y)
    if len_test != len_res:
        diff = len_test - len_res
        x = x[:-diff]
    return x


def get_max_acc(test_vals, result):
    max_value = max(result)
    max_index = result.index(max_value)
    return test_vals[max_index]


def training_data_size_optimize(type_switch, bands_in):
    training_test_vals = [500, 750, 1000, 1250, 1500, 1750, 2000,
                          2250, 2500, 2750, 2800, 2850, 2900, 3000,
                          3250, 3500, 4000, 4250, 4500, 4750, 5000]
    result_trainsize_vals = []
    for i in training_test_vals:
        try:
            if type_switch == 'Complex':
                fp = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_points_" + str(
                    i) + "_v4.shp"
                train, test = load_sample_training_data(fp, bands_in)
                clf = apply_random_forest(train, 'RF_complex_train_'+str(i), 150, bands_in)
                val = accuracy_assessment(clf, test, 'RF_stackv2_train_'+str(i))
                result_trainsize_vals.append(val)
            elif type_switch == 'Simple':
                fp = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_points_" \
                                          + str(i) \
                                          + "_v4.shp"
                train, test = load_sample_training_data(fp, bands_in)
                clf = apply_random_forest(train, 'RF_complex_train_'+str(i), 150, bands_in)
                val = accuracy_assessment(clf, test, 'RF_stackv2_train_'+str(i))
                result_trainsize_vals.append(val)
            else:
                print('Specify classes type: Simple or Complex')
                break
        except:
            val = np.nan
            result_trainsize_vals.append(val)
            break

    training_test_vals = match_result_lengths(training_test_vals, result_trainsize_vals)
    max_acc = get_max_acc(training_test_vals, result_trainsize_vals)
    return training_test_vals, result_trainsize_vals, max_acc


def trees_size_optimize(type_switch, bands_in, train, test):
    trees_test_vals = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450]
    result_trees_vals = []
    for i in trees_test_vals:
        try:
            if type_switch == 'Complex':
                clf = apply_random_forest(train, 'RF_complex_trees_'+str(i), i, bands_in)
                val = accuracy_assessment(clf, test, 'RF_complex_trees'+str(i))
                result_trees_vals.append(val)
            elif type_switch == 'Simple':
                clf = apply_random_forest(train, 'RF_simple_trees_' + str(i), i, bands_in)
                val = accuracy_assessment(clf, test, 'RF_simple_trees' + str(i))
                result_trees_vals.append(val)
            else:
                print('Specify classes type: Simple or Complex')
                break
        except:
            val = np.nan
            result_trees_vals.append(val)
            break

    trees_test_vals = match_result_lengths(trees_test_vals, result_trees_vals)
    max_acc = get_max_acc(trees_test_vals, result_trees_vals)
    return trees_test_vals, result_trees_vals, max_acc


def optimised_classification_run(bands_in, train_size, tree_size, run_name, type_switch):
    fp = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/" + type_switch + "_points_" + str(train_size)\
         + "_v4.shp"
    train, test = load_sample_training_data(fp, bands_in)
    clf = apply_random_forest(train, run_name, tree_size, bands_in)
    acc_val = accuracy_assessment(clf, test, run_name)
    return acc_val, clf


"""
Global paths and variable settings
"""
# fp_train_ext = "/home/markdj/Dropbox/artio/polesia/val/Vegetation_extent_rough.shp"
fp_train_ext = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Project_area.shp"  # Area covered by the training data
fp_target_ext = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Classif_area.shp"  # Area to be classified
fp_export_dir = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Classified/"

# Set desired primary training data
fp_train_points_complex = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_points_2000_v4.shp"
fp_train_points_simple = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_points_2000_v4.shp"

label = 'VALUE'  # Name of the classes column in your training data
scale = 20  # Sets the output scale of the analysis

aoi = geemap.shp_to_ee(fp_train_ext)
date_list = [('2018-03-01', '2018-03-30'),
             ('2018-04-01', '2018-04-30'), ('2018-05-01', '2018-05-31'),
             ('2018-06-01', '2018-06-30'), ('2018-07-01', '2018-07-30'),
             ('2018-10-01', '2018-10-30')]

s2_params = {
    'CLOUD_FILTER': 60,  # int, max cloud coverage (%) permitted in a scene
    'CLD_PRB_THRESH': 40,  # int, 's2cloudless' 'probability' band value > thresh = cloud
    'NIR_DRK_THRESH': 0.15,  # float, if Band 8 (NIR) < NIR_DRK_THRESH = possible shadow
    'CLD_PRJ_DIST': 1,  # int, max distance [TODO: km or 100m?] from cloud edge for possible shadow
    'BUFFER': 50,  # int, distance (m) used to buffer cloud edges
    # 'S2BANDS': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    'S2BANDS': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']  # list of str, which S2 bands to return?
}


"""
Data load and prep
"""
# Training data satellite stack
stack = create_data_stack_v2(aoi, date_list, s2_params)
band_names = stack.bandNames()
trainingbands = band_names.getInfo()
print('Training bands are:', trainingbands)

# Load and sample the training data
train_complex, test_complex = load_sample_training_data(fp_train_points_complex, trainingbands)
train_simple, test_simple = load_sample_training_data(fp_train_points_simple, trainingbands)


"""
Class spectral analysis
"""
# Get spectral stats
spectral_stats(band_names, train_complex, '_indexed_spectra_gapfilled')


"""
Random forest classification
"""
# Test for training data size optimization
training_test, result_trainsize, opti_trainSize = training_data_size_optimize('Simple', trainingbands)
line_plot(training_test, result_trainsize, 'Training data sample size (reduced stack - simple classes)')

# Test for number of trees optimization
trees_test, result_trees, opti_treeSize = trees_size_optimize('Simple', trainingbands, train_simple, test_simple)
line_plot(trees_test, result_trees, 'Number of trees (reduced stack - simple classes)')

# Stand alone (optimised) classification
acc_val_simple, clf_simple = optimised_classification_run(trainingbands,
                                                          opti_trainSize, opti_treeSize,
                                                          'Optimum_simple_', 'Simple')

# Feature importance analysis
feature_importance_analysis(clf_simple, 'Optimum_simple_')


"""
SVM classification
"""
clf_svm_complex = apply_SVM(train_complex, 'SVM_complex')
accuracy_assessment(clf_svm_complex, test_complex, 'SVM_complex')

"""
Gradient tree boost classification
"""
clf_gtb_complex = apply_gradient_tree_boost(train_complex, 'GTB_complex')
accuracy_assessment(clf_gtb_complex, test_complex, 'GTB_complex')



