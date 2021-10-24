"""
Functions to run the v2 of a classifier pipeline
"""
import sys
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


def load_sample_training_data(fp_train_points):
    """
    This function takes a filepath to a point shape file of your training data.
    It returns a training and test sampled dataset with a 70-30 split.
    It operates over the global raster data stack.
    """
    print('Loading training data:', fp_train_points+'...')
    training_points = geemap.shp_to_ee(fp_train_points)
    data = stack.select(trainingbands).sampleRegions(collection=training_points,
                                                     properties=[label],
                                                     scale=scale)

    # Split into train and test
    split = 0.7
    data = data.randomColumn(seed=0)
    train = data.filter(ee.Filter.lt('random',split))
    test = data.filter(ee.Filter.gte('random',split))
    print('Training data loaded and ready!')
    return train, test


def apply_random_forest(train, export_name):
    """
    https://developers.google.com/earth-engine/apidocs/ee-classifier-smilerandomforest

    This function takes a train dataset from load_sample_training_data, and a simple string name e.g. 'RF_simple'.
    It returns the prepped classification EE function.
    It downloads to the local export directory the applied RF classification.
    """
    print('Setting up random forest classifier for', export_name+'...')
    init_params = {"numberOfTrees": 100,
                   "variablesPerSplit": None,
                   "minLeafPopulation": 1,
                   "bagFraction": 0.5,
                   "maxNodes": None,
                   "seed": 0}

    clf = ee.Classifier.smileRandomForest(**init_params).train(train, label, trainingbands)

    # Carry out the Random Forest
    print('Using random forest to classify region...')
    target_area = geemap.shp_to_ee(fp_target_ext)
    target_stack = stack.clip(target_area)
    classified = target_stack.select(trainingbands).classify(clf)

    # Export results to local
    file_out = fp_export_dir+export_name+'.tif'
    roi = target_area.geometry()
    geemap.ee_export_image(classified, filename=file_out, scale=scale, file_per_band=False, region=roi)
    print('Random forest classification complete for', export_name+'!')
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
    print('Random forest classification complete for', export_name+'!')
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
    resub_error_matrix = trainAccuracy.getInfo()

    training_overall_accuracy = np.around(trainAccuracy.accuracy().getInfo(), decimals=4)
    print('Training data performance overall accuracy:', training_overall_accuracy)

    kappa_train = np.around(trainAccuracy.kappa().getInfo(), decimals=4)
    print('Kappa coefficient for training data (-1 to 1) =', kappa_train)

    #producers_train = trainAccuracy.producersAccuracy().getInfo()
    #table_writer(producers_train, 'Producers_acc_train_', export_name)

    #consumers_train = trainAccuracy.consumersAccuracy().getInfo()
    #table_writer(consumers_train, 'Consumers_acc_train_', export_name)

    training_csv = os.path.join(fp_export_dir, 'Training_confusion_matrix'+export_name+'.csv')
    with open(training_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(resub_error_matrix)

    # Run test data assessment
    tested = test.classify(clf)
    test_accuracy = tested.errorMatrix('VALUE', 'classification')

    test_error_matrix = test_accuracy.getInfo()
    table_writer(test_error_matrix, 'Error_matrix_test_', export_name)

    test_overall_accuracy = np.around(test_accuracy.accuracy().getInfo(), decimals=4)
    print('Test data performance overall accuracy:', test_overall_accuracy)

    kappa_test = np.around(test_accuracy.kappa().getInfo(), decimals=4)
    print('Kappa coefficient for test data (-1 to 1) =', kappa_test)

    #producers_test = test_accuracy.producersAccuracy().getInfo()
    #table_writer(producers_test, 'Producers_acc_test_', export_name)

    #consumers_test = test_accuracy.consumersAccuracy().getInfo()
    #table_writer(consumers_test, 'Consumers_acc_test_', export_name)


"""
Global paths and variable settings
"""
# fp_train_ext = "/home/markdj/Dropbox/artio/polesia/val/Vegetation_extent_rough.shp"
fp_train_ext = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Project_area.shp"  # Area covered by the training data
fp_target_ext = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Classif_area.shp"  # Area to be classified
fp_export_dir = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Classified/"

# Two different versions of training data
fp_train_points_simple = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Simple_points_2000.shp"
fp_train_points_complex = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Training_data/Complex_points_1750.shp"

label = 'VALUE'  # Name of the classes column in your training data
scale = 20  # Sets the output scale of the analysis

aoi = geemap.shp_to_ee(fp_train_ext)
date_list = [('2018-12-01', '2019-02-01'), ('2019-05-01', '2019-05-31')]

s2_params = {
    'CLOUD_FILTER': 60,  # int, max cloud coverage (%) permitted in a scene
    'CLD_PRB_THRESH': 40,  # int, 's2cloudless' 'probability' band value > thresh = cloud
    'NIR_DRK_THRESH': 0.15,  # float, if Band 8 (NIR) < NIR_DRK_THRESH = possible shadow
    'CLD_PRJ_DIST': 1,  # int, max distance [TODO: km or 100m?] from cloud edge for possible shadow
    'BUFFER': 100,  # int, distance (m) used to buffer cloud edges
    # 'S2BANDS': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'] #list of str, which S2 bands to return?
    'S2BANDS': ['B2', 'B3', 'B4', 'B6', 'B8', 'B11']  # list of str, which S2 bands to return?
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
train_simple, test_simple = load_sample_training_data(fp_train_points_simple)
train_complex, test_complex = load_sample_training_data(fp_train_points_complex)


"""
Random forest classification
"""
clf_rf_simple = apply_random_forest(train_simple, 'RF_simple')
accuracy_assessment(clf_rf_simple, test_simple, 'RF_simple')

clf_rf_complex = apply_random_forest(train_complex, 'RF_complex')
accuracy_assessment(clf_rf_complex, test_complex, 'RF_complex')

"""
CART classification
"""
clf_cart_simple = apply_CART(train_simple, 'CART_simple')
accuracy_assessment(clf_cart_simple, test_simple, 'CART_simple')

clf_cart_complex = apply_CART(train_complex, 'CART_complex')
accuracy_assessment(clf_cart_complex, test_complex, 'CART_complex')

"""
SVM classification
"""
clf_svm_simple = apply_SVM(train_simple, 'SVM_simple')
accuracy_assessment(clf_svm_simple, test_simple, 'SVM_simple')

clf_svm_complex = apply_SVM(train_complex, 'SVM_complex')
accuracy_assessment(clf_svm_complex, test_complex, 'SVM_complex')

"""
Gradient tree boost classification
"""
clf_gtb_simple = apply_gradient_tree_boost(train_simple, 'GTB_simple')
accuracy_assessment(clf_gtb_simple, test_simple, 'GTB_simple')

clf_gtb_complex = apply_gradient_tree_boost(train_complex, 'GTB_complex')
accuracy_assessment(clf_gtb_complex, test_complex, 'GTB_complex')



