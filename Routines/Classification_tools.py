"""
This script contains all the classification functions to carry out landcover mapping in Polesia.

Uses random forest for simplicity, free data processing on GEE and ease of adaption to other targets.
"""
import numpy as np
from geemap import geemap
import ee
import sys
import os
import csv
sys.path.append("C:/Users/tpfdo/OneDrive/Documents/GitHub/Polesia-Landcover/Routines/")
from Utilities import table_writer

ee.Initialize()


def generate_RF_model(run_type, trees_num, train,  class_col_name, training_bands):
    print('Setting up random forest classifier for', run_type+'...')
    init_params = {"numberOfTrees": trees_num,
                   "variablesPerSplit": None,
                   "minLeafPopulation": 1,
                   "bagFraction": 0.5,
                   "maxNodes": None,
                   "seed": 0}
    clf = ee.Classifier.smileRandomForest(**init_params).train(train, class_col_name, training_bands)
    return clf


def apply_random_forest(export_name, training_bands, fp_target_ext, stack, scale, fp_export_dir, clf):
    """
    https://developers.google.com/earth-engine/apidocs/ee-classifier-smilerandomforest
    It downloads to the local export directory the applied RF classification.
    """
    # Carry out the Random Forest
    print('Using random forest to classify region', export_name+'...')
    target_area = geemap.shp_to_ee(fp_target_ext)
    #target_stack = stack.clip(target_area)
    classified = stack.select(training_bands).classify(clf)

    # Export results to local
    file_out = fp_export_dir+export_name+'.tif'
    roi = target_area.geometry()
    geemap.ee_export_image(classified, filename=file_out, scale=scale, file_per_band=False, region=roi)
    print('Random forest classification complete for', export_name+'!')
    return clf


def accuracy_assessment_basic(clf, test, export_name):
    print('Carrying out basic accuracy assessment for', export_name+'...')
    tested = test.classify(clf)
    test_accuracy = tested.errorMatrix('VALUE', 'classification')

    test_overall_accuracy = np.around(test_accuracy.accuracy().getInfo(), decimals=4)
    print('Test data performance overall accuracy:', test_overall_accuracy)
    return test_overall_accuracy


def accuracy_assessment_full(clf, test, export_name, fp_export_dir):
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
    test_accuracy = tested.errorMatrix('VALUE', 'classification')

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
