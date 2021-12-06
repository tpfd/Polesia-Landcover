"""
This script contains any general functions that do not belong in any other category/have uses in all areas
of the mapping effort.
"""
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd


def table_writer(table, table_name, export_name, fp_export_dir):
    out_csv = os.path.join(fp_export_dir, table_name + export_name + '.csv')
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)
    print(table_name+export_name, 'written out.')


def line_plot(x, y, x_label, fp_export_dir):
    plt.plot(x, y, color='red', marker='x')
    plt.xlabel(x_label)
    plt.ylabel('Accuracy vs test (%)')
    plt.savefig(fp_export_dir+x_label+'.png')
    plt.close('all')


def get_max_acc(test_vals, result):
    max_value = max(result)
    max_index = result.index(max_value)
    return test_vals[max_index], test_vals[max_index]


def load_presets(fp_settings_txt):
    preset_table = pd.read_csv(fp_settings_txt)
    try:
        preset_table.index = preset_table.Variable
    except:
        pass
    fp_train_simple_points = preset_table['Value']['fp_train_simple_points']
    fp_train_complex_points = preset_table['Value']['fp_train_complex_points']
    trees_complex = preset_table['Value']['trees_complex']
    training_complex = preset_table['Value']['training_complex']
    trees_simple = preset_table['Value']['trees_simple']
    training_simple = preset_table['Value']['training_simple']
    return fp_train_simple_points, fp_train_complex_points, trees_complex, training_complex, trees_simple, \
           training_simple


def generate_empty_preset_table():
    preset_table = pd.DataFrame(columns=['Value'])
    variables = ['fp_train_complex_points', 'fp_train_simple_points',
                 'trees_complex', 'training_complex',
                 'trees_simple', 'training_simple']
    for i in variables:
        preset_table.loc[i] = [None]
    return preset_table


def match_result_lengths(x, y):
    len_test = len(x)
    len_res = len(y)
    if len_test != len_res:
        diff = len_test - len_res
        x = x[:-diff]
    return x


def get_list_of_files(directory, suffix):
    list_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                list_files.append(os.path.join(root, file))
    return list_files