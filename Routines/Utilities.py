"""
This script contains any general functions that do not belong in any other category/have uses in all areas
of the mapping effort.
"""

def table_writer(table, table_name, export_name):
    out_csv = os.path.join(fp_export_dir, table_name + export_name + '.csv')
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)
    print(table_name+export_name, 'written out.')