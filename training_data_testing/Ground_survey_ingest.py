"""
Quick script to parse in ground survey data and return a compound table
"""

import pandas as pd
from pathlib import Path

# Top directory in which all tables are located
dir_path = "D:/tpfdo/Documents/Artio_drive/Projects/Polesia/Field_surveys/"

# Set the names of the columns in the output table
df_out = pd.DataFrame(columns=['Lat', 'Lon', 'Type', 'Square_ID'])

# Get the columns and rows in which you are interested
# Skips the first row in order to get data needed
for path in Path(dir_path).rglob('*.xlsx'):
    print('Reading:', path.name)
    df_in = pd.read_excel(path, skiprows=[0])
    df_out = df_out.append({'Lat': df_in.iloc[0, [4]][0],
                            'Lon': df_in.iloc[0, [5]][0],
                            'Type': df_in.iloc[0, [7]][0],
                            'Square_ID': df_in.iloc[0, [8]].iloc[0]},
                           ignore_index=True)

# Exports to top directory
df_out.to_csv(dir_path+'Ground_surveys.csv')
print('Record exported')
