import os
from geemap.conversion import *
work_dir = os.path.join(os.path.expanduser('~'), 'geemap')

js_dir = "C:/Users/tpfdo/OneDrive/Documents/GitHub/Polesia-Landcover/classification_testing/Java_scripts"
js_to_python_dir(in_dir=js_dir, out_dir=js_dir, use_qgis=True)
