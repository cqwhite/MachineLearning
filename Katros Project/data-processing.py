import os
import pandas as pd
import re

# df = pd.read_csv (r'Path where the CSV file is stored\File name.csv')
dataframes = {}
for file in os.listdir("./ml_data"):
    norad_id = re.findall("\d+", file)[0]
    dataframes[norad_id] = pd.read_csv("./ml_data/" + file)

print(dataframes)
