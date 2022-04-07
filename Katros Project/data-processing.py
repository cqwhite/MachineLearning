import os
import pandas as pd
import re

# Loop over files in directory
# Import with Pandas and keep in dictionary

dataframes = {}
for file in os.listdir("./ml_data"):
    norad_id = re.findall("\d+", file)[0]
    dataframes[norad_id] = pd.read_csv("./ml_data/" + file)

print(dataframes)
