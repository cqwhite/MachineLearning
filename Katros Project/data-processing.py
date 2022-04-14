# David Chalifoux, Connor White, Quinn Partain, Micah Odell
from email.utils import parsedate_to_datetime
from math import comb
import os
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

# Loop over files in directory
# Import with Pandas and keep in dictionary
# Processes all files in ./ml_data
# Note: No longer being used
def read_ml_data():
    doa_df = {}
    maneuvers_df = {}
    for file in os.listdir("./ml_data"):
        norad_id = re.findall("\d+", file)[0]
        data_type = re.findall("doa|maneuvers", file)[0]
        if data_type == "doa":
            doa_df[norad_id] = pd.read_csv("./ml_data/" + file)
        elif data_type == "maneuvers":
            maneuvers_df[norad_id] = pd.read_csv("./ml_data/" + file)


def combine_rows(doa_df):
    unique_rx_times = doa_df.primary_rx_time.unique()

    count = 0
    rows = []
    for time in unique_rx_times:
        row = {"primary_rx_time": time}
        matching_rows = doa_df[doa_df["primary_rx_time"] == time]

        for x in range(len(matching_rows)):
            primary_ant = matching_rows.iloc[x]["primary_ant_id"]
            secondary_ant = matching_rows.iloc[x]["secondary_ant_id"]
            tdoa_column_name = str(primary_ant) + "_" + str(secondary_ant) + "_tdoa"
            fdoa_column_name = str(primary_ant) + "_" + str(secondary_ant) + "_fdoa"
            tdoa_value = matching_rows.iloc[x]["tdoa"]
            fdoa_value = matching_rows.iloc[x]["tdoa"]
            row[tdoa_column_name] = tdoa_value
            row[fdoa_column_name] = fdoa_value
            print(count, end="\r")
            count += 1
        rows.append(row)
    return pd.DataFrame(rows)


def match_truth(doa_df, truth_df):
    # Loop over truth data and match with maneuver data
    for row in truth_df.itertuples():
        mask = (doa_df["primary_rx_time"] >= row[4]) & (
            doa_df["primary_rx_time"] <= row[5]
        )

        doa_df.loc[mask, "maneuver"] = 1

    # Return dataframe with combined maneuver data
    return doa_df


# Read doa and truth files
doa_df = pd.read_csv("./ml_data/42709_doa.csv", parse_dates=["primary_rx_time"])
truth_df = pd.read_csv("42709_maneuvers_truthv2.csv")

# Combine rows
combined_df = combine_rows(doa_df)

# Add maneuver column
combined_df = combined_df.assign(maneuver=0)

# Add the truth data to the DOA data
match_df = match_truth(combined_df, truth_df)

# Standardize features by removing the mean and scaling to unit variance.
columns_to_scale = [
    "175_177_tdoa",
    "175_177_fdoa",
    "175_176_tdoa",
    "175_176_fdoa",
    "176_177_tdoa",
    "176_177_fdoa",
]
columns_to_add = []
for column in columns_to_scale:
    columns_to_add.append(column + "_scaled")
scaler = StandardScaler()
match_df[columns_to_add] = scaler.fit_transform(match_df[columns_to_scale])

# Split - 60% train, 20% test, 20% validate
train, test, validate = np.split(
    match_df.sample(frac=1, random_state=9),
    [int(0.6 * len(match_df)), int(0.8 * len(match_df))],
)

# Save to CSV
train.head(100).to_csv("./output.txt")
print("Example data output to ./output.txt")
