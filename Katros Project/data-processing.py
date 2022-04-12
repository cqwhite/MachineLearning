# David Chalifoux, Connor White, Quinn Partain, Micah Odell
import os
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
import numpy as np

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


def match_truth(doa_file, truth_file):
    # Read doa and truth files
    doa_df = pd.read_csv(doa_file, parse_dates=["primary_rx_time"]).assign(
        maneuver=False
    )
    truth_df = pd.read_csv(truth_file)

    # Loop over truth data and match with maneuver data
    for row in truth_df.itertuples():
        mask = (doa_df["primary_rx_time"] >= row[4]) & (
            doa_df["primary_rx_time"] <= row[5]
        )

        doa_df.loc[mask, "maneuver"] = True

    # Return dataframe with combined maneuver data
    return doa_df


# Add the truth data to the DOA data
match_df = match_truth("./ml_data/42709_doa.csv", "42709_maneuvers_truthv2.csv")

# Standardize features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
match_df[["tdoa_scaled", "fdoa_scaled"]] = scaler.fit_transform(
    match_df[["tdoa", "fdoa"]]
)
print(match_df)

# Split - 60% train, 20% test, 20% validate
train, test, validate = np.split(
    match_df.sample(frac=1, random_state=9),
    [int(0.6 * len(match_df)), int(0.8 * len(match_df))],
)

# Save to CSV
train.head(100).to_csv("./output.txt")
print("Example data output to ./output.txt")
