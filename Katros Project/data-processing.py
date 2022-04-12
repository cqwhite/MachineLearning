# David Chalifoux, Connor White, Quinn Partain, Micah Odell
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
    # print(doa_df[doa_df["primary_rx_time"] == unique_rx_times[0]])
    # print(unique_rx_times[0])
    # print(doa_df)
    count = 0
    new_df = pd.DataFrame(columns=['sat_id','sat_name','norad_id','primary_rx_time','primary_ant_id','secondary_ant_id','tdoa','fdoa','maneuver','175_177_tdoa','175_177_fdoa','175_176_tdoa','175_176_fdoa','176_177_tdoa','176_177_fdoa'])
    for times in unique_rx_times:
        temp = doa_df[doa_df["primary_rx_time"] == times]

        for x in range(len(temp)):
            firstAnt = temp.iloc[x]['primary_ant_id']
            secondAnt = temp.iloc[x]['secondary_ant_id']
            name = str(firstAnt)+"_"+str(secondAnt)+"_tdoa"
            temp[name] = temp.iloc[x]['tdoa']
            name = str(firstAnt)+"_"+str(secondAnt)+"_fdoa"
            temp[name] = temp.iloc[x]['fdoa']
            new_df.loc[len(new_df.index)] = temp.iloc[x]

        count += 1
        if count % 100 == 0:
            print("running....", count)

    print(new_df)

  




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
doa_df = pd.read_csv("./ml_data/42709_doa.csv", parse_dates=["primary_rx_time"]).assign(
    maneuver=0
)
truth_df = pd.read_csv("42709_maneuvers_truthv2.csv")

# Combine rows
combine_rows(doa_df)

# # Add the truth data to the DOA data
# match_df = match_truth(doa_df, truth_df)

# # Standardize features by removing the mean and scaling to unit variance.
# scaler = StandardScaler()
# match_df[["tdoa_scaled", "fdoa_scaled"]] = scaler.fit_transform(
#     match_df[["tdoa", "fdoa"]]
# )
# print(match_df)

# # Split - 60% train, 20% test, 20% validate
# train, test, validate = np.split(
#     match_df.sample(frac=1, random_state=9),
#     [int(0.6 * len(match_df)), int(0.8 * len(match_df))],
# )

# # Save to CSV
# train.head(100).to_csv("./output.txt")
# print("Example data output to ./output.txt")
