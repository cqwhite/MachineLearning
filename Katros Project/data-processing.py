import os
import pandas as pd
import re
import dateutil.parser

doa_df = {}
maneuvers_df = {}

# Loop over files in directory
# Import with Pandas and keep in dictionary
# Processes all files in ./ml_data
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
    doa_df = pd.read_csv(doa_file, parse_dates=["primary_rx_time"]).assign(
        maneuver=False
    )
    truth_df = pd.read_csv(truth_file)

    # Loop over truth data and match with maneuver data
    for row in truth_df.itertuples():
        print(row[4])
    mask = (doa_df["primary_rx_time"] >= "2021-09-12T04:18:54.000Z") & (
        doa_df["primary_rx_time"] <= "2021-09-12T05:57:02.000Z"
    )

    doa_df.loc[mask, "maneuver"] = True
    # doa_in_range = doa_df[
    #     doa_df["primary_rx_time"] > dateutil.parser.isoparse("2021-09-12T04:18:54.000Z")
    # ]
    # doa_in_range = doa_df[
    #     doa_df["primary_rx_time"] < dateutil.parser.isoparse("2021-09-12T05:57:02.000Z")
    # ]
    print(doa_df[doa_df["maneuver"] == True])


# for key, value in maneuvers_df.items():
#     print(key)
# print(maneuvers_df)

match_truth("./ml_data/42709_doa.csv", "42709_maneuvers_truthv2.csv")
