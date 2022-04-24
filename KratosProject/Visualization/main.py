"""
Visualize the data set using plotly express

David Chalifoux, Quinn Partain, Connor White, Micah Odell
"""
import plotly.express as px
import pandas as pd

train_df = pd.read_csv("./42709_train.csv").dropna()

fig = px.scatter_matrix(
    train_df,
    dimensions=[
        "175_177_tdoa",
        "175_177_fdoa",
        "175_176_tdoa",
        "175_176_fdoa",
        "176_177_tdoa",
        "176_177_fdoa",
    ],
    color="maneuver",
    title="Satelite 42709",
)
fig.show()
