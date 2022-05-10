"""
Uses SKlearn's PCA impelementation to reduce the dimensions of the data to 3.
Plots the data in 3D.
"""
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA


train_df = pd.read_csv("./42709_train.csv").dropna()
principal_df = train_df[
    [
        "175_177_tdoa",
        "175_177_fdoa",
        "175_176_tdoa",
        "175_176_fdoa",
        "176_177_tdoa",
        "176_177_fdoa",
    ]
]
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(principal_df)
principal_df = pd.DataFrame(
    data=principalComponents,
    columns=["Principal_Component_1", "Principal_Component_2", "Principal_Component_3"],
)
principal_df["maneuver"] = train_df["maneuver"]


fig = px.scatter_3d(
    principal_df,
    x="Principal_Component_1",
    y="Principal_Component_2",
    z="Principal_Component_3",
    color="maneuver",
)
fig.show()
