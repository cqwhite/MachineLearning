"""
Example of using kNN for outlier detection

Tested different combinations of TDOA and FDOA data.
So far, 176_177_tdoa with 176_177_fdoa seems to be the most performant.

RESULTS:
    On Training Data:
    KNN ROC:0.5731, precision @ rank n:0.2151

    On Test Data:
    KNN ROC:0.5821, precision @ rank n:0.25

David Chalifoux, Connor White, Quinn Partain, Micah Odell
"""
from pyod.models.knn import KNN
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
import pandas as pd

if __name__ == "__main__":
    train_df = pd.read_csv("./42709_train.csv").dropna()
    test_df = pd.read_csv("./42709_test.csv").dropna()

    X_train = train_df[
        [
            # "175_177_tdoa",
            # "175_177_fdoa",
            # "175_176_tdoa",
            # "175_176_fdoa",
            "176_177_tdoa",
            "176_177_fdoa",
        ]
    ]
    X_test = test_df[
        [
            # "175_177_tdoa",
            # "175_177_fdoa",
            # "175_176_tdoa",
            # "175_176_fdoa",
            "176_177_tdoa",
            "176_177_fdoa",
        ]
    ]
    y_train = train_df["maneuver"]
    y_test = test_df["maneuver"]

    # train kNN detector
    CLF_NAME = "KNN"
    CONTAMINATION = len(train_df.query("maneuver == 1")) / len(train_df)
    clf = KNN(
        contamination=CONTAMINATION,
        algorithm="ball_tree",
        n_neighbors=12,
        method="largest",
    )
    clf.fit(X_train)

    # print parameters
    print("Algorithm:", clf.algorithm)
    print("Contamination:", clf.contamination)
    print("N-Neigbors:", clf.n_neighbors)
    print("Method:", clf.method)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(CLF_NAME, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(CLF_NAME, y_test, y_test_scores)

    # visualize the results
    # visualize(
    #     CLF_NAME,
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     y_train_pred,
    #     y_test_pred,
    #     show_figure=True,
    #     save_figure=True,
    # )
