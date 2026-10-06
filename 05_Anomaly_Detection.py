"""
05_Anomaly_Detection.py
------------------------
Trains an Isolation Forest that flags workers whose productivity pattern
is unusual compared with everyone else (for example, very low output
combined with heavy overtime, or a sharp productivity drop).

Like K-Means, this is unsupervised: there are no "anomaly" labels. The
model isolates workers who are easy to separate from the crowd.

Data flow: notebooks/worker_features.csv -> this script -> notebooks/anomaly_model.pkl
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(SCRIPT_DIR, "worker_features.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "anomaly_model.pkl")

FEATURE_COLS = [
    "Avg_Productivity",
    "Productivity_Change",
    "Overtime_Ratio",
    "Attendance_Rate",
]

# Expected share of unusual workers. 0.05 means roughly 5% (about 8 of 150).
CONTAMINATION = 0.05


def main():
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} workers from worker_features.csv")

    X = df[FEATURE_COLS]

    model = IsolationForest(
        n_estimators=100, contamination=CONTAMINATION, random_state=42
    )
    model.fit(X)

    # predict() returns -1 for anomalies and 1 for normal workers
    df["Is_Anomaly"] = (model.predict(X) == -1).astype(int)
    # decision_function(): lower (more negative) = more anomalous
    df["Anomaly_Score"] = model.decision_function(X).round(3)

    # Which feature is furthest from normal for each worker (for explanations)
    z = (X - X.mean()) / X.std()
    top_feature = z.abs().idxmax(axis=1)
    df["Main_Deviation"] = [
        f"{feat} ({'high' if z.loc[i, feat] > 0 else 'low'})"
        for i, feat in top_feature.items()
    ]

    n_anomalies = int(df["Is_Anomaly"].sum())
    print(f"\nFlagged {n_anomalies} of {len(df)} workers as anomalies "
          f"({n_anomalies / len(df):.1%}).")

    print("\n=== Average feature values, all workers (for comparison) ===")
    print(X.mean().round(3).to_string())

    print("\n=== Flagged workers (most anomalous first) ===")
    flagged = df[df["Is_Anomaly"] == 1].sort_values("Anomaly_Score")
    cols = ["Worker_ID", "Worker_Name", "Team_ID"] + FEATURE_COLS + ["Anomaly_Score", "Main_Deviation"]
    print(flagged[cols].to_string(index=False))

    joblib.dump({"model": model, "features": FEATURE_COLS}, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
