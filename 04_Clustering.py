"""
04_Clustering.py
-----------------
Trains a K-Means model that groups workers into performance segments
based on how they actually work (productivity, attendance, overtime,
workload). Unlike the Random Forest, there are no labels here: K-Means
finds the groups on its own from the data.

Data flow: notebooks/worker_features.csv -> this script -> notebooks/cluster_model.pkl
"""

import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(SCRIPT_DIR, "worker_features.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "cluster_model.pkl")

FEATURE_COLS = [
    "Avg_Productivity",
    "Productivity_Change",
    "Attendance_Rate",
    "Overtime_Ratio",
    "Workload_Score",
]

N_CLUSTERS = 3
# Cluster numbers from K-Means are arbitrary (0, 1, 2), so we name them
# after training by ranking clusters on average productivity.
SEGMENT_NAMES = ["High Performer", "Steady Performer", "Needs Support"]


def main():
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} workers from worker_features.csv")

    # K-Means uses distances, so put all features on the same scale first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])

    print("\n=== Silhouette score by number of clusters (higher = better separated) ===")
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        print(f"  k={k}: {silhouette_score(X_scaled, labels):.3f}")
    print(f"\nUsing k={N_CLUSTERS} (High / Steady / Needs Support).")

    model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(X_scaled)

    profile = df.groupby("Cluster")[FEATURE_COLS].mean().round(3)
    profile["Workers"] = df.groupby("Cluster").size()

    order = profile["Avg_Productivity"].sort_values(ascending=False).index.tolist()
    cluster_names = {int(c): name for c, name in zip(order, SEGMENT_NAMES)}
    profile["Segment"] = profile.index.map(cluster_names)

    print("\n=== Cluster profiles (average feature values per cluster) ===")
    print(profile.sort_values("Avg_Productivity", ascending=False).to_string())

    print("\n=== Sample workers per cluster ===")
    df["Segment"] = df["Cluster"].map(cluster_names)
    for name in SEGMENT_NAMES:
        sample = df[df["Segment"] == name][["Worker_ID", "Worker_Name", "Team_ID", "Avg_Productivity"]].head(3)
        print(f"\n{name}:")
        print(sample.to_string(index=False))

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURE_COLS,
            "cluster_names": cluster_names,
        },
        MODEL_PATH,
    )
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
