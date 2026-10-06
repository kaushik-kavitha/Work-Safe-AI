"""
03_Fatigue_Model.py
--------------------
Trains a Random Forest classifier that predicts each worker's fatigue
risk level (Low / Medium / High / Critical) from the 5 engineered features.

Your dataset has no ready-made "fatigue" column, so this script first
builds a Fatigue Score (0-100) from the 5 features using a transparent
weighted formula, converts it into 4 risk levels, and then trains the
Random Forest to predict those levels.

Data flow: notebooks/worker_features.csv -> this script -> notebooks/fatigue_model.pkl
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(SCRIPT_DIR, "worker_features.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "fatigue_model.pkl")

FEATURE_COLS = [
    "Overtime_Ratio",
    "Break_Adequacy",
    "Consecutive_Days",
    "Workload_Score",
    "Productivity_Change",
]
RISK_ORDER = ["Low", "Medium", "High", "Critical"]


def minmax(series):
    """Scale a column to 0-1 so features with different units can be combined."""
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())


def compute_fatigue_score(df):
    """
    Fatigue Score (0-100). Higher = more fatigued.
      + more overtime, more consecutive days, heavier workload  -> higher
      + fewer/shorter breaks (low Break_Adequacy)               -> higher
      + productivity declining (low Productivity_Change)        -> higher
    """
    score = (
        0.25 * minmax(df["Overtime_Ratio"])
        + 0.15 * (1 - minmax(df["Break_Adequacy"]))
        + 0.25 * minmax(df["Consecutive_Days"])
        + 0.15 * minmax(df["Workload_Score"])
        + 0.20 * (1 - minmax(df["Productivity_Change"]))
    )
    return (score * 100).round(1)


def assign_risk_level(scores):
    """Bottom 40% Low, next 30% Medium, next 20% High, top 10% Critical."""
    pct = scores.rank(pct=True, method="first")
    return pd.cut(pct, bins=[0, 0.4, 0.7, 0.9, 1.0], labels=RISK_ORDER, include_lowest=True)


def main():
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} workers from worker_features.csv")

    df["Fatigue_Score"] = compute_fatigue_score(df)
    df["Risk_Level"] = assign_risk_level(df["Fatigue_Score"])

    print("\n=== Risk level distribution (labels the model learns from) ===")
    print(df["Risk_Level"].value_counts().reindex(RISK_ORDER))

    X = df[FEATURE_COLS]
    y = df["Risk_Level"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Evaluation on held-out test set ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=RISK_ORDER, zero_division=0))
    print("Confusion matrix (rows = actual, cols = predicted, order: Low, Medium, High, Critical):")
    print(confusion_matrix(y_test, y_pred, labels=RISK_ORDER))

    print("\n=== Feature importance ===")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importance.round(3))

    # Refit on all workers for the final saved model
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X, y)

    joblib.dump(
        {"model": final_model, "features": FEATURE_COLS, "risk_order": RISK_ORDER},
        MODEL_PATH,
    )
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
