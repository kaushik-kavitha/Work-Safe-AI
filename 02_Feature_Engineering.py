"""
02_Feature_Engineering.py
---------------------------
Builds the per-worker feature table used by all three ML models
(Random Forest, K-Means, Isolation Forest): Overtime Ratio, Break
Adequacy, Consecutive Days, Workload Score, Productivity Change.

Data flow: worksafe.db -> this script -> notebooks/worker_features.csv
"""

import os
import sqlite3
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "worksafe.db")
OUT_PATH = os.path.join(SCRIPT_DIR, "worker_features.csv")

WORKLOAD_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
EXPECTED_BREAK_MINUTES = 45


def load_data(conn):
    activity = pd.read_sql_query("SELECT * FROM FactWorkerActivity;", conn)
    attendance = pd.read_sql_query("SELECT * FROM FactAttendance;", conn)
    workers = pd.read_sql_query("SELECT * FROM DimWorker;", conn)
    return activity, attendance, workers


def build_features(activity, attendance, workers):
    activity = activity.copy()
    activity["Date_parsed"] = pd.to_datetime(activity["Date"], format="%d-%m-%Y")
    activity["Workload_Score_Raw"] = activity["Workload_Level"].map(WORKLOAD_MAP)

    rows = []
    for worker_id, group in activity.groupby("Worker_ID"):
        group = group.sort_values("Date_parsed")

        total_hours = group["Hours_Worked"].sum()
        total_overtime = group["Overtime_Hours"].sum()
        overtime_ratio = round(total_overtime / total_hours, 3) if total_hours > 0 else 0.0

        break_adequacy = round(group["Break_Minutes"].mean() / EXPECTED_BREAK_MINUTES, 3)

        consecutive_days = int(group["Consecutive_Days"].max())

        workload_score = round(group["Workload_Score_Raw"].mean(), 3)

        # productivity change: last 15 days vs first 15 days
        n = len(group)
        split = max(1, min(15, n // 2))
        early_avg = group["Productivity_Index"].iloc[:split].mean()
        recent_avg = group["Productivity_Index"].iloc[-split:].mean()
        productivity_change = round(recent_avg - early_avg, 3)

        rows.append({
            "Worker_ID": worker_id,
            "Overtime_Ratio": overtime_ratio,
            "Break_Adequacy": break_adequacy,
            "Consecutive_Days": consecutive_days,
            "Workload_Score": workload_score,
            "Productivity_Change": productivity_change,
            "Avg_Productivity": round(group["Productivity_Index"].mean(), 3),
            "Avg_Overtime_Hours": round(group["Overtime_Hours"].mean(), 3),
        })

    features_df = pd.DataFrame(rows)

    # attach attendance rate per worker
    att_rate = attendance.groupby("Worker_ID")["Present_Flag"].mean().reset_index()
    att_rate.columns = ["Worker_ID", "Attendance_Rate"]
    att_rate["Attendance_Rate"] = att_rate["Attendance_Rate"].round(3)
    features_df = features_df.merge(att_rate, on="Worker_ID", how="left")

    # attach worker metadata for readability (not used as model input directly)
    features_df = features_df.merge(
        workers[["Worker_ID", "Worker_Name", "Worker_Type", "Skill_Level", "Team_ID", "Project_ID"]],
        on="Worker_ID", how="left"
    )

    return features_df


def main():
    conn = sqlite3.connect(DB_PATH)
    activity, attendance, workers = load_data(conn)
    conn.close()

    print(f"Loaded {len(activity)} activity rows, {len(attendance)} attendance rows, {len(workers)} workers.")

    features_df = build_features(activity, attendance, workers)
    features_df.to_csv(OUT_PATH, index=False)

    print(f"\nBuilt feature table: {features_df.shape[0]} workers x {features_df.shape[1]} columns")
    print(f"Saved to {OUT_PATH}")
    print("\n=== Sample ===")
    print(features_df.head())
    print("\n=== Feature summary stats ===")
    print(features_df[["Overtime_Ratio", "Break_Adequacy", "Consecutive_Days",
                        "Workload_Score", "Productivity_Change", "Attendance_Rate"]].describe())


if __name__ == "__main__":
    main()