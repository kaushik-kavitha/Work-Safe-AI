"""
01_EDA.py
----------
Exploratory Data Analysis on WorkSafe AI data. Reads from worksafe.db,
prints summary stats, and shows charts in a window (nothing is saved).

Data flow: worksafe.db -> this script -> printed stats + on-screen charts
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "worksafe.db")


def load_data(conn):
    activity = pd.read_sql_query("SELECT * FROM FactWorkerActivity;", conn)
    attendance = pd.read_sql_query("SELECT * FROM FactAttendance;", conn)
    workers = pd.read_sql_query("SELECT * FROM DimWorker;", conn)
    projects = pd.read_sql_query("SELECT * FROM DimProject;", conn)
    return activity, attendance, workers, projects


def basic_stats(activity, attendance, workers):
    print("=== Shape ===")
    print(f"  FactWorkerActivity: {activity.shape}")
    print(f"  FactAttendance:     {attendance.shape}")
    print(f"  DimWorker:          {workers.shape}")

    print("\n=== Productivity_Index summary ===")
    print(activity["Productivity_Index"].describe())

    print("\n=== Overtime_Hours summary ===")
    print(activity["Overtime_Hours"].describe())

    print("\n=== Consecutive_Days summary ===")
    print(activity["Consecutive_Days"].describe())

    print("\n=== Missing values per column (FactWorkerActivity) ===")
    print(activity.isnull().sum())

    print("\n=== Worker_Type distribution ===")
    print(workers["Worker_Type"].value_counts())

    print("\n=== Skill_Level distribution ===")
    print(workers["Skill_Level"].value_counts())

    print("\n=== Attendance rate ===")
    print(f"  Present rate: {attendance['Present_Flag'].mean():.2%}")
    print(f"  Leave rate:   {attendance['Leave_Flag'].mean():.2%}")


def correlation_check(activity):
    print("\n=== Correlation matrix (numeric activity fields) ===")
    numeric_cols = ["Hours_Worked", "Overtime_Hours", "Break_Minutes",
                    "Output_Units", "Expected_Output", "Productivity_Index",
                    "Consecutive_Days"]
    corr = activity[numeric_cols].corr()
    print(corr.round(2))
    return corr


def show_charts(activity, workers, corr):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    sns.histplot(activity["Productivity_Index"], bins=30, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Productivity Index Distribution")

    sample = activity.sample(min(1000, len(activity)))
    sns.scatterplot(data=sample, x="Overtime_Hours", y="Productivity_Index",
                    alpha=0.4, ax=axes[0, 1])
    axes[0, 1].set_title("Overtime Hours vs Productivity Index")

    by_days = activity.groupby("Consecutive_Days")["Productivity_Index"].mean().reset_index()
    sns.lineplot(data=by_days, x="Consecutive_Days", y="Productivity_Index", ax=axes[0, 2])
    axes[0, 2].set_title("Avg Productivity by Consecutive Days")

    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=axes[1, 0])
    axes[1, 0].set_title("Correlation Heatmap")

    sns.countplot(data=workers, x="Worker_Type", ax=axes[1, 1])
    axes[1, 1].set_title("Worker Type Distribution")

    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    conn = sqlite3.connect(DB_PATH)
    activity, attendance, workers, projects = load_data(conn)
    conn.close()

    basic_stats(activity, attendance, workers)
    corr = correlation_check(activity)
    show_charts(activity, workers, corr)

    print("\nEDA complete.")


if __name__ == "__main__":
    main()