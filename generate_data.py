"""
generate_data.py
-----------------
Validates that the 5 required WorkSafe AI CSVs exist in CsvData/ and
match the expected schema. Does NOT create or modify any data.

Data flow: CsvData/*.csv (checked here) -> load_to_sql.py -> worksafe.db
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "CsvData")

EXPECTED_SCHEMA = {
    "workers.csv": [
        "Worker_ID", "Worker_Name", "Worker_Type", "Skill_Level",
        "Experience_Years", "Team_ID", "Project_ID", "Join_Date",
    ],
    "worker_activity.csv": [
        "Record_ID", "Worker_ID", "Project_ID", "Activity_ID", "Date",
        "Shift", "Hours_Worked", "Overtime_Hours", "Break_Minutes",
        "Workload_Level", "Output_Units", "Expected_Output",
        "Productivity_Index", "Consecutive_Days",
    ],
    "activities.csv": [
        "Activity_ID", "Activity_Name", "Activity_Type", "Difficulty_Level",
    ],
    "attendance.csv": [
        "Attendance_ID", "Worker_ID", "Date", "Present_Flag",
        "Late_Minutes", "Leave_Flag",
    ],
    "projects.csv": [
        "Project_ID", "Project_Name", "Location", "Project_Type",
        "Start_Date", "Budget", "End_Date",
    ],
}


def validate(filename, expected_columns):
    path = os.path.join(CSV_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing '{path}'. Put your CSVs in the CsvData/ folder.")

    df = pd.read_csv(path)
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing expected columns: {missing}")

    print(f"  {filename:<22} rows={len(df):<7} cols={len(df.columns)}  OK")


def main():
    print(f"Validating dataset in '{CSV_DIR}' ...\n")
    for filename, columns in EXPECTED_SCHEMA.items():
        validate(filename, columns)
    print("\nAll 5 files present and schema-valid.")


if __name__ == "__main__":
    main()