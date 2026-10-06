"""
load_to_sql.py
----------------
Creates worksafe.db (SQLite) using SQL/create_tables.sql, then loads
the 5 CSVs from CsvData/ into their matching tables.

Data flow: CsvData/*.csv + SQL/create_tables.sql -> worksafe.db
"""

import sqlite3
import pandas as pd

DB_PATH = "worksafe.db"
SCHEMA_PATH = "SQL/create_tables.sql"
CSV_DIR = "CsvData"

# maps: table name -> (csv filename, ordered column list matching the table)
TABLE_MAP = {
    "DimWorker": "workers.csv",
    "DimProject": "projects.csv",
    "DimActivity": "activities.csv",
    "FactWorkerActivity": "worker_activity.csv",
    "FactAttendance": "attendance.csv",
}


def create_schema(conn):
    print(f"Creating schema from {SCHEMA_PATH} ...")
    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    print("  Schema created (6 tables: DimWorker, DimProject, DimActivity, "
          "FactWorkerActivity, FactAttendance, FactFatigue).")


def load_table(conn, table_name, csv_filename):
    csv_path = f"{CSV_DIR}/{csv_filename}"
    df = pd.read_csv(csv_path)

    # clear existing rows first, so re-running this script doesn't duplicate data
    conn.execute(f"DELETE FROM {table_name};")

    df.to_sql(table_name, conn, if_exists="append", index=False)
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"  {table_name:<22} <- {csv_filename:<22} rows loaded: {count}")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)

        print("\nLoading CSVs into tables ...")
        for table_name, csv_filename in TABLE_MAP.items():
            load_table(conn, table_name, csv_filename)

        # FactFatigue is intentionally left empty here — populated in Step 10
        fatigue_count = conn.execute("SELECT COUNT(*) FROM FactFatigue").fetchone()[0]
        print(f"  {'FactFatigue':<22} <- (empty, populated later)  rows: {fatigue_count}")

        conn.commit()
        print(f"\nDone. Database saved to {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()