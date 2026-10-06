"""
check_db.py
------------
Sanity-check script for worksafe.db — run this after load_to_sql.py to
confirm the schema and data loaded correctly before building anything
on top of it (EDA, features, ML models).

Data flow: reads worksafe.db, prints row counts + sample joins. Writes nothing.
"""

import sqlite3
from pathlib import Path
import pandas as pd

# Resolve relative to this file (notebooks/ is one level below root), so the
# script works no matter which directory it's run from.
DB_PATH = Path(__file__).resolve().parent.parent / "worksafe.db"


def run_query(conn, label, query):
    print(f"\n--- {label} ---")
    df = pd.read_sql_query(query, conn)
    print(df)
    return df


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found — run load_to_sql.py first.")
    # mode=ro: fail instead of silently creating an empty DB
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)

    # 1. Row counts for every table
    print("=== Row counts ===")
    tables = ["DimWorker", "DimProject", "DimActivity",
              "FactWorkerActivity", "FactAttendance", "FactFatigue"]
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<22} {count}")

    # 2. Sample rows from each dimension table
    run_query(conn, "Sample: DimWorker", "SELECT * FROM DimWorker LIMIT 5;")
    run_query(conn, "Sample: DimProject", "SELECT * FROM DimProject LIMIT 5;")
    run_query(conn, "Sample: DimActivity", "SELECT * FROM DimActivity LIMIT 5;")

    # 3. Join check: worker activity joined with worker + project names
    run_query(conn, "JOIN check: worker activity with names", """
        SELECT fwa.Worker_ID, dw.Worker_Name, dp.Project_Name,
               fwa.Date, fwa.Productivity_Index, fwa.Overtime_Hours
        FROM FactWorkerActivity fwa
        JOIN DimWorker dw ON fwa.Worker_ID = dw.Worker_ID
        JOIN DimProject dp ON fwa.Project_ID = dp.Project_ID
        LIMIT 5;
    """)

    # 4. Orphan check: any FactWorkerActivity rows pointing to a worker
    #    that doesn't exist in DimWorker? Should be 0.
    orphan_workers = conn.execute("""
        SELECT COUNT(*) FROM FactWorkerActivity fwa
        LEFT JOIN DimWorker dw ON fwa.Worker_ID = dw.Worker_ID
        WHERE dw.Worker_ID IS NULL;
    """).fetchone()[0]
    print(f"\n=== Orphan check ===")
    print(f"  FactWorkerActivity rows with no matching DimWorker: {orphan_workers}")

    # 5. Quick aggregate: avg productivity per project (a real business query)
    run_query(conn, "Avg productivity per project", """
        SELECT dp.Project_Name, ROUND(AVG(fwa.Productivity_Index), 3) AS Avg_Productivity
        FROM FactWorkerActivity fwa
        JOIN DimProject dp ON fwa.Project_ID = dp.Project_ID
        GROUP BY dp.Project_Name
        ORDER BY Avg_Productivity DESC;
    """)

    conn.close()
    print("\nDone. If orphan count is 0 and all row counts/joins look right, the DB is good.")


if __name__ == "__main__":
    main()