-- ============================================================
-- WorkSafe AI — Database Schema
-- Dimension tables = reference/lookup data
-- Fact tables      = measured records (activity, attendance, ML output)
-- ============================================================

-- ---------------- DIMENSION TABLES ----------------

CREATE TABLE IF NOT EXISTS DimWorker (
    Worker_ID          TEXT PRIMARY KEY,
    Worker_Name        TEXT NOT NULL,
    Worker_Type        TEXT,
    Skill_Level        TEXT,
    Experience_Years   INTEGER,
    Team_ID             TEXT,
    Project_ID          TEXT,
    Join_Date           TEXT
);

CREATE TABLE IF NOT EXISTS DimProject (
    Project_ID     TEXT PRIMARY KEY,
    Project_Name   TEXT NOT NULL,
    Location        TEXT,
    Project_Type    TEXT,
    Start_Date       TEXT,
    Budget            INTEGER,
    End_Date          TEXT
);

CREATE TABLE IF NOT EXISTS DimActivity (
    Activity_ID       TEXT PRIMARY KEY,
    Activity_Name     TEXT NOT NULL,
    Activity_Type     TEXT,
    Difficulty_Level  INTEGER
);

-- ---------------- FACT TABLES ----------------

CREATE TABLE IF NOT EXISTS FactWorkerActivity (
    Record_ID            TEXT PRIMARY KEY,
    Worker_ID             TEXT NOT NULL,
    Project_ID             TEXT NOT NULL,
    Activity_ID             TEXT NOT NULL,
    Date                      TEXT NOT NULL,
    Shift                     TEXT,
    Hours_Worked             REAL,
    Overtime_Hours           REAL,
    Break_Minutes             INTEGER,
    Workload_Level            TEXT,
    Output_Units               INTEGER,
    Expected_Output             INTEGER,
    Productivity_Index          REAL,
    Consecutive_Days             INTEGER,
    FOREIGN KEY (Worker_ID) REFERENCES DimWorker(Worker_ID),
    FOREIGN KEY (Project_ID) REFERENCES DimProject(Project_ID),
    FOREIGN KEY (Activity_ID) REFERENCES DimActivity(Activity_ID)
);

CREATE TABLE IF NOT EXISTS FactAttendance (
    Attendance_ID    TEXT PRIMARY KEY,
    Worker_ID          TEXT NOT NULL,
    Date                TEXT NOT NULL,
    Present_Flag         INTEGER,
    Late_Minutes          INTEGER,
    Leave_Flag             INTEGER,
    FOREIGN KEY (Worker_ID) REFERENCES DimWorker(Worker_ID)
);

-- FactFatigue starts EMPTY. It is populated later (Step 10) by
-- score_worker(), which combines the Random Forest, K-Means, and
-- Isolation Forest model outputs for each worker.
CREATE TABLE IF NOT EXISTS FactFatigue (
    Fatigue_ID        INTEGER PRIMARY KEY AUTOINCREMENT,
    Worker_ID           TEXT NOT NULL,
    Date_Scored          TEXT,
    Fatigue_Score          REAL,
    Risk_Level               TEXT,
    Cluster                   INTEGER,
    Is_Anomaly                 INTEGER,
    Recommendation               TEXT,
    FOREIGN KEY (Worker_ID) REFERENCES DimWorker(Worker_ID)
);