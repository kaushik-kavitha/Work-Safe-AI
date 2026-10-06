# WorkSafe AI

A workforce fatigue-risk and productivity dashboard for construction project management,
built end-to-end in Python: from raw CSV data through SQL, feature engineering, three
machine learning models, and an interactive Streamlit dashboard.

## What it does

WorkSafe AI tracks 150 workers across 6 construction projects over 90 days and surfaces:
- **Fatigue risk** per worker (Low / Medium / High / Critical), predicted by a Random Forest model
- **Performance segments**, discovered by K-Means clustering
- **Anomalous productivity patterns**, flagged by an Isolation Forest model
- A filterable dashboard (by worker, project, or team) with KPI cards, charts, a ranked
  at-risk worker list, a keyword-based Q&A box, and PDF report export

## pipeline
CsvData/ (raw data)
│
▼
load_to_sql.py ──> worksafe.db (SQLite, 6-table schema)
│
▼
notebooks/02_Feature_Engineering.py ──> worker_features.csv
│
├──> notebooks/03_Fatigue_Model.py ──> fatigue_model.pkl (Random Forest)
├──> notebooks/04_Clustering.py ──> cluster_model.pkl (K-Means)
└──> notebooks/05_Anomaly_Detection.py ──> anomaly_model.pkl (Isolation Forest)
│
▼

Project structure

WorkSafe_AI/
├── CsvData/ # raw workforce data
├── notebooks/ # EDA, feature engineering, and the 3 ML models
├── SQL/ # database schema (create_tables.sql)
├── app.py # Streamlit dashboard
├── generate_data.py # validates the dataset against the expected schema
├── load_to_sql.py # loads CSVs into worksafe.db
└── requirements.txt

## Running it locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

python generate_data.py
python load_to_sql.py
python notebooks/02_Feature_Engineering.py
python notebooks/03_Fatigue_Model.py
python notebooks/04_Clustering.py
python notebooks/05_Anomaly_Detection.py

streamlit run app.py
```

## Models

| Model | Type | Purpose |
|---|---|---|
| Random Forest | Supervised classification | Predicts fatigue risk level from 5 engineered features (overtime ratio, break adequacy, consecutive days, workload score, productivity change) |
| K-Means | Unsupervised clustering | Groups workers into performance segments |
| Isolation Forest | Unsupervised anomaly detection | Flags workers with unusual productivity patterns |

**Note:** the dataset is synthetic and this project is decision-support, not a medical
or safety diagnosis. Fatigue risk labels are derived from a transparent weighted formula
over the engineered features, not from real-world ground truth.

## Limitations

- Dataset is synthetic (generated/validated, not collected from real sites)
- Q&A box uses keyword matching against live dashboard data, not an LLM
- No authentication/login — intended for local, single-user use

