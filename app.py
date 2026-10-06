"""
app.py  (Step 11a: layout + Executive Overview)
------------------------------------------------
WorkSafe AI dashboard (Streamlit), following the design in the project:
  - Left navigation: Executive Overview, Fatigue Risk, Teams, AI Decision Center
  - Right panel: three synced slicers (Worker, Project, Team) + Reset filters
  - Footer: Export (added in a later sub-step)

Data flow:
  notebooks/worker_features.csv --+
  notebooks/fatigue_model.pkl   --+--> score_all_workers() --> one row per worker
  notebooks/cluster_model.pkl   --+                                  |
  notebooks/anomaly_model.pkl   --+                                  v
  worksafe.db (FactWorkerActivity, DimProject) ---> slicers filter ---> pages

Run with:  streamlit run app.py
"""

import os
import sqlite3
from io import BytesIO

import joblib
import altair as alt
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "worksafe.db")
NOTEBOOKS_DIR = os.path.join(SCRIPT_DIR, "notebooks")
FEATURES_PATH = os.path.join(NOTEBOOKS_DIR, "worker_features.csv")

PAGES = ["Executive Overview", "Fatigue Risk", "Teams", "AI Decision Center"]

ALL_WORKERS, ALL_PROJECTS, ALL_TEAMS = "All workers", "All projects", "All teams"

RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {"Low": "#5B9A1F", "Medium": "#B8741A", "High": "#D95B2F", "Critical": "#A31515"}

RECOMMENDATIONS = {
    "Critical": "Give the worker a rest day soon, cut overtime, and have a supervisor review the workload.",
    "High": "Limit overtime this week and make sure full breaks are taken.",
    "Medium": "Keep monitoring; check that breaks reach the 45-minute target.",
    "Low": "No action needed.",
}

st.set_page_config(page_title="WorkSafe AI", page_icon="W", layout="wide")

STYLE = """
<style>
[data-testid="stSidebar"] { background: #FDEFF3; }
.app-header { background:#FDEFF3; border:1px solid #F3D3DF; border-radius:12px;
              padding:14px 20px; display:flex; align-items:center; justify-content:space-between; }
.app-header .left { display:flex; align-items:center; gap:14px; }
.logo { background:#C2185B; color:#fff; width:44px; height:44px; border-radius:10px;
        display:flex; align-items:center; justify-content:center; font-weight:700; font-size:20px; }
.app-title { font-size:22px; font-weight:700; color:#7B1246; line-height:1.2; }
.app-tag { font-size:12px; color:#C2185B; }
.app-user { font-size:13px; color:#7B1246; }
.page-title { font-size:24px; font-weight:700; margin-bottom:0; }
.kpi { border:1px solid #F3D3DF; border-radius:10px; padding:14px 16px; background:#fff; }
.kpi-icon { width:30px; height:30px; border-radius:7px; display:flex; align-items:center;
            justify-content:center; font-size:15px; margin-bottom:8px; }
.kpi-label { font-size:12px; color:#666; }
.kpi-value { font-size:28px; font-weight:700; color:#2b2b2b; }
.panel-title { font-size:12px; font-weight:700; color:#7B1246; letter-spacing:1px; margin-bottom:6px; }
.qa-title { font-size:14px; font-weight:700; color:#C2185B; }
.footer-text { font-size:13px; color:#C2185B; padding-top:8px; }
.stButton > button { border:1px solid #C2185B; color:#C2185B; }
</style>
"""


# ---------------------------------------------------------------
# Loading models and data
# ---------------------------------------------------------------
@st.cache_resource
def load_models():
    fatigue = joblib.load(os.path.join(NOTEBOOKS_DIR, "fatigue_model.pkl"))
    cluster = joblib.load(os.path.join(NOTEBOOKS_DIR, "cluster_model.pkl"))
    anomaly = joblib.load(os.path.join(NOTEBOOKS_DIR, "anomaly_model.pkl"))
    return fatigue, cluster, anomaly


@st.cache_data
def load_activity():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT Worker_ID, Date, Overtime_Hours, Productivity_Index FROM FactWorkerActivity;",
        conn,
    )
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return df


@st.cache_data
def load_projects():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT Project_ID, Project_Name FROM DimProject;", conn)
    conn.close()
    return df


# ---------------------------------------------------------------
# Scoring: combines the three trained models for every worker
# ---------------------------------------------------------------
@st.cache_data
def score_all_workers():
    fatigue, cluster, anomaly = load_models()
    df = pd.read_csv(FEATURES_PATH)
    df = df.merge(load_projects(), on="Project_ID", how="left")

    # Random Forest -> risk level, and fatigue score = chance of High or Critical (%)
    rf = fatigue["model"]
    proba = pd.DataFrame(rf.predict_proba(df[fatigue["features"]]), columns=rf.classes_)
    df["Risk_Level"] = rf.predict(df[fatigue["features"]])
    df["Fatigue_Score"] = ((proba["High"] + proba["Critical"]) * 100).round(1)

    # K-Means -> performance segment
    scaled = cluster["scaler"].transform(df[cluster["features"]])
    cluster_ids = cluster["model"].predict(scaled)
    df["Segment"] = [cluster["cluster_names"][int(c)] for c in cluster_ids]

    # Isolation Forest -> anomaly flag (predict returns -1 for anomalies)
    df["Is_Anomaly"] = (anomaly["model"].predict(df[anomaly["features"]]) == -1).astype(int)

    df["Recommendation"] = df["Risk_Level"].map(RECOMMENDATIONS)
    return df


# ---------------------------------------------------------------
# Slicers (shared by every page, kept in session state)
# ---------------------------------------------------------------
def reset_filters():
    st.session_state["f_worker"] = ALL_WORKERS
    st.session_state["f_project"] = ALL_PROJECTS
    st.session_state["f_team"] = ALL_TEAMS


def render_slicers(scores):
    st.markdown('<div class="panel-title">SLICERS</div>', unsafe_allow_html=True)
    st.selectbox("Worker", [ALL_WORKERS] + sorted(scores["Worker_ID"].unique()), key="f_worker")
    st.selectbox("Project", [ALL_PROJECTS] + sorted(scores["Project_Name"].dropna().unique()), key="f_project")
    st.selectbox("Team", [ALL_TEAMS] + sorted(scores["Team_ID"].unique()), key="f_team")
    st.button("Reset filters", on_click=reset_filters)


def apply_filters(scores, activity):
    f = scores
    if st.session_state["f_worker"] != ALL_WORKERS:
        f = f[f["Worker_ID"] == st.session_state["f_worker"]]
    if st.session_state["f_project"] != ALL_PROJECTS:
        f = f[f["Project_Name"] == st.session_state["f_project"]]
    if st.session_state["f_team"] != ALL_TEAMS:
        f = f[f["Team_ID"] == st.session_state["f_team"]]
    acts = activity[activity["Worker_ID"].isin(f["Worker_ID"])]
    return f, acts


# ---------------------------------------------------------------
# Pages
# ---------------------------------------------------------------
def kpi_card(icon, color, label, value):
    return (f'<div class="kpi"><div class="kpi-icon" style="background:{color}">{icon}</div>'
            f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>')


def answer_question(question, filtered):
    """
    Rule-based Q&A: matches keywords in the question to a canned lookup
    against the current (filtered) worker scores. This is NOT an LLM —
    it's simple keyword matching, same as the "student project" approach
    described in the build guide's chatbot section.
    """
    q = question.lower().strip()
    if not q:
        return "Type a question first, e.g. \"which workers have high fatigue risk\"."

    def worker_list(df, cols=("Worker_ID", "Worker_Name", "Team_ID", "Risk_Level", "Fatigue_Score")):
        if df.empty:
            return "None in the current selection."
        lines = [
            f"- {r.Worker_ID} ({r.Worker_Name}, Team {r.Team_ID}) — {r.Risk_Level} risk, {r.Fatigue_Score:.0f}% fatigue score"
            for r in df[list(cols)].itertuples()
        ]
        return "\n".join(lines)

    if any(k in q for k in ["critical", "high fatigue", "high risk"]):
        levels = ["Critical"] if "critical" in q and "high" not in q else ["High", "Critical"]
        subset = filtered[filtered["Risk_Level"].isin(levels)].sort_values("Fatigue_Score", ascending=False).head(10)
        return f"Workers at {'/'.join(levels)} risk (top 10 shown):\n\n" + worker_list(subset)

    if "low risk" in q:
        subset = filtered[filtered["Risk_Level"] == "Low"].head(10)
        return f"{len(filtered[filtered['Risk_Level'] == 'Low'])} workers are Low risk. Sample:\n\n" + worker_list(subset)

    if "anomaly" in q or "unusual" in q:
        subset = filtered[filtered["Is_Anomaly"] == 1]
        return f"{len(subset)} workers flagged as anomalies:\n\n" + worker_list(subset)

    if "team" in q:
        by_team = filtered.groupby("Team_ID")["Avg_Productivity"].mean().mul(100).round(1).sort_values()
        worst, best = by_team.index[0], by_team.index[-1]
        return (f"Team productivity ranges from {by_team.iloc[0]:.1f}% ({worst}, lowest) "
                f"to {by_team.iloc[-1]:.1f}% ({best}, highest).")

    if "overtime" in q:
        subset = filtered.sort_values("Avg_Overtime_Hours", ascending=False).head(5)
        lines = [f"- {r.Worker_ID} ({r.Worker_Name}) — {r.Avg_Overtime_Hours:.1f}h avg overtime"
                 for r in subset.itertuples()]
        return "Highest average overtime:\n\n" + "\n".join(lines)

    if "productiv" in q:
        avg = filtered["Avg_Productivity"].mean() * 100
        return f"Average productivity across {len(filtered)} workers in the current selection is {avg:.1f}%."

    if "how many" in q and "worker" in q:
        return f"{len(filtered)} workers are in the current selection."

    return ("I can answer questions about risk levels, teams, overtime, productivity, "
            "worker counts, and anomalies. Try: \"which workers have high fatigue risk\", "
            "\"which team has the lowest productivity\", or \"who has the most overtime\".")


def show_overview(filtered, acts):
    st.markdown('<div class="page-title">Executive Overview</div>', unsafe_allow_html=True)
    st.caption("Overall workforce performance and key insights")

    high_risk = int(filtered["Risk_Level"].isin(["High", "Critical"]).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("W", "#5B9A1F", "Workers shown", len(filtered)), unsafe_allow_html=True)
    c2.markdown(kpi_card("P", "#5B9A1F", "Avg productivity",
                         f"{acts['Productivity_Index'].mean() * 100:.0f}%"), unsafe_allow_html=True)
    c3.markdown(kpi_card("R", "#B8741A", "High risk", high_risk), unsafe_allow_html=True)
    c4.markdown(kpi_card("O", "#3B8BDB", "Avg overtime",
                         f"{acts['Overtime_Hours'].mean():.1f}h"), unsafe_allow_html=True)

    st.markdown("#### Productivity by worker")
    chart_df = filtered.assign(Productivity=filtered["Avg_Productivity"] * 100)
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Worker_ID:N", sort="-y",
                    axis=alt.Axis(labels=False, ticks=False, title="Workers (highest productivity first)")),
            y=alt.Y("Productivity:Q", title="Avg productivity (%)"),
            color=alt.Color("Risk_Level:N",
                            scale=alt.Scale(domain=RISK_ORDER, range=[RISK_COLORS[r] for r in RISK_ORDER]),
                            legend=alt.Legend(title="Risk level")),
            tooltip=["Worker_ID", "Worker_Name", "Team_ID",
                     alt.Tooltip("Productivity:Q", format=".1f", title="Productivity %"),
                     "Risk_Level", "Fatigue_Score"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")

    st.markdown('<div class="qa-title">Q&A -- ask a question about your data</div>', unsafe_allow_html=True)
    q_col, b_col = st.columns([5, 1])
    question = q_col.text_input("Question", placeholder="e.g. which workers have high fatigue risk",
                                 label_visibility="collapsed", key="qa_input")
    ask = b_col.button("Send")
    if ask:
        st.session_state["qa_answer"] = answer_question(question, filtered)
    if st.session_state.get("qa_answer"):
        st.markdown(st.session_state["qa_answer"])
    st.caption("Answers are matched by keyword against the current selection, not generated by an AI model.")


def show_fatigue_risk(filtered):
    st.markdown('<div class="page-title">Fatigue Risk</div>', unsafe_allow_html=True)
    st.caption("Risk-level breakdown across the current selection")

    counts = filtered["Risk_Level"].value_counts().reindex(RISK_ORDER, fill_value=0)
    cols = st.columns(4)
    for col, level in zip(cols, RISK_ORDER):
        col.markdown(
            kpi_card(level[0], RISK_COLORS[level], level, int(counts[level])),
            unsafe_allow_html=True,
        )

    dist_df = counts.rename_axis("Risk_Level").reset_index(name="Workers")
    bar_col, donut_col = st.columns(2)

    with bar_col:
        st.markdown("#### Risk distribution")
        bar = (
            alt.Chart(dist_df)
            .mark_bar()
            .encode(
                x=alt.X("Risk_Level:N", sort=RISK_ORDER, title="Risk level"),
                y=alt.Y("Workers:Q"),
                color=alt.Color("Risk_Level:N",
                                scale=alt.Scale(domain=RISK_ORDER,
                                                range=[RISK_COLORS[r] for r in RISK_ORDER]),
                                legend=None),
                tooltip=["Risk_Level", "Workers"],
            )
            .properties(height=280)
        )
        st.altair_chart(bar, width="stretch")

    with donut_col:
        st.markdown("#### Risk breakdown")
        donut = (
            alt.Chart(dist_df)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta("Workers:Q"),
                color=alt.Color("Risk_Level:N",
                                scale=alt.Scale(domain=RISK_ORDER,
                                                range=[RISK_COLORS[r] for r in RISK_ORDER]),
                                legend=alt.Legend(title="Risk level")),
                tooltip=["Risk_Level", "Workers"],
            )
            .properties(height=280)
        )
        st.altair_chart(donut, width="stretch")

    st.markdown("#### Workers currently at High or Critical risk")
    at_risk = (
        filtered[filtered["Risk_Level"].isin(["High", "Critical"])]
        [["Worker_ID", "Worker_Name", "Team_ID", "Project_Name", "Risk_Level",
          "Fatigue_Score", "Recommendation"]]
        .sort_values("Fatigue_Score", ascending=False)
        .rename(columns={"Project_Name": "Project"})
    )
    if at_risk.empty:
        st.success("No workers at High or Critical risk in the current selection.")
    else:
        st.dataframe(at_risk, hide_index=True, width="stretch")


def show_teams(filtered, acts):
    st.markdown('<div class="page-title">Teams</div>', unsafe_allow_html=True)
    st.caption("Team productivity ranking and trend over time")

    team_avg = (
        filtered.groupby("Team_ID")["Avg_Productivity"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index(name="Productivity")
        .sort_values("Productivity", ascending=False)
    )

    st.markdown("#### Team productivity ranking")
    rank_chart = (
        alt.Chart(team_avg)
        .mark_bar(color="#C2185B")
        .encode(
            y=alt.Y("Team_ID:N", sort="-x", title="Team"),
            x=alt.X("Productivity:Q", title="Avg productivity (%)"),
            tooltip=["Team_ID", alt.Tooltip("Productivity:Q", format=".1f")],
        )
        .properties(height=max(220, 24 * len(team_avg)))
    )
    st.altair_chart(rank_chart, width="stretch")

    st.markdown("#### Productivity trend by team")
    team_lookup = filtered[["Worker_ID", "Team_ID"]]
    trend = acts.merge(team_lookup, on="Worker_ID", how="inner")
    if trend.empty:
        st.info("No activity data for the current selection.")
    else:
        # Weekly buckets (not daily) keep the heatmap readable and less noisy
        trend = trend.copy()
        trend["Week_Start"] = trend["Date"] - pd.to_timedelta(trend["Date"].dt.weekday, unit="D")
        weekly = (
            trend.groupby(["Week_Start", "Team_ID"])["Productivity_Index"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index(name="Productivity")
        )
        heatmap = (
            alt.Chart(weekly)
            .mark_rect()
            .encode(
                x=alt.X("Week_Start:T", title="Week"),
                y=alt.Y("Team_ID:N", title="Team", sort=sorted(weekly["Team_ID"].unique())),
                color=alt.Color("Productivity:Q", title="Avg productivity (%)",
                                scale=alt.Scale(scheme="redyellowgreen")),
                tooltip=["Team_ID", "Week_Start:T", alt.Tooltip("Productivity:Q", format=".1f")],
            )
            .properties(height=max(220, 26 * weekly["Team_ID"].nunique()))
        )
        st.altair_chart(heatmap, width="stretch")
        st.caption("Weekly average productivity per team. Red cells mark weeks worth a closer look.")


def show_decision_center(filtered):
    st.markdown('<div class="page-title">AI Decision Center</div>', unsafe_allow_html=True)
    st.caption("Highest-priority worker and the ranked at-risk list")

    ranked = filtered.sort_values("Fatigue_Score", ascending=False)
    top = ranked.iloc[0]

    st.markdown(
        f'''
        <div style="border:2px solid {RISK_COLORS[top["Risk_Level"]]}; border-radius:12px;
                    padding:20px; background:#fff;">
          <div style="font-size:12px; letter-spacing:1px; color:{RISK_COLORS[top["Risk_Level"]]};
                      font-weight:700;">TOP PRIORITY</div>
          <div style="font-size:22px; font-weight:700; margin-top:4px;">
            {top["Worker_Name"]} &nbsp;<span style="color:#888; font-weight:400;">({top["Worker_ID"]})</span>
          </div>
          <div style="color:#666; margin-bottom:10px;">
            Team {top["Team_ID"]} &middot; {top["Project_Name"] if pd.notna(top["Project_Name"]) else "Unassigned project"}
          </div>
          <div style="display:flex; gap:28px; margin-bottom:12px;">
            <div><div class="kpi-label">Fatigue score</div>
                 <div class="kpi-value">{top["Fatigue_Score"]:.0f}%</div></div>
            <div><div class="kpi-label">Risk level</div>
                 <div class="kpi-value" style="color:{RISK_COLORS[top["Risk_Level"]]}">{top["Risk_Level"]}</div></div>
            <div><div class="kpi-label">Productivity</div>
                 <div class="kpi-value">{top["Avg_Productivity"] * 100:.0f}%</div></div>
            <div><div class="kpi-label">Overtime</div>
                 <div class="kpi-value">{top["Avg_Overtime_Hours"]:.1f}h</div></div>
            <div><div class="kpi-label">Consecutive days</div>
                 <div class="kpi-value">{top["Consecutive_Days"]:.0f}</div></div>
          </div>
          <div style="background:#FDEFF3; border-radius:8px; padding:10px 14px;">
            <b>Recommendation:</b> {top["Recommendation"]}
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("#### Ranked at-risk workers")
    table = ranked[
        ["Worker_ID", "Worker_Name", "Team_ID", "Project_Name", "Risk_Level",
         "Fatigue_Score", "Avg_Overtime_Hours", "Consecutive_Days", "Recommendation"]
    ].rename(columns={"Project_Name": "Project", "Avg_Overtime_Hours": "Avg Overtime (h)"})
    st.dataframe(table, hide_index=True, width="stretch")


def show_placeholder(name):
    st.markdown(f'<div class="page-title">{name}</div>', unsafe_allow_html=True)
    st.info(f"The {name} page is built in a later sub-step.")


# ---------------------------------------------------------------
# App shell: header, left navigation, content + right slicers, footer
# ---------------------------------------------------------------
def build_pdf_report(filtered):
    """
    Builds a PDF export scoped to whatever the slicers currently show,
    matching the "Export your filtered report" behaviour in the design.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title="WorkSafe AI Report")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleBrand", parent=styles["Title"], textColor=colors.HexColor("#C2185B"))
    header_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FDEFF3")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])

    elements = [
        Paragraph("WorkSafe AI &ndash; Workforce Report", title_style),
        Paragraph(f"Scope: {len(filtered)} workers, based on the filters selected when exported.",
                  styles["Normal"]),
        Spacer(1, 14),
    ]

    counts = filtered["Risk_Level"].value_counts().reindex(RISK_ORDER, fill_value=0)
    kpi_table = Table(
        [["Metric", "Value"],
         ["Workers shown", len(filtered)],
         ["Avg productivity", f"{filtered['Avg_Productivity'].mean() * 100:.1f}%"],
         ["High risk (High + Critical)", int(counts['High'] + counts['Critical'])],
         ["Avg overtime", f"{filtered['Avg_Overtime_Hours'].mean():.1f}h"]],
        hAlign="LEFT",
    )
    kpi_table.setStyle(header_style)
    elements += [kpi_table, Spacer(1, 16), Paragraph("Risk Level Breakdown", styles["Heading2"])]

    risk_table = Table([["Risk Level", "Workers"]] + [[lvl, int(counts[lvl])] for lvl in RISK_ORDER], hAlign="LEFT")
    risk_table.setStyle(header_style)
    elements += [risk_table, Spacer(1, 16), Paragraph("Workers at High or Critical Risk", styles["Heading2"])]

    at_risk = filtered[filtered["Risk_Level"].isin(["High", "Critical"])].sort_values(
        "Fatigue_Score", ascending=False
    )
    if at_risk.empty:
        elements.append(Paragraph("None in the current selection.", styles["Normal"]))
    else:
        rows = [["Worker", "Team", "Risk", "Fatigue %", "Recommendation"]]
        for r in at_risk.itertuples():
            rows.append([f"{r.Worker_ID} ({r.Worker_Name})", r.Team_ID, r.Risk_Level,
                        f"{r.Fatigue_Score:.0f}%", r.Recommendation])
        at_risk_table = Table(rows, hAlign="LEFT", colWidths=[110, 40, 50, 55, 190])
        at_risk_table.setStyle(header_style)
        elements.append(at_risk_table)

    elements += [
        Spacer(1, 18),
        Paragraph(
            "Scores are decision-support output from a synthetic dataset and rule-based "
            "recommendations. This is not a medical diagnosis.",
            styles["Italic"],
        ),
    ]

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()


def main():
    st.markdown(STYLE, unsafe_allow_html=True)

    scores = score_all_workers()
    activity = load_activity()

    st.markdown(
        '<div class="app-header"><div class="left"><div class="logo">W</div><div>'
        '<div class="app-title">WorkSafe AI</div>'
        '<div class="app-tag">Safer workers · Smarter decisions · Stronger projects</div>'
        '</div></div><div class="app-user">user</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

    content_col, slicer_col = st.columns([4, 1.2])
    with slicer_col:
        render_slicers(scores)          # slicers first, so their values are ready for the page
    filtered, acts = apply_filters(scores, activity)

    with content_col:
        if filtered.empty:
            st.warning("No workers match the current filters. Try Reset filters.")
        elif page == "Executive Overview":
            show_overview(filtered, acts)
        elif page == "Fatigue Risk":
            show_fatigue_risk(filtered)
        elif page == "Teams":
            show_teams(filtered, acts)
        elif page == "AI Decision Center":
            show_decision_center(filtered)
        else:
            show_placeholder(page)

    st.write("")
    f1, f2, f3 = st.columns([3, 2, 1])
    f1.markdown('<div class="footer-text">WorkSafe AI · Construction workforce analytics</div>',
                unsafe_allow_html=True)
    f2.markdown('<div class="footer-text">Export your filtered report</div>', unsafe_allow_html=True)
    with f3:
        f3.download_button(
            "Export",
            data=build_pdf_report(filtered),
            file_name="worksafe_ai_report.pdf",
            mime="application/pdf",
        )


main()
