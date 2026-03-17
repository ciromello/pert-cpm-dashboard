# pert_cpm_dashboard_live.py
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="PERT/CPM Live Dashboard", layout="wide")
st.title("📅 PERT/CPM Live Dashboard — edit data and see results immediately")

# -------------------------
# File / backup setup
# -------------------------
MAIN_CSV = "myfiles/PERT_CPM_Project_Template.csv"
BACKUP_DIR = "myfiles/backups"
os.makedirs("myfiles", exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Create default csv if missing
if not os.path.exists(MAIN_CSV):
    df_default = pd.DataFrame(columns=[
        "Task Name", "Expected Duration", "Depends On",
        "Baseline Start", "Baseline Finish", "Actual Start", "Actual Finish", "Percent Complete"
    ])
    df_default.to_csv(MAIN_CSV, index=False)

# Load CSV
df_orig = pd.read_csv(MAIN_CSV)

# Keep a copy of the original in session_state to detect changes
if "saved_df" not in st.session_state:
    st.session_state["saved_df"] = df_orig.copy()

# -------------------------
# Sidebar: editor + controls
# -------------------------
st.sidebar.header("📝 Edit Project Data")
edited_df = st.data_editor(
    st.session_state["saved_df"],
    num_rows="dynamic",
    key="task_editor",
    width="stretch",
    hide_index=True,
)

# Recalculate/refresh control (optional)
st.sidebar.markdown("### ⚙️ Controls")
recalc_mode = st.sidebar.radio(
    "Update behavior",
    ("Instant", "On-demand (click Recalculate)"),
    index=0,
    help="Instant = updates as soon as you edit the table. On-demand = click Recalculate to update."
)
if recalc_mode == "On-demand (click Recalculate)":
    do_recalc = st.sidebar.button("🔄 Recalculate")
else:
    do_recalc = True  # always trigger when Instant

# -------------------------
# Auto-save + backup (only when table changed)
# -------------------------
def save_with_backup(old_df, new_df):
    # Create timestamped backup of old_df then overwrite main CSV
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_path = os.path.join(BACKUP_DIR, f"PERT_CPM_Project_Template_{ts}.csv")
    try:
        old_df.to_csv(backup_path, index=False)
    except Exception:
        # best-effort but don't crash
        pass
    new_df.to_csv(MAIN_CSV, index=False)
    st.sidebar.success(f"✅ Auto-saved at {ts}")
    st.sidebar.info(f"💾 Backup: {os.path.basename(backup_path)}")

# If data changed in editor, update session and optionally save immediately
if not edited_df.equals(st.session_state["saved_df"]):
    # Update session var so the editor shows the updated data consistently
    st.session_state["saved_df"] = edited_df.copy()

    # Save backup + overwrite CSV right away
    save_with_backup(df_orig, edited_df)
    # Update df_orig to reflect saved file
    df_orig = edited_df.copy()

# Only proceed to compute if user wants recalculation (Instant or clicked)
if do_recalc:
    df = st.session_state["saved_df"].copy()

    # Ensure required columns exist
    for col in ["Task Name", "Expected Duration", "Depends On",
                "Baseline Start", "Baseline Finish", "Actual Start", "Actual Finish", "Percent Complete"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Try to coerce data types
    # Duration -> numeric
    df["Expected Duration"] = pd.to_numeric(df["Expected Duration"], errors="coerce").fillna(0).astype(float)
    # Percent Complete -> numeric (optional)
    if "Percent Complete" in df.columns:
        df["Percent Complete"] = pd.to_numeric(df["Percent Complete"], errors="coerce").fillna(0).astype(float)

    # Build directed graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        task = str(row["Task Name"]).strip()
        if task and pd.notna(row["Expected Duration"]):
            G.add_node(task, duration=float(row["Expected Duration"]))

    # Add edges from 'Depends On' (comma-separated)
    for _, row in df.iterrows():
        task = str(row["Task Name"]).strip()
        deps_field = row.get("Depends On", "")
        if pd.notna(deps_field) and task in G.nodes:
            deps = [d.strip() for d in str(deps_field).split(",") if d.strip()]
            for dep in deps:
                if dep in G.nodes:
                    G.add_edge(dep, task)

    # CPM calculations
    def forward_pass(G):
        ES, EF = {}, {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            es = max((EF[p] for p in preds), default=0)
            dur = float(G.nodes[node]["duration"])
            ES[node] = es
            EF[node] = es + dur
        return ES, EF

    def backward_pass(G, EF):
        LS, LF = {}, {}
        max_time = max(EF.values()) if EF else 0
        for node in reversed(list(nx.topological_sort(G))):
            succs = list(G.successors(node))
            lf = min((LS[s] for s in succs), default=max_time)
            dur = float(G.nodes[node]["duration"])
            LF[node] = lf
            LS[node] = lf - dur
        return LS, LF

    ES, EF = forward_pass(G)
    LS, LF = backward_pass(G, EF)

    # attach ES/EF/LS/LF/Slack to nodes
    for n in G.nodes:
        G.nodes[n]["ES"] = ES[n]
        G.nodes[n]["EF"] = EF[n]
        G.nodes[n]["LS"] = LS[n]
        G.nodes[n]["LF"] = LF[n]
        G.nodes[n]["Slack"] = LS[n] - ES[n]

    critical_path = [n for n in nx.topological_sort(G) if abs(G.nodes[n]["Slack"]) < 1e-9]
    total_duration = max(EF.values()) if EF else 0

    # Sidebar controls: project start date and options
    st.sidebar.header("📅 Project Settings")
    project_start = st.sidebar.date_input("Project start date:", datetime.today())
    show_progress = st.sidebar.checkbox("Show progress on Gantt (Percent Complete)", value=True)
    show_baseline = st.sidebar.checkbox("Auto-set Baseline from plan when blank", value=True)

    # Tabs for views
    tab_net, tab_gantt, tab_comp = st.tabs(["🕸️ Network", "📆 Gantt", "📊 Baseline vs Actual"])

    # ---- NETWORK TAB ----
    with tab_net:
        # Structured left-to-right layout based on dependency depth and CSV order
        layer = {}
        for n in nx.topological_sort(G):
            preds = list(G.predecessors(n))
            layer[n] = 0 if not preds else max(layer[p] + 1 for p in preds)

        row_order = {task: i for i, task in enumerate(df["Task Name"].dropna().unique())}

        pos = {}
        for n in G.nodes:
            # x = horizontal position (left to right by dependency level)
            x = layer[n]
            # y = vertical position (CSV order, top row = top)
            y = -row_order.get(n, 0) * 4.5  # increase vertical spacing (↑ this number for more space)
            pos[n] = (x, y)



        viz_mode = st.radio("Visualization mode:", ["🔴 Critical Path", "🌈 Slack Heatmap"], horizontal=True)

        fig, ax = plt.subplots(figsize=(18, 14))
        if viz_mode == "🔴 Critical Path":
            colors = ["lightcoral" if n in critical_path else "lightblue" for n in G.nodes]
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=["red" if (u in critical_path and v in critical_path) else "gray" for u, v in G.edges],
                                   arrowstyle="->", arrowsize=14)
        else:
            slack_vals = [G.nodes[n]["Slack"] for n in G.nodes]
            cmap = plt.cm.coolwarm_r
            norm = mcolors.Normalize(vmin=min(slack_vals), vmax=max(slack_vals))
            colors = [cmap(norm(G.nodes[n]["Slack"])) for n in G.nodes]
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label("Slack (days)", rotation=270, labelpad=15)

            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrowstyle="->", arrowsize=14)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=2800, edgecolors="black")
        labels = {n: f"{n}\nES:{G.nodes[n]['ES']:.0f} EF:{G.nodes[n]['EF']:.0f}\nLS:{G.nodes[n]['LS']:.0f} LF:{G.nodes[n]['LF']:.0f}\nSlack:{G.nodes[n]['Slack']:.0f}\nDur:{G.nodes[n]['duration']:.0f}" for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        ax.set_title(f"PERT/CPM Network — Total Duration: {total_duration:.1f} days")
        ax.axis("off")
        st.pyplot(fig)

    # ---- GANTT TAB ----
    with tab_gantt:
        # Build planned Gantt (ES/EF -> dates)
        planned = []
        for n in G.nodes:
            start_date = project_start + timedelta(days=int(G.nodes[n]["ES"]))
            end_date = project_start + timedelta(days=int(G.nodes[n]["EF"]))
            planned.append({
                "Task": n,
                "Start": start_date,
                "Finish": end_date,
                "Duration": G.nodes[n]["duration"],
                "Percent Complete": float(df.loc[df["Task Name"] == n, "Percent Complete"].iat[0]) if (df["Task Name"] == n).any() else 0
            })
            # optionally auto-fill baseline in edited table and later save it
            if show_baseline:
                mask = df["Task Name"] == n
                if mask.any():
                    if pd.isna(df.loc[mask, "Baseline Start"]).all():
                        df.loc[mask, "Baseline Start"] = start_date
                    if pd.isna(df.loc[mask, "Baseline Finish"]).all():
                        df.loc[mask, "Baseline Finish"] = end_date

        planned_df = pd.DataFrame(planned)

        # Save baseline back to CSV if auto-fill occurred
        df.to_csv(MAIN_CSV, index=False)

        if show_progress and "Percent Complete" in planned_df.columns:
            # Map percent -> color
            def pct_color(p):
                if pd.isna(p): return "lightgray"
                p = float(p)
                if p >= 100: return "green"
                if p > 0: return "orange"
                return "lightgray"
            planned_df["Color"] = planned_df["Percent Complete"].apply(pct_color)
            fig_g = px.timeline(
                planned_df, x_start="Start", x_end="Finish", y="Task",
                color="Color", color_discrete_map={"green":"green","orange":"orange","lightgray":"lightgray"},
                title="Planned Schedule (with progress color)"
            )
            fig_g.update_traces(customdata=planned_df[["Percent Complete"]])
            fig_g.update_traces(hovertemplate="<b>%{y}</b><br>Start: %{x_start}<br>Finish: %{x_end}<br>Progress: %{customdata[0]}%")
        else:
            fig_g = px.timeline(planned_df, x_start="Start", x_end="Finish", y="Task",
                                color="Percent Complete" if "Percent Complete" in planned_df.columns else None,
                                title="Planned Schedule")
        fig_g.update_yaxes(autorange="reversed")
        fig_g.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig_g, use_container_width=True)

    # ---- BASELINE vs ACTUAL TAB ----
    with tab_comp:
        st.markdown("### Baseline vs Actual Comparison")

        # Build comparison rows from edited df: baseline rows + actual rows
        comp = []
        for _, row in df.iterrows():
            task = row["Task Name"]
            # parse baseline and actual
            def safe_date(val):
                """Convert a value to datetime or NaT if invalid."""
                if pd.isna(val):
                    return pd.NaT
                if isinstance(val, bool):
                    return pd.NaT
                try:
                    return pd.to_datetime(val, errors="coerce")
                except Exception:
                    return pd.NaT

            bstart = safe_date(row.get("Baseline Start"))
            bfinish = safe_date(row.get("Baseline Finish"))
            astart = safe_date(row.get("Actual Start"))
            afinish = safe_date(row.get("Actual Finish"))

            if pd.notna(bstart) and pd.notna(bfinish):
                comp.append({"Task": task, "Type": "Baseline", "Start": bstart, "Finish": bfinish})
            if pd.notna(astart) and pd.notna(afinish):
                comp.append({"Task": task, "Type": "Actual", "Start": astart, "Finish": afinish})

        if comp:
            comp_df = pd.DataFrame(comp)
            fig_comp = px.timeline(comp_df, x_start="Start", x_end="Finish", y="Task", color="Type",
                                   color_discrete_map={"Baseline":"gray","Actual":"dodgerblue"},
                                   title="Baseline vs Actual")
            fig_comp.update_yaxes(autorange="reversed")
            fig_comp.update_layout(height=600, template="plotly_white")
            st.plotly_chart(fig_comp, use_container_width=True)

            # delay summary
            summary = []
            for _, row in df.iterrows():
                task = row["Task Name"]
                try:
                    bfin = pd.to_datetime(row.get("Baseline Finish", pd.NaT))
                    afin = pd.to_datetime(row.get("Actual Finish", pd.NaT))
                    if pd.notna(bfin) and pd.notna(afin):
                        delay = (afin - bfin).days
                        status = "✅ On Time" if delay <= 0 else f"⚠️ {delay} days late"
                        summary.append({"Task": task, "Delay (days)": delay, "Status": status})
                except Exception:
                    pass
            if summary:
                st.dataframe(pd.DataFrame(summary))
        else:
            st.info("Add Baseline Start/Finish and Actual Start/Finish to the table to compare progress.")
else:
    st.info("No recalculation requested yet. Set update to Instant or click Recalculate.")
