# key_highlights_analytics_v2.py
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.io as pio
import os
from datetime import datetime, timedelta, date
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import base64

# -------------------------
# Page & paths
# -------------------------
st.set_page_config(page_title="Key Highlights & Analytics v2", layout="wide")
st.title("📊 Key Highlights & Analytics — v2")

MAIN_CSV = "myfiles/PERT_CPM_Project_Template.csv"
BACKUP_DIR = "myfiles/backups"
REPORT_DIR = "myfiles/reports"
PROGRESS_LOG = "myfiles/progress_log.csv"
os.makedirs("myfiles", exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# -------------------------
# Ensure CSV exists (minimal template)
# -------------------------
if not os.path.exists(MAIN_CSV):
    template = pd.DataFrame(columns=[
        "Task Name", "Expected Duration", "Depends On",
        "Baseline Start", "Baseline Finish", "Actual Start", "Actual Finish", "Percent Complete"
    ])
    template.to_csv(MAIN_CSV, index=False)

# -------------------------
# Load & editor
# -------------------------
df_raw = pd.read_csv(MAIN_CSV)

st.sidebar.header("📝 Edit Project Data")
edited_df = st.data_editor(
    df_raw,
    num_rows="dynamic",
    key="editor_v2",
    width="stretch",
    hide_index=True,
)

def save_with_backup(old_df, new_df):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_path = os.path.join(BACKUP_DIR, f"PERT_CPM_Project_Template_{ts}.csv")
    try:
        old_df.to_csv(backup_path, index=False)
    except Exception:
        pass
    new_df.to_csv(MAIN_CSV, index=False)
    st.sidebar.success(f"✅ Saved at {ts}")
    st.sidebar.info(f"💾 Backup: {os.path.basename(backup_path)}")

if not edited_df.equals(df_raw):
    save_with_backup(df_raw, edited_df)
    df_raw = edited_df.copy()

df = df_raw.copy()

# Ensure columns exist & coerce types
required_cols = ["Task Name", "Expected Duration", "Depends On",
                 "Baseline Start", "Baseline Finish", "Actual Start", "Actual Finish", "Percent Complete"]
for col in required_cols:
    if col not in df.columns:
        df[col] = pd.NA

df["Expected Duration"] = pd.to_numeric(df["Expected Duration"], errors="coerce").fillna(0).astype(float)
df["Percent Complete"] = pd.to_numeric(df["Percent Complete"], errors="coerce").fillna(0).astype(float)

# -------------------------
# Build graph & CPM
# -------------------------
G = nx.DiGraph()
for _, row in df.iterrows():
    task = str(row["Task Name"]).strip()
    if task != "" and pd.notna(row["Expected Duration"]):
        G.add_node(task, duration=float(row["Expected Duration"]))

for _, row in df.iterrows():
    task = str(row["Task Name"]).strip()
    deps_field = row.get("Depends On", "")
    if task in G.nodes and pd.notna(deps_field):
        deps = [d.strip() for d in str(deps_field).split(",") if d.strip()]
        for dep in deps:
            if dep in G.nodes:
                G.add_edge(dep, task)

if len(G.nodes) == 0:
    st.warning("No tasks found. Add Task Name + Expected Duration to the table to see analytics.")
    st.stop()

def forward_pass(G):
    ES, EF = {}, {}
    for node in nx.topological_sort(G):
        es = max((EF[p] for p in G.predecessors(node)), default=0)
        dur = float(G.nodes[node]["duration"])
        ES[node] = es
        EF[node] = es + dur
    return ES, EF

def backward_pass(G, EF):
    LS, LF = {}, {}
    max_time = max(EF.values()) if EF else 0
    for node in reversed(list(nx.topological_sort(G))):
        lf = min((LS[s] for s in G.successors(node)), default=max_time)
        dur = float(G.nodes[node]["duration"])
        LF[node] = lf
        LS[node] = lf - dur
    return LS, LF

ES, EF = forward_pass(G)
LS, LF = backward_pass(G, EF)

for n in G.nodes:
    G.nodes[n]["ES"] = ES[n]
    G.nodes[n]["EF"] = EF[n]
    G.nodes[n]["LS"] = LS[n]
    G.nodes[n]["LF"] = LF[n]
    G.nodes[n]["Slack"] = LS[n] - ES[n]

critical_path = [n for n in nx.topological_sort(G) if abs(G.nodes[n]["Slack"]) < 1e-9]
total_duration = max(EF.values()) if EF else 0

# metrics
avg_pct = df["Percent Complete"].replace({pd.NA: 0}).mean()
delayed = 0
for _, r in df.iterrows():
    bfin = pd.to_datetime(r.get("Baseline Finish", pd.NaT), errors="coerce")
    afin = pd.to_datetime(r.get("Actual Finish", pd.NaT), errors="coerce")
    if pd.notna(bfin) and pd.notna(afin) and afin > bfin:
        delayed += 1

# -------------------------
# Auto-log daily progress
# -------------------------
def append_progress_log(avg_pct):
    today_str = date.today().isoformat()
    if not os.path.exists(PROGRESS_LOG):
        pd.DataFrame(columns=["Date", "AvgPercentComplete"]).to_csv(PROGRESS_LOG, index=False)
    log_df = pd.read_csv(PROGRESS_LOG)
    if not ((log_df["Date"] == today_str).any()):
        new_row = {"Date": today_str, "AvgPercentComplete": float(avg_pct)}
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
        log_df.to_csv(PROGRESS_LOG, index=False)
    return pd.read_csv(PROGRESS_LOG)

progress_log_df = append_progress_log(avg_pct)

# -------------------------
# UI: Key Highlights panel
# -------------------------
st.subheader("🟨 Key Highlights")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Duration (days)", f"{total_duration:.1f}")
k2.metric("Avg % Complete", f"{avg_pct:.1f}%")
k3.metric("Critical Path Tasks", len(critical_path))
k4.metric("Delayed Tasks", delayed)
st.markdown("---")

# -------------------------
# Network Chart (matplotlib)
# -------------------------
st.subheader("🕸️ PERT/CPM Network")

# --- Custom horizontal layout: CSV order + dependency depth ---
# Compute topological levels (distance from project start)
levels = {}
for node in nx.topological_sort(G):
    preds = list(G.predecessors(node))
    if not preds:
        levels[node] = 0
    else:
        levels[node] = max(levels[p] + 1 for p in preds)

# Assign positions based on CSV row order and dependency depth
row_order = {task: i for i, task in enumerate(df["Task Name"]) if pd.notna(task)}

pos = {}
for task in G.nodes:
    x = levels.get(task, 0) * 2.5 # further left for earlier levels (note negative for right-to-left)
    y = -row_order.get(task, 0) * 3.5  # preserve CSV vertical order
    pos[task] = (x, y)


fig_net, ax = plt.subplots(figsize=(14, 12))
node_colors = ["lightcoral" if n in critical_path else "lightblue" for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2800, edgecolors="black", ax=ax)
edge_colors = ["red" if (u in critical_path and v in critical_path) else "gray" for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowstyle="->", arrowsize=12, ax=ax)
labels = {n: f"{n}\nES:{G.nodes[n]['ES']:.0f} EF:{G.nodes[n]['EF']:.0f}\nSlack:{G.nodes[n]['Slack']:.0f}\nDur:{G.nodes[n]['duration']:.0f}" for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
ax.set_title(f"PERT/CPM Network — Total Duration: {total_duration:.1f} days")
ax.axis("off")
st.pyplot(fig_net)

# -------------------------
# Gantt Chart with Color Mode Toggle
# -------------------------
st.subheader("📆 Planned Gantt Chart")
project_start = st.sidebar.date_input("Project start date (for Gantt):", datetime.today())

# Choose color mode
color_mode = st.radio("Gantt Color Mode:", ["Progress Mode", "Baseline Deviation Mode"], horizontal=True)

# Build gantt rows
gantt_rows = []
for n in G.nodes:
    start_dt = project_start + timedelta(days=int(G.nodes[n]["ES"]))
    finish_dt = project_start + timedelta(days=int(G.nodes[n]["EF"]))
    pct = df.loc[df["Task Name"] == n, "Percent Complete"].iat[0] if (df["Task Name"] == n).any() else 0
    # baseline/actual parsing (coerce)
    bfin = pd.to_datetime(df.loc[df["Task Name"] == n, "Baseline Finish"].iat[0], errors="coerce") if (df["Task Name"] == n).any() else pd.NaT
    afin = pd.to_datetime(df.loc[df["Task Name"] == n, "Actual Finish"].iat[0], errors="coerce") if (df["Task Name"] == n).any() else pd.NaT
    gantt_rows.append({
        "Task": n,
        "Start": start_dt,
        "Finish": finish_dt,
        "Percent Complete": float(pct),
        "BaselineFinish": bfin,
        "ActualFinish": afin,
        "Critical": n in critical_path
    })

gdf = pd.DataFrame(gantt_rows)

if color_mode == "Progress Mode":
    def pct_color(p):
        if pd.isna(p): return "lightgray"
        p = float(p)
        if p >= 100: return "green"
        if p > 0: return "orange"
        return "lightgray"
    gdf["Color"] = gdf["Percent Complete"].apply(pct_color)
    fig_gantt = px.timeline(
        gdf, x_start="Start", x_end="Finish", y="Task",
        color="Color",
        color_discrete_map={"green":"green","orange":"orange","lightgray":"lightgray"},
        title="Planned Gantt — Progress Mode"
    )
    fig_gantt.update_traces(customdata=gdf[["Percent Complete", "BaselineFinish", "ActualFinish"]])
    fig_gantt.update_traces(hovertemplate="<b>%{y}</b><br>Start: %{x_start}<br>Finish: %{x_end}<br>Progress: %{customdata[0]}%<br>Baseline Finish: %{customdata[1]}<br>Actual Finish: %{customdata[2]}")
else:
    # Baseline deviation mode color logic: Ahead (green) / Behind (red) / On or missing (lightgray)
    def deviation_status(row):
        b = row["BaselineFinish"]
        a = row["ActualFinish"]
        if pd.isna(b) or pd.isna(a):
            return "nodata"
        # deviation = actual - baseline in days
        dev = (pd.to_datetime(a) - pd.to_datetime(b)).days
        if dev < 0:
            return "ahead"
        elif dev > 0:
            return "behind"
        else:
            return "on"
    gdf["DevStatus"] = gdf.apply(deviation_status, axis=1)
    color_map = {"ahead":"green", "behind":"red", "on":"lightgray", "nodata":"lightgray"}
    gdf["Color"] = gdf["DevStatus"].map(color_map)
    fig_gantt = px.timeline(
        gdf, x_start="Start", x_end="Finish", y="Task",
        color="Color",
        color_discrete_map=color_map,
        title="Planned Gantt — Baseline Deviation Mode"
    )
    fig_gantt.update_traces(customdata=gdf[["Percent Complete", "BaselineFinish", "ActualFinish", "DevStatus"]])
    fig_gantt.update_traces(hovertemplate="<b>%{y}</b><br>Start: %{x_start}<br>Finish: %{x_end}<br>Baseline Finish: %{customdata[1]}<br>Actual Finish: %{customdata[2]}<br>Deviation status: %{customdata[3]}")

fig_gantt.update_yaxes(autorange="reversed")
fig_gantt.update_layout(height=600, template="plotly_white")
st.plotly_chart(fig_gantt, use_container_width=True)

# Legend for Baseline Deviation Mode
if color_mode == "Baseline Deviation Mode":
    st.markdown(
        "- 🟢 **Ahead**: Actual Finish earlier than Baseline Finish\n"
        "- 🔴 **Behind**: Actual Finish later than Baseline Finish\n"
        "- ⚪ **On / No data**: No baseline/actual to compare or on schedule"
    )
st.markdown("---")

# -------------------------
# Progress Over Time
# -------------------------
st.subheader("📈 Progress Over Time")
plog = progress_log_df.copy()
plog["Date"] = pd.to_datetime(plog["Date"], errors="coerce")
plog = plog.sort_values("Date")
if not plog.empty:
    fig_progress = px.line(plog, x="Date", y="AvgPercentComplete", markers=True, title="Average % Complete over Time")
    fig_progress.update_layout(yaxis_title="% Complete", xaxis_title="Date", template="plotly_white")
    st.plotly_chart(fig_progress, use_container_width=True)
else:
    st.info("No progress log yet.")

# -------------------------
# Export PDF and HTML (both charts + metrics)
# -------------------------
st.markdown("---")
st.subheader("📤 Export Reports")

def fig_to_png_bytes_matplotlib(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def fig_to_png_bytes_plotly(fig):
    buf = BytesIO()
    # uses kaleido under the hood
    img_bytes = pio.to_image(fig, format="png")
    buf.write(img_bytes)
    buf.seek(0)
    return buf

def generate_pdf_report(path_pdf):
    net_buf = fig_to_png_bytes_matplotlib(fig_net)
    gantt_buf = fig_to_png_bytes_plotly(fig_gantt)
    c = canvas.Canvas(path_pdf, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Key Highlights — PERT/CPM Report (v2)")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 100, "Project Overview")
    c.setFont("Helvetica", 10)
    c.drawString(60, height - 120, f"Total Duration: {total_duration:.1f} days")
    c.drawString(60, height - 135, f"Avg % Complete: {avg_pct:.1f}%")
    c.drawString(60, height - 150, f"Critical Path Tasks: {len(critical_path)}")
    c.drawString(60, height - 165, f"Delayed Tasks: {delayed}")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 190, "PERT/CPM Network Chart")
    net_img = ImageReader(net_buf)
    c.drawImage(net_img, 50, height - 560, width=500, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 50, "Planned Gantt Chart")
    gantt_img = ImageReader(gantt_buf)
    c.drawImage(gantt_img, 50, height - 560, width=500, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()

def generate_html_report(path_html):
    metrics_html = f"""
    <h1>Key Highlights — PERT/CPM Report (v2)</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>Project Overview</h2>
    <ul>
      <li>Total Duration: {total_duration:.1f} days</li>
      <li>Avg % Complete: {avg_pct:.1f}%</li>
      <li>Critical Path Tasks: {len(critical_path)}</li>
      <li>Delayed Tasks: {delayed}</li>
    </ul>
    <h2>PERT/CPM Network Chart</h2>
    """

    net_buf = fig_to_png_bytes_matplotlib(fig_net)
    net_b64 = base64.b64encode(net_buf.getvalue()).decode("utf-8")
    img_tag = f'<img src="data:image/png;base64,{net_b64}" style="max-width:900px;">'

    gantt_html = pio.to_html(fig_gantt, include_plotlyjs='cdn', full_html=False)

    full_html = f"""
    <html>
    <head><meta charset="utf-8"><title>Key Highlights Report (v2)</title></head>
    <body>
    {metrics_html}
    {img_tag}
    <h2>Planned Gantt Chart</h2>
    {gantt_html}
    </body>
    </html>
    """
    with open(path_html, "w", encoding="utf-8") as f:
        f.write(full_html)

col_pdf, col_html = st.columns(2)
with col_pdf:
    if st.button("💾 Export PDF Report (Network + Gantt)"):
        out_pdf = os.path.join(REPORT_DIR, f"Key_Highlights_Report_v2_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf")
        try:
            generate_pdf_report(out_pdf)
            st.success(f"PDF saved: {out_pdf}")
            with open(out_pdf, "rb") as fh:
                st.download_button("⬇️ Download PDF", fh, file_name=os.path.basename(out_pdf))
        except Exception as e:
            st.error(f"PDF export failed: {e}")
with col_html:
    if st.button("🌐 Export HTML Report (interactive)"):
        out_html = os.path.join(REPORT_DIR, f"Key_Highlights_Report_v2_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html")
        try:
            generate_html_report(out_html)
            st.success(f"HTML saved: {out_html}")
            with open(out_html, "rb") as fh:
                st.download_button("⬇️ Download HTML", fh, file_name=os.path.basename(out_html))
        except Exception as e:
            st.error(f"HTML export failed: {e}")

st.info(f"Reports folder: {REPORT_DIR}")
