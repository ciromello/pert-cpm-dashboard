import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import tempfile, shutil

# === Setup output folder ===
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# === Load CSV safely ===
csv_path = "myfiles/PERT_CPM_Project_Template.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

try:
    df = pd.read_csv(csv_path)
except PermissionError:
    print("⚠️ File is open in another program (like Excel). Please close it and re-run.")
    exit()

# === Clean dependencies ===
df["Depends On"] = df["Depends On"].fillna("").astype(str)

# === Build directed graph ===
G = nx.DiGraph()

# Add nodes
for _, row in df.iterrows():
    G.add_node(row["Task Name"], duration=row["Expected Duration"])

# Add edges
for _, row in df.iterrows():
    deps = [d.strip() for d in row["Depends On"].split(",") if d.strip()]
    for dep in deps:
        if dep in df["Task Name"].values:
            G.add_edge(dep, row["Task Name"])

# === Forward and backward passes ===
def forward_pass(G):
    ES, EF = {}, {}
    for node in nx.topological_sort(G):
        es = max((EF[pred] for pred in G.predecessors(node)), default=0)
        dur = G.nodes[node]["duration"]
        ES[node] = es
        EF[node] = es + dur
    return ES, EF

def backward_pass(G, EF):
    LF, LS = {}, {}
    max_time = max(EF.values())
    for node in reversed(list(nx.topological_sort(G))):
        lf = min((LS[succ] for succ in G.successors(node)), default=max_time)
        dur = G.nodes[node]["duration"]
        LF[node] = lf
        LS[node] = lf - dur
    return LS, LF

ES, EF = forward_pass(G)
LS, LF = backward_pass(G, EF)

# === Slack and critical path ===
for node in G.nodes():
    G.nodes[node]["slack"] = LS[node] - ES[node]

critical_path = [n for n in G.nodes() if G.nodes[n]["slack"] == 0]
total_duration = max(EF.values())

# === Custom structured layout (top → left, spaced vertically) ===
layer = {}
for n in nx.topological_sort(G):
    preds = list(G.predecessors(n))
    layer[n] = 0 if not preds else max(layer[p] + 1 for p in preds)

# Higher in CSV = more to the left
row_order = {task: i for i, task in enumerate(df["Task Name"].dropna().unique())}

pos = {}
horizontal_scale = 6.0   # how much horizontal distance between dependency layers
vertical_scale = 4.0     # how much vertical space between tasks

for n in G.nodes:
    # invert horizontally for right-to-left
    x = layer[n] * horizontal_scale
    y = -row_order.get(n, 0) * vertical_scale
    pos[n] = (x, y)

# === Draw chart ===
fig, ax = plt.subplots(figsize=(24, 14))  # larger and taller figure
node_colors = ["lightcoral" if n in critical_path else "lightblue" for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=4200, edgecolors="black", ax=ax)
nx.draw_networkx_edges(
    G, pos, ax=ax,
    arrowstyle="->", arrowsize=16,
    connectionstyle="arc3,rad=0.08", edge_color="gray"
)

labels = {
    n: f"{n}\nES:{ES[n]} | EF:{EF[n]}\nLS:{LS[n]} | LF:{LF[n]}\nSlack:{G.nodes[n]['slack']} | Dur:{G.nodes[n]['duration']}"
    for n in G.nodes()
}
nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold", ax=ax)

plt.title(f"PERT/CPM Network Chart\nTotal Project Duration: {total_duration:.1f} days",
          fontsize=18, fontweight="bold")
plt.axis("off")
plt.tight_layout()

# === Save outputs safely ===
png_path = os.path.join(output_dir, "PERT_CPM_Network.png")
pdf_path = os.path.join(output_dir, "PERT_CPM_Network.pdf")
csv_out = os.path.join(output_dir, "PERT_CPM_Table.csv")

try:
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
except PermissionError:
    print("⚠️ Output image files are open. Please close them and re-run.")

# === Build and export PERT table ===
table_data = []
for n in G.nodes():
    table_data.append({
        "Task": n,
        "Duration": G.nodes[n]["duration"],
        "ES": ES[n],
        "EF": EF[n],
        "LS": LS[n],
        "LF": LF[n],
        "Slack": G.nodes[n]["slack"],
        "On Critical Path": "Yes" if n in critical_path else "No"
    })

pert_table = pd.DataFrame(table_data).sort_values(by="ES")

# Safe write to CSV
try:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    pert_table.to_csv(tmp.name, index=False)
    shutil.move(tmp.name, csv_out)
except PermissionError:
    print("⚠️ Could not save table CSV (file locked). Close it and re-run.")

plt.show()

print(f"\n✅ Outputs saved to '{output_dir}' folder:")
print(f"   - Network chart (PNG): {png_path}")
print(f"   - Network chart (PDF): {pdf_path}")
print(f"   - PERT table (CSV):    {csv_out}")
