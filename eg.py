# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the main result CSV
# df = pd.read_csv("main_result.csv")

# # Rename columns for display
# rename = {
#     "method": "Method",
#     "coverage": "Coverage",
#     "useful_rate": "Useful Rate",
#     "hallucination_rate": "Hallucination",
#     "avg_set_size": "Set Size",
#     "precision_commit": "Precision@Commit",
# }
# df = df[list(rename.keys())].rename(columns=rename)
# df = df.round(4)

# # Convert all values to strings
# cell_text = df.values.tolist()
# col_labels = df.columns.tolist()

# # Create figure
# fig, ax = plt.subplots(figsize=(12, 3.5))
# ax.axis("off")

# # Create table
# table = ax.table(
#     cellText=cell_text,
#     colLabels=col_labels,
#     cellLoc="center",
#     loc="center",
# )

# # Auto-adjust column widths based on text length
# table.auto_set_column_width(col=list(range(len(col_labels))))

# # Increase font size and scale
# table.auto_set_font_size(False)
# table.set_fontsize(13)
# table.scale(1.0, 1.8)   # only scale row height, leave width as computed

# # Style header and rows
# for (row, col), cell in table.get_celld().items():
#     if row == 0:
#         cell.set_facecolor("#40466e")
#         cell.set_text_props(color="white", fontweight="bold")
#     else:
#         cell.set_facecolor("#f7f7f7" if row % 2 == 0 else "white")

# plt.tight_layout(pad=0.1)
# plt.savefig("main_result_table.pdf", dpi=200, bbox_inches="tight")
# plt.savefig("main_result_table.png", dpi=200, bbox_inches="tight")
# print("Saved: main_result_table.pdf / .png")



import math, numpy as np, pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "fed_dscp_scored_examples.csv"
ALPHA = 0.10
LETTERS = ["A","B","C","D"]
SIGNAL = "entropy"
N_BINS = 2
MIN_CALIB_PER_BIN = 10
TAU_V2 = 20

def conformal_qhat(scores, alpha=ALPHA):
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0: return 1.0
    n = len(scores)
    q_level = min(math.ceil((n+1)*(1-alpha))/n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))

def get_probs(row): return {L: float(row[f"p_{L}"]) for L in LETTERS}
def top1_label(row): return max(get_probs(row), key=get_probs(row).get)
def lac_set(row, qhat):
    p = get_probs(row)
    s = [y for y,v in p.items() if 1.0-v <= qhat]
    return s or [top1_label(row)]

def aps_score_row(row):
    p = get_probs(row)
    items = sorted(p.items(), key=lambda kv: kv[1], reverse=True)
    cum = 0.0
    for y,v in items:
        cum += v
        if y == row["gold"]: return cum
    return 1.0

def aps_set(row, qhat):
    p = get_probs(row)
    items = sorted(p.items(), key=lambda kv: kv[1], reverse=True)
    s, cum = [], 0.0
    for y,v in items:
        cum += v
        if cum <= qhat: s.append(y)
        else: break
    return s or [top1_label(row)]

def get_edges(vals): return np.quantile(vals, [0.5])
def assign_bin(val, edges): return 0 if val <= edges[0] else 1

df = pd.read_csv(INPUT_CSV)
client_ids = sorted(df["client_id"].unique())

if "aps_score_correct" not in df.columns:
    df["aps_score_correct"] = df.apply(aps_score_row, axis=1)

client_edges = {}
for cid in client_ids:
    val = df[(df.client_id==cid) & (df.split=="validation")][SIGNAL]
    client_edges[cid] = get_edges(val)
df["dscp_bin"] = df.apply(lambda row: assign_bin(row[SIGNAL], client_edges[row.client_id]), axis=1)

# ---------- Quantile functions ----------
def get_quantile(method, cid, b):
    cal = df[(df.client_id==cid) & (df.split=="calibration")]
    if method == "global_lac":
        return conformal_qhat(df[df.split=="calibration"]["score_correct"])
    elif method == "local_lac":
        return conformal_qhat(cal["score_correct"])
    elif method == "local_aps":
        return conformal_qhat(cal["aps_score_correct"])
    elif method == "fed_dscp":
        bin_cal = cal[cal["dscp_bin"]==b]
        if len(bin_cal) >= MIN_CALIB_PER_BIN:
            return conformal_qhat(bin_cal["score_correct"])
        else:
            return conformal_qhat(cal["score_correct"])
    elif method == "shrink_v2":
        global_bin = df[(df.split=="calibration") & (df.dscp_bin==b)]["score_correct"]
        gq = conformal_qhat(global_bin) if len(global_bin)>0 else 0.5
        bin_cal = cal[cal["dscp_bin"]==b]
        n = len(bin_cal)
        if n >= MIN_CALIB_PER_BIN:
            lq = conformal_qhat(bin_cal["score_correct"])
        else:
            lq = conformal_qhat(cal["score_correct"])
        lam = n/(n+TAU_V2)
        return lam * lq + (1-lam) * gq
    else:
        raise ValueError


methods = ["global_lac", "local_lac", "local_aps", "fed_dscp", "shrink_v2"]
results = {}
for method in methods:
    commits = hallu = correct = 0
    covered = []
    sizes = []
    for _, row in df[df.split=="test"].iterrows():
        cid = int(row.client_id)
        b = int(row.dscp_bin)
        gold = row.gold
        q = get_quantile(method, cid, b)
        if method == "local_aps":
            ps = aps_set(row, q)
        else:
            ps = lac_set(row, q)
        covered.append(int(gold in ps))
        sizes.append(len(ps))
        if len(ps) == 1:
            commits += 1
            if gold in ps:
                correct += 1
            else:
                hallu += 1
    total = len(df[df.split=="test"])
    cov = np.mean(covered)
    useful = correct / total if total>0 else 0.0
    hall = hallu / total if total>0 else 0.0
    sz = np.mean(sizes)
    prec = correct / commits if commits>0 else 1.0
    results[method] = {
        "Coverage": round(cov,4),
        "Useful Rate": round(useful,4),
        "Hallucination": round(hall,4),
        "Set Size": round(sz,4),
        "Precision@Commit": round(prec,4),
    }

method_names = {
    "global_lac": "Global‑LAC",
    "local_lac": "Local‑LAC",
    "local_aps": "Local‑APS",
    "fed_dscp": "Fed‑DSCP",
    "shrink_v2": "Fed‑Shrink‑v2",
}
rows = []
for key, name in method_names.items():
    r = results[key]
    rows.append([name, r["Coverage"], r["Useful Rate"], r["Hallucination"], r["Set Size"], r["Precision@Commit"]])

df_tab = pd.DataFrame(rows, columns=["Method", "Coverage", "Useful Rate", "Hallucination", "Set Size", "Precision@Commit"])
df_tab = df_tab.round(4)

# ---------- Plot table ----------
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis("off")
table = ax.table(cellText=df_tab.values, colLabels=df_tab.columns, cellLoc="center", loc="center")
table.auto_set_column_width(col=list(range(len(df_tab.columns))))
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1.0, 1.8)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#40466e")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#f7f7f7" if row % 2 == 0 else "white")

plt.tight_layout(pad=0.1)
plt.savefig("main_result_table_correct.pdf", dpi=200, bbox_inches="tight")
plt.savefig("main_result_table_correct.png", dpi=200, bbox_inches="tight")
print("Saved: main_result_table_correct.pdf / .png")