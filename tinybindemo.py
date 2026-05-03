import math, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_CSV = "fed_dscp_scored_examples.csv"
ALPHA = 0.10
LETTERS = ["A","B","C","D"]
SIGNAL = "entropy"
N_BINS = 2
TAU_V2 = 20

df = pd.read_csv(INPUT_CSV)
client_ids = sorted(df["client_id"].unique())

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

# Bin edges
def get_edges(vals): return np.quantile(vals, [0.5])
def assign_bin(val, edges): return 0 if val <= edges[0] else 1

client_edges = {}
for cid in client_ids:
    val = df[(df.client_id==cid) & (df.split=="validation")][SIGNAL]
    client_edges[cid] = get_edges(val)

df["dscp_bin"] = df.apply(lambda row: assign_bin(row[SIGNAL], client_edges[row.client_id]), axis=1)

CID = 3
BIN = 1
KEEP = 3

cal_mask = (df["client_id"] == CID) & (df["split"] == "calibration")
bin_mask = cal_mask & (df["dscp_bin"] == BIN)
all_idx = df[bin_mask].index
keep_idx = np.random.choice(all_idx, size=KEEP, replace=False)
drop_idx = set(all_idx) - set(keep_idx)

# Remove those points from calibration — they won’t be used
df_tiny = df.copy()
df_tiny.loc[list(drop_idx), "split"] = "dropped"

# ---------- Global per‑bin quantiles (from full original data) ----------
global_bin_qhat = {}
for b in range(N_BINS):
    scores = df[(df.split=="calibration") & (df.dscp_bin==b)]["score_correct"]
    global_bin_qhat[b] = conformal_qhat(scores) if len(scores)>0 else 0.5

# Bin 1 quantile from only 3 points (dangerous)
tiny_cal = df_tiny[(df_tiny.client_id==CID) & (df_tiny.split=="calibration") & (df_tiny.dscp_bin==BIN)]
tiny_q = conformal_qhat(tiny_cal["score_correct"])   # only 3 points

# Shrunken version
n_tiny = len(tiny_cal)
lam = n_tiny / (n_tiny + TAU_V2)
shrunken_q = lam * tiny_q + (1 - lam) * global_bin_qhat[BIN]

# Bin 0 uses full calibration
cal_other = df_tiny[(df_tiny.client_id==CID) & (df_tiny.split=="calibration") & (df_tiny.dscp_bin==0)]
q_other = conformal_qhat(cal_other["score_correct"]) if len(cal_other)>0 else tiny_q

# ---------- Evaluate on Client 3's test set ----------
test_cid = df[(df.client_id==CID) & (df.split=="test")]

def evaluate(builder):
    commits = hallu = correct = total = 0
    covered, sizes = [], []
    for _, row in test_cid.iterrows():
        b = int(row["dscp_bin"])
        q = builder(b)
        pred = lac_set(row, q)
        sizes.append(len(pred))
        covered.append(int(row["gold"] in pred))
        if len(pred) == 1:
            commits += 1
            if row["gold"] in pred:
                correct += 1
            else:
                hallu += 1
        total += 1
    cov = np.mean(covered)
    use = correct / total if total else 0
    hall = hallu / total if total else 0
    prec = correct / commits if commits else 1.0
    return {"Coverage": cov, "Useful Rate": use, "Hallucination": hall, "Set Size": np.mean(sizes), "Precision@Commit": prec}

results = {
    "Fed‑DSCP (3‑point quantile)": evaluate(lambda b: tiny_q if b == BIN else q_other),
    "Fed‑Shrink‑v2 (with global anchor)": evaluate(lambda b: shrunken_q if b == BIN else q_other),
}

for name, m in results.items():
    print(f"{name}    Coverage: {m['Coverage']:.3f}  Useful: {m['Useful Rate']:.3f}  Hallucination: {m['Hallucination']:.3f}  Set Size: {m['Set Size']:.2f}  Prec@Commit: {m['Precision@Commit']:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
metrics = ["Coverage", "Useful Rate", "Hallucination"]
x = np.arange(len(metrics))
width = 0.35
for i, (name, vals) in enumerate(results.items()):
    values = [vals[m] for m in metrics]
    ax.bar(x + i*width, values, width, label=name)
ax.set_xticks(x + width/2)
ax.set_xticklabels(metrics)
ax.set_title("Client 3: Bin 1 only has 3 calibration examples")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("tiny_bin_demo_final.pdf", dpi=150)
plt.savefig("tiny_bin_demo_final.png", dpi=150)
print("\nSaved plot: tiny_bin_demo_final.pdf / .png")