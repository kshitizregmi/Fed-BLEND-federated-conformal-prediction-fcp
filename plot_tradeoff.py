import math, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "fed_dscp_scored_examples.csv"
ALPHA_REF = 0.10
LETTERS = ["A","B","C","D"]
FEAT_COLS = ["entropy", "max_prob", "margin", "energy"]
N_BINS = 2
MIN_CALIB_PER_BIN = 10
CVAE_EPOCHS = 300          # 300 for final
AUGMENT_SIZE = 200
LATENT_DIM = 4
TAU_SHRINK = 5             # CVAE shrinkage
TAU_V2 = 20                # Fed‑Shrink‑v2 shrinkage
ALPHAS = np.round(np.arange(0.02, 0.30, 0.02), 3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = pd.read_csv(INPUT_CSV)
client_ids = sorted(df["client_id"].unique())

def conformal_qhat(scores, alpha):
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

if "aps_score_correct" not in df.columns:
    df["aps_score_correct"] = df.apply(aps_score_row, axis=1)

def get_edges(vals): return np.quantile(vals, [0.5])
def assign_bin(val, edges): return 0 if val <= edges[0] else 1

client_edges = {}
for cid in client_ids:
    val = df[(df.client_id==cid)&(df.split=="validation")]["entropy"]
    client_edges[cid] = get_edges(val)

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(1+4,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU())
        self.mu = nn.Linear(64, LATENT_DIM)
        self.logvar = nn.Linear(64, LATENT_DIM)
        self.dec = nn.Sequential(nn.Linear(LATENT_DIM+4,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,1))
    def encode(self, x, c):
        h = self.enc(torch.cat([x,c],1))
        return self.mu(h), self.logvar(h)
    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5*logvar)*torch.randn_like(logvar)
    def decode(self, z, c): return self.dec(torch.cat([z,c],1))
    def forward(self, x, c):
        mu, logvar = self.encode(x,c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z,c), mu, logvar

print("Training CVAE (for plotting, may take a while)...")
train_data = df[df.split=="calibration"]
X_train = train_data[FEAT_COLS].values.astype(np.float32)
y_train = train_data["score_correct"].values.astype(np.float32).reshape(-1,1)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
y_mean, y_std = y_train.mean(), y_train.std()
y_train = (y_train - y_mean) / y_std
loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                    batch_size=256, shuffle=True)
cvae = CVAE().to(DEVICE)
opt = torch.optim.Adam(cvae.parameters(), lr=1e-3)
for epoch in range(1, CVAE_EPOCHS+1):
    cvae.train()
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        recon, mu, logvar = cvae(y, x)
        loss = F.mse_loss(recon,y) + 0.001 * (-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp()))/y.size(0)
        loss.backward(); opt.step()
    if epoch%100==0: print(f"  epoch {epoch}")
cvae.eval()

X_all = scaler.transform(df[df.split=="calibration"][FEAT_COLS].values.astype(np.float64))
y_all = df[df.split=="calibration"]["score_correct"].values.astype(np.float64)
h = np.std(X_all, axis=0) * (len(y_all)**(-1/(4+4))) + 1e-8
g = np.std(y_all) * (len(y_all)**(-1/5)) + 1e-8

def kde_quantile(x, alpha):
    diff = (X_all - x.reshape(1,-1)) / h
    w = np.exp(-0.5 * np.sum(diff**2, axis=1))
    w /= w.sum()
    idx = np.random.choice(len(y_all), size=500, p=w)
    samples = y_all[idx] + np.random.normal(0, g, 500)
    return np.quantile(samples, 1-alpha)

def compute_all_qhats(alpha):
    global_cal = df[df.split=="calibration"]
    g_lac = conformal_qhat(global_cal["score_correct"].values, alpha)
    g_aps = conformal_qhat(global_cal["aps_score_correct"].values, alpha)

    loc_lac = {}
    loc_aps = {}
    dscp = {}
    for cid in client_ids:
        cal = df[(df.client_id==cid)&(df.split=="calibration")]
        loc_lac[cid] = conformal_qhat(cal["score_correct"].values, alpha)
        loc_aps[cid] = conformal_qhat(cal["aps_score_correct"].values, alpha)
        dscp[cid] = {}
        for b in range(N_BINS):
            bc = cal[cal["dscp_bin"]==b]
            dscp[cid][b] = conformal_qhat(bc["score_correct"].values, alpha) if len(bc)>=MIN_CALIB_PER_BIN else loc_lac[cid]

    global_bin_qhat = {}
    for b in range(N_BINS):
        scores = df[(df.split=="calibration")&(df.dscp_bin==b)]["score_correct"]
        global_bin_qhat[b] = conformal_qhat(scores.values, alpha) if len(scores)>0 else g_lac

    def shrink_v2(cid, b):
        n = len(df[(df.client_id==cid)&(df.split=="calibration")&(df.dscp_bin==b)])
        lam = n/(n+TAU_V2)
        return lam * dscp[cid][b] + (1-lam) * global_bin_qhat[b]

    # CVAE augmented quantile (per client per bin)
    cvae_qhat = {}
    for cid in client_ids:
        cvae_qhat[cid] = {}
        val = df[(df.client_id==cid)&(df.split=="validation")]
        for b in range(N_BINS):
            mask = val["entropy"].apply(lambda x: assign_bin(x, client_edges[cid])) == b
            bin_data = val[mask]
            if len(bin_data)==0:
                cvae_qhat[cid][b] = dscp[cid][b]; continue
            avg_feat = bin_data[FEAT_COLS].mean().values.astype(np.float32).reshape(1,-1)
            avg_feat = scaler.transform(avg_feat)
            with torch.no_grad():
                c = torch.tensor(avg_feat).to(DEVICE).expand(AUGMENT_SIZE,-1)
                z = torch.randn(AUGMENT_SIZE, LATENT_DIM).to(DEVICE)
                gen = cvae.decode(z, c).cpu().numpy().flatten()
            gen = gen * y_std + y_mean
            real = df[(df.client_id==cid)&(df.split=="calibration")&(df.dscp_bin==b)]["score_correct"].values
            combined = np.concatenate([real, gen]) if len(real)>0 else gen
            cvae_qhat[cid][b] = conformal_qhat(combined, alpha)

    def cvae_shrink(cid, b):
        n = len(df[(df.client_id==cid)&(df.split=="calibration")&(df.dscp_bin==b)])
        lam = n/(n+TAU_SHRINK)
        return lam * dscp[cid][b] + (1-lam) * cvae_qhat[cid][b]

    # KDE quantiles
    kde_q = {}
    for cid in client_ids:
        kde_q[cid] = {}
        val = df[(df.client_id==cid)&(df.split=="validation")]
        for b in range(N_BINS):
            mask = val["entropy"].apply(lambda x: assign_bin(x, client_edges[cid])) == b
            bin_data = val[mask]
            if len(bin_data)==0:
                kde_q[cid][b] = dscp[cid][b]; continue
            avg_feat = bin_data[FEAT_COLS].mean().values.astype(np.float64).reshape(1,-1)
            avg_feat = scaler.transform(avg_feat).flatten()
            kde_q[cid][b] = kde_quantile(avg_feat, alpha)

    def kde_shrink(cid, b):
        n = len(df[(df.client_id==cid)&(df.split=="calibration")&(df.dscp_bin==b)])
        lam = n/(n+TAU_SHRINK)
        return lam * dscp[cid][b] + (1-lam) * kde_q[cid][b]

    return {
        "global_lac": g_lac,
        "local_lac": loc_lac,
        "local_aps": loc_aps,
        "fed_dscp": dscp,
        "shrink_v2": shrink_v2,
        "cvae_cp": cvae_shrink,
        "condkde": kde_shrink,
    }

methods = ["global_lac", "local_lac", "local_aps", "fed_dscp",
           "shrink_v2", "cvae_cp", "condkde"]
labels = {
    "global_lac": "Global-LAC",
    "local_lac": "Local-LAC",
    "local_aps": "Local-APS",
    "fed_dscp": "Fed-DSCP",
    "shrink_v2": "Fed-Shrink-v2",
    "cvae_cp": "Fed-CVAE-CP (τ=5)",
    "condkde": "Fed-CondKDE-Shrink",
}
colors = {
    "global_lac": "#9ca3af",
    "local_lac": "#f59e0b",
    "local_aps": "#3b82f6",
    "fed_dscp": "#10b981",
    "shrink_v2": "#8b5cf6",
    "cvae_cp": "#ef4444",
    "condkde": "#f97316",
}

sweep_rows = []
curve_rows = []

print("\nSweeping α values...")
for alpha in ALPHAS:
    qhats = compute_all_qhats(alpha)
    test_all = df[df["split"]=="test"]

    commits = {m:0 for m in methods}
    hallus = {m:0 for m in methods}
    correct_commits = {m:0 for m in methods}
    total = 0
    sizes = {m:[] for m in methods}
    covered = {m:[] for m in methods}

    for _, row in test_all.iterrows():
        cid = int(row.client_id); gold = row.gold; b = int(row.dscp_bin)
        total += 1
        for method in methods:
            if method == "global_lac":
                q = qhats["global_lac"]
            elif method == "local_lac":
                q = qhats["local_lac"][cid]
            elif method == "local_aps":
                q = qhats["local_aps"][cid]
            elif method == "fed_dscp":
                q = qhats["fed_dscp"][cid][b]
            elif method == "shrink_v2":
                q = qhats["shrink_v2"](cid, b)
            elif method == "cvae_cp":
                q = qhats["cvae_cp"](cid, b)
            elif method == "condkde":
                q = qhats["condkde"](cid, b)

            pred = lac_set(row, q)
            sizes[method].append(len(pred))
            covered[method].append(int(gold in pred))
            if len(pred) == 1:
                commits[method] += 1
                if gold in pred:
                    correct_commits[method] += 1
                else:
                    hallus[method] += 1

    for method in methods:
        c_rate = commits[method]/total
        h_rate = hallus[method]/total
        p_commit = correct_commits[method]/commits[method] if commits[method]>0 else 1.0
        sweep_rows.append({
            "alpha": alpha,
            "method": method,
            "commit_rate": c_rate,
            "hallucination_rate": h_rate,
            "precision_commit": p_commit,
        })
        curve_rows.append({
            "alpha": alpha,
            "method": method,
            "coverage": np.mean(covered[method]),
            "set_size": np.mean(sizes[method]),
        })
    print(f"  α={alpha:.2f} done")

sweep = pd.DataFrame(sweep_rows)
curves = pd.DataFrame(curve_rows)
sweep.to_csv("tradeoff_sweep.csv", index=False)
curves.to_csv("pareto_curves.csv", index=False)

# Plot 1: Commit rate vs Hallucination rate & Precision@commit
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: commit rate vs hallucination rate
ax = axes[0]
for method in methods:
    sub = sweep[sweep["method"]==method].sort_values("commit_rate")
    ax.plot(sub["commit_rate"], sub["hallucination_rate"],
            color=colors[method], label=labels[method],
            linewidth=2.2, marker="o", markersize=3.5)
    ref = sweep[(sweep["method"]==method) & (sweep["alpha"]==ALPHA_REF)]
    if not ref.empty:
        ax.scatter(ref["commit_rate"], ref["hallucination_rate"],
                   color=colors[method], s=100, zorder=5,
                   edgecolors="white", linewidths=1.5)
ax.axhline(ALPHA_REF, color="#ef4444", linestyle="--", linewidth=1.2,
           label=f"Hallucination target ({int(ALPHA_REF*100)}%)")
ax.set_xlabel("Commit Rate", fontsize=11)
ax.set_ylabel("Hallucination Rate", fontsize=11)
ax.set_title("Commit Rate vs Hallucination Rate\n(Filled dots = α = 0.10)", fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.grid(True, alpha=0.3)

# Right: precision@commit vs commit rate
ax = axes[1]
for method in methods:
    sub = sweep[sweep["method"]==method].sort_values("commit_rate")
    ax.plot(sub["commit_rate"], sub["precision_commit"],
            color=colors[method], label=labels[method],
            linewidth=2.2, marker="o", markersize=3.5)
    ref = sweep[(sweep["method"]==method) & (sweep["alpha"]==ALPHA_REF)]
    if not ref.empty:
        ax.scatter(ref["commit_rate"], ref["precision_commit"],
                   color=colors[method], s=100, zorder=5,
                   edgecolors="white", linewidths=1.5)
ax.set_xlabel("Commit Rate", fontsize=11)
ax.set_ylabel("Precision@Commit", fontsize=11)
ax.set_title("Reliability When Committing\n(Filled dots = α = 0.10)", fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tradeoff_commit_hallucination.pdf", dpi=150, bbox_inches="tight")
plt.savefig("tradeoff_commit_hallucination.png", dpi=150, bbox_inches="tight")
print("Saved: tradeoff_commit_hallucination.pdf / .png")

# Plot 2: Coverage vs Set Size (Pareto curve)
fig, ax = plt.subplots(figsize=(8, 6))
for method in methods:
    sub = curves[curves["method"]==method].sort_values("coverage")
    ax.plot(sub["coverage"], sub["set_size"],
            color=colors[method], label=labels[method],
            linewidth=2.2, marker="o", markersize=3.5)
    ref = curves[(curves["method"]==method) & (curves["alpha"]==ALPHA_REF)]
    if not ref.empty:
        ax.scatter(ref["coverage"], ref["set_size"],
                   color=colors[method], s=100, zorder=5,
                   edgecolors="white", linewidths=1.5)

ax.axvline(1-ALPHA_REF, color="#e5e7eb", linestyle="--", linewidth=1.2,
           label=f"Target coverage ({int((1-ALPHA_REF)*100)}%)")
ax.set_xlabel("Coverage", fontsize=12)
ax.set_ylabel("Average Prediction Set Size", fontsize=12)
ax.set_title("Coverage–Efficiency Trade‑off (Pareto Curve)\nLower‑right is better. Filled dots = α = 0.10.", fontsize=11)
ax.legend(fontsize=9, framealpha=0.9)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_xlim(0.78, 1.01)
ax.set_ylim(0.9, 3.2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pareto_coverage_setsize.pdf", dpi=150, bbox_inches="tight")
plt.savefig("pareto_coverage_setsize.png", dpi=150, bbox_inches="tight")
print("Saved: pareto_coverage_setsize.pdf / .png")

# MATCHED-COVERAGE COMPARISON (Local‑LAC vs Fed‑CVAE‑CP)
print("\nMATCHED-COVERAGE COMPARISON (Local‑LAC vs Fed‑CVAE‑CP)")
lac_pts = curves[curves["method"]=="local_lac"].sort_values("coverage")
cvae_pts = curves[curves["method"]=="cvae_cp"].sort_values("coverage")
print(f"{'Coverage':>10}  {'Local-LAC sz':>14}  {'Fed-CVAE-CP sz':>16}  {'Δ set size':>10}")
print("-" * 58)
for _, lac_row in lac_pts.iterrows():
    target_cov = lac_row["coverage"]
    idx = (cvae_pts["coverage"] - target_cov).abs().idxmin()
    c_row = cvae_pts.loc[idx]
    delta = c_row["set_size"] - lac_row["set_size"]
    print(f"{target_cov:>10.1%}  {lac_row['set_size']:>14.3f}  {c_row['set_size']:>16.3f}  {delta:>+10.3f}")

#  SUMMARY TABLE (α = 0.10)
print("\n\n" + "="*80)
print("FINAL SUMMARY TABLE  (α = 0.10)")
print("="*80)

ref_sweep = sweep[sweep["alpha"] == ALPHA_REF].copy()
ref_curves = curves[curves["alpha"] == ALPHA_REF][["method", "coverage", "set_size"]].copy()
ref = ref_sweep.merge(ref_curves, on="method", how="left")
ref["useful_rate"] = ref["commit_rate"] * ref["precision_commit"]
ref["meets_target"] = ref["coverage"] >= (1 - ALPHA_REF)

display_cols = [
    "method", "coverage", "set_size", "commit_rate",
    "hallucination_rate", "precision_commit", "useful_rate", "meets_target"
]
rename_dict = {
    "method": "Method",
    "coverage": "Coverage",
    "set_size": "Avg Set Size",
    "commit_rate": "Commit Rate",
    "hallucination_rate": "Hallucination Rate",
    "precision_commit": "Precision@Commit",
    "useful_rate": "Useful Rate",
    "meets_target": "Meets Coverage Target",
}

method_order = ["global_lac", "local_lac", "local_aps", "fed_dscp", "shrink_v2", "cvae_cp", "condkde"]
ref["method"] = pd.Categorical(ref["method"], categories=method_order, ordered=True)
ref = ref.sort_values("method")
table_df = ref[display_cols].rename(columns=rename_dict)
print(table_df.round(4).to_string(index=False))

print("\nGlossary:")
print("  Coverage:            fraction of correct answer inside set (higher = safer)")
print("  Avg Set Size:        average # of answers (smaller = less abstention)")
print("  Commit Rate:         fraction of single‑answer predictions")
print("  Hallucination Rate:  fraction of committed wrong answers")
print("  Precision@Commit:    among committed, fraction correct")
print("  Useful Rate:         fraction of correct committed answers")
print("  Meets Coverage Target: True if coverage >= 0.90")

print("\nAll plots saved.")