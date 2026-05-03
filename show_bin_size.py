import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_CSV = "fed_dscp_scored_examples.csv"
SIGNAL = "entropy"
N_BINS = 2
MIN_CALIB = 10

df = pd.read_csv(INPUT_CSV)
client_ids = sorted(df["client_id"].unique())

def get_edges(vals):
    return np.quantile(vals, [0.5])

def assign_bin(val, edges):
    return 0 if val <= edges[0] else 1

# Build edges from validation set
client_edges = {}
for cid in client_ids:
    val = df[(df["client_id"] == cid) & (df["split"] == "validation")][SIGNAL]
    client_edges[cid] = get_edges(val)

# Assign bins to all rows
df["dscp_bin"] = df.apply(
    lambda row: assign_bin(row[SIGNAL], client_edges[row["client_id"]]), axis=1
)

cal = df[df["split"] == "calibration"]
counts = cal.groupby(["client_id", "dscp_bin"]).size().unstack(fill_value=0)
counts.columns = [f"Bin {int(b)}" for b in counts.columns]

print("Calibration examples per client and difficulty bin\n")
print(counts.to_string())
print(f"\nBins with ≤ {MIN_CALIB} examples are highlighted as small.\n")

for cid in counts.index:
    for b in counts.columns:
        n = counts.loc[cid, b]
        flag = "  <-- SMALL" if n <= MIN_CALIB else ""
        print(f"Client {int(cid):2d}  {b}: {n:4d}{flag}")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(counts))
width = 0.35
for i, bin_label in enumerate(counts.columns):
    bars = ax.bar(x + i*width, counts[bin_label], width, label=bin_label)
    for bar, val in zip(bars, counts[bin_label].values):
        if val <= MIN_CALIB:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=8, color='red')

ax.set_xticks(x + width/2)
ax.set_xticklabels([f"Client {int(c)}" for c in counts.index])
ax.set_ylabel("Number of calibration examples")
ax.set_title("Calibration set size per client and difficulty bin")
ax.legend()
ax.axhline(y=MIN_CALIB, color='red', linestyle='--', alpha=0.7,
           label=f'Min safe threshold ({MIN_CALIB})')
plt.tight_layout()
plt.savefig("calibration_bin_sizes.pdf", dpi=150)
plt.savefig("calibration_bin_sizes.png", dpi=150)
print("\nSaved plot: calibration_bin_sizes.pdf / .png")