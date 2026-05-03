from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("scienceqa_sft_clients_with_images")
OUT = Path("final_partition_figures")
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "calibration", "test"]
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_client(name):
    return name.replace("client_", "Client ")


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    print("Saved:", OUT / f"{name}.png")
    print("Saved:", OUT / f"{name}.pdf")



records = []

client_dirs = sorted(
    [p for p in ROOT.glob("client_*") if p.is_dir()],
    key=lambda p: int(p.name.split("_")[-1])
)

if not client_dirs:
    raise FileNotFoundError(f"No client folders found in: {ROOT.resolve()}")

for client_dir in client_dirs:
    client = clean_client(client_dir.name)

    for split in SPLITS:
        path = client_dir / f"{split}.jsonl"
        if not path.exists():
            continue

        for ex in read_jsonl(path):
            records.append({
                "client": client,
                "split": split,
                "category": ex.get("category", "Unknown"),
            })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No examples loaded. Check ROOT path.")

client_order = sorted(df["client"].unique(), key=lambda x: int(x.split()[-1]))

print(df.groupby(["client", "split"]).size().unstack(fill_value=0))



train_df = df[df["split"] == "train"].copy()

counts = (
    train_df.groupby(["client", "category"])
    .size()
    .reset_index(name="count")
)

counts["client_id"] = counts["client"].str.extract(r"(\d+)").astype(int)

cat_order = (
    counts.sort_values(["client_id", "count"], ascending=[True, False])
    ["category"]
    .drop_duplicates()
    .tolist()
)

counts["x"] = counts["category"].map({c: i for i, c in enumerate(cat_order)})
counts["y"] = counts["client"].map({c: i for i, c in enumerate(client_order)})

max_count = counts["count"].max()

# Controlled bubble size: visible but avoids overlap
counts["size"] = 60 + 500 * np.sqrt(counts["count"] / max_count)

fig_w = max(10.5, 0.48 * len(cat_order))
fig_h = 3.4

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

sc = ax.scatter(
    counts["x"],
    counts["y"],
    s=counts["size"],
    c=counts["count"],
    cmap="Blues",
    edgecolors="black",
    linewidths=0.55,
    alpha=0.93,
    vmin=0,
    vmax=max_count
)

# Bubble annotations
for _, r in counts.iterrows():
    ax.text(
        r["x"],
        r["y"],
        str(int(r["count"])),
        ha="center",
        va="center",
        fontsize=5.2,
        fontweight="bold",
        color="white" if r["count"] > 0.65 * max_count else "black",
    )

ax.set_xticks(range(len(cat_order)))
ax.set_xticklabels(
    cat_order,
    rotation=90,
    ha="center",
    va="top",
    fontsize=6.6
)

ax.set_yticks(range(len(client_order)))
ax.set_yticklabels(client_order, fontsize=8)

ax.set_xlabel("ScienceQA category", fontweight="bold", labelpad=8)
ax.set_ylabel("Federated client", fontweight="bold", labelpad=8)

ax.set_xlim(-0.5, len(cat_order) - 0.5)
ax.set_ylim(-0.5, len(client_order) - 0.5)

# Matrix grid
ax.set_xticks(np.arange(-0.5, len(cat_order), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(client_order), 1), minor=True)

ax.grid(which="minor", color="#E3E3E3", linewidth=0.65)
ax.grid(which="major", visible=False)
ax.tick_params(which="minor", bottom=False, left=False)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

# Colorbar
cbar = fig.colorbar(
    sc,
    ax=ax,
    fraction=0.025,
    pad=0.015
)

cbar.set_label(
    "Training samples",
    fontsize=8,
    fontweight="bold",
    labelpad=8
)

cbar.ax.tick_params(labelsize=7, width=0.6, length=3)
cbar.outline.set_linewidth(0.7)

plt.tight_layout()
savefig(fig, "fig1_rotated_noniid_bubble_matrix")
plt.show()


# FIGURE 2: GROUPED SPLIT-SIZE BAR PLOT
split_counts = (
    df.groupby(["client", "split"])
    .size()
    .unstack(fill_value=0)
    .reindex(client_order)
)

split_counts = split_counts[SPLITS]
split_counts = split_counts.rename(columns={
    "train": "Train",
    "validation": "Val",
    "calibration": "Calib",
    "test": "Test",
})

fig, ax = plt.subplots(figsize=(6.2, 3.0))

x = np.arange(len(split_counts))
bar_w = 0.18

offsets = np.linspace(
    -bar_w * 1.5,
    bar_w * 1.5,
    len(split_counts.columns)
)

for offset, col in zip(offsets, split_counts.columns):
    vals = split_counts[col].values

    bars = ax.bar(
        x + offset,
        vals,
        width=bar_w,
        label=col,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.9,
    )

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(split_counts.max()) * 0.012,
            str(int(v)),
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )

ax.set_xlabel("Federated client", fontweight="bold")
ax.set_ylabel("Examples", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(split_counts.index)

ax.legend(
    frameon=False,
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.17),
    columnspacing=1.0,
    handlelength=1.2,
)

ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.set_ylim(0, split_counts.values.max() * 1.18)

plt.tight_layout()
savefig(fig, "fig2_grouped_client_split_sizes_clean")
plt.show()

print("Done. Saved figures to:", OUT.resolve())