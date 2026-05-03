import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_ROOT = Path("scienceqa_sft_clients_with_images")
OUT_DIR = Path("eda_paper_plots")

CLIENTS = [0, 1, 2, 3, 4]
SPLITS = ["train", "validation", "calibration", "test"]

SPLIT_LABELS = {
    "train": "Train",
    "validation": "Validation",
    "calibration": "Calibration",
    "test": "Test",
}

GROUP_COL = "topic"      
TOP_N_GLOBAL = 25
TOP_N_CLIENT = 15

OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# MATPLOTLIB PAPER STYLE
# ============================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def read_jsonl(path):
    rows = []

    if not path.exists():
        print(f"Missing file: {path}")
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


records = []

for client_id in CLIENTS:
    for split in SPLITS:
        path = DATA_ROOT / f"client_{client_id}" / f"{split}.jsonl"
        rows = read_jsonl(path)

        for row in rows:
            records.append({
                "client_id": client_id,
                "split": split,
                "split_label": SPLIT_LABELS[split],
                "category": row.get("category", "UNKNOWN"),
                "subject": row.get("subject", "UNKNOWN"),
                "topic": row.get("topic", "UNKNOWN"),
                "has_image": bool(row.get("has_image", False)),
                "answer": row.get("answer", None),
            })

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(f"No rows loaded from {DATA_ROOT.resolve()}")

print("Loaded examples:", len(df))
print(df.groupby(["client_id", "split"]).size())



split_sizes = (
    df.groupby(["client_id", "split"])
    .size()
    .reset_index(name="count")
)

topic_counts = (
    df.groupby(["client_id", "split", GROUP_COL])
    .size()
    .reset_index(name="count")
)

split_sizes.to_csv(OUT_DIR / "table_split_sizes_by_client.csv", index=False)
topic_counts.to_csv(OUT_DIR / f"table_{GROUP_COL}_counts_by_client_split.csv", index=False)


pivot = (
    split_sizes
    .pivot(index="client_id", columns="split", values="count")
    .fillna(0)
    .reindex(columns=SPLITS)
)

pivot.columns = [SPLIT_LABELS[c] for c in pivot.columns]

fig, ax = plt.subplots(figsize=(8.5, 4.8))

pivot.plot(
    kind="bar",
    ax=ax,
    width=0.78,
    edgecolor="black",
    linewidth=0.4,
)

ax.set_title("Data Split Size Across Federated Clients")
ax.set_xlabel("Client ID")
ax.set_ylabel("Number of Examples")
ax.set_xticklabels([f"Client {i}" for i in pivot.index], rotation=0)
ax.legend(title="Split", frameon=False, ncol=2)

for container in ax.containers:
    ax.bar_label(container, fontsize=7, padding=2)

ax.text(
    0.01,
    -0.22,
    "Note: Each client is split into train, validation, calibration, and test subsets.",
    transform=ax.transAxes,
    fontsize=8,
    ha="left",
    va="top",
)

fig.tight_layout()

fig.savefig(OUT_DIR / "fig1_split_size_per_client.png", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig1_split_size_per_client.pdf", bbox_inches="tight")
plt.close(fig)

top_topics = df[GROUP_COL].value_counts().head(TOP_N_GLOBAL).index

matrix = (
    df[df[GROUP_COL].isin(top_topics)]
    .groupby([GROUP_COL, "client_id"])
    .size()
    .reset_index(name="count")
    .pivot(index=GROUP_COL, columns="client_id", values="count")
    .fillna(0)
)

matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=True).index]

fig, ax = plt.subplots(figsize=(8.5, max(5.5, 0.28 * len(matrix))))

im = ax.imshow(matrix.values, aspect="auto")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of Examples")

ax.set_title(f"Non-IID {GROUP_COL.capitalize()} Distribution Across Clients")
ax.set_xlabel("Federated Client")
ax.set_ylabel(GROUP_COL.capitalize())

ax.set_xticks(range(len(matrix.columns)))
ax.set_xticklabels([f"C{c}" for c in matrix.columns])

ax.set_yticks(range(len(matrix.index)))
ax.set_yticklabels(matrix.index)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        value = int(matrix.iloc[i, j])
        if value > 0:
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=6,
            )

ax.text(
    0.01,
    -0.13,
    f"Note: Only the top {TOP_N_GLOBAL} most frequent {GROUP_COL}s are shown. "
    "Darker cells indicate higher concentration in a client.",
    transform=ax.transAxes,
    fontsize=8,
    ha="left",
    va="top",
)

fig.tight_layout()

fig.savefig(OUT_DIR / f"fig2_client_vs_{GROUP_COL}_matrix.png", bbox_inches="tight")
fig.savefig(OUT_DIR / f"fig2_client_vs_{GROUP_COL}_matrix.pdf", bbox_inches="tight")
plt.close(fig)



for client_id in CLIENTS:
    client_df = df[df["client_id"] == client_id].copy()

    if client_df.empty:
        continue

    top_client_topics = (
        client_df[GROUP_COL]
        .value_counts()
        .head(TOP_N_CLIENT)
        .index
    )

    plot_df = client_df[client_df[GROUP_COL].isin(top_client_topics)]

    pivot = (
        plot_df
        .groupby([GROUP_COL, "split"])
        .size()
        .reset_index(name="count")
        .pivot(index=GROUP_COL, columns="split", values="count")
        .fillna(0)
        .reindex(columns=SPLITS)
    )

    pivot.columns = [SPLIT_LABELS[c] for c in pivot.columns]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(8.8, max(5.2, 0.35 * len(pivot))))

    pivot.plot(
        kind="barh",
        ax=ax,
        width=0.78,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_title(f"Client {client_id}: {GROUP_COL.capitalize()} Distribution by Split")
    ax.set_xlabel("Number of Examples")
    ax.set_ylabel(GROUP_COL.capitalize())
    ax.legend(title="Split", frameon=False)

    ax.text(
        0.01,
        -0.16,
        f"Note: The plot shows the top {TOP_N_CLIENT} {GROUP_COL}s for Client {client_id}. "
        "Similar split proportions indicate stratified splitting within the client.",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / f"fig3_client_{client_id}_{GROUP_COL}_distribution_by_split.png",
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / f"fig3_client_{client_id}_{GROUP_COL}_distribution_by_split.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


image_counts = (
    df.groupby(["client_id", "split", "has_image"])
    .size()
    .reset_index(name="count")
)

image_counts["type"] = image_counts["has_image"].map({
    True: "Image-based",
    False: "Text-only",
})

for client_id in CLIENTS:
    client_img = image_counts[image_counts["client_id"] == client_id]

    if client_img.empty:
        continue

    pivot = (
        client_img
        .pivot(index="split", columns="type", values="count")
        .fillna(0)
        .reindex(index=SPLITS)
    )

    pivot.index = [SPLIT_LABELS[s] for s in pivot.index]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_title(f"Client {client_id}: Image-Based vs Text-Only Examples")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of Examples")
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(title="Example Type", frameon=False)

    for container in ax.containers:
        ax.bar_label(container, fontsize=7, label_type="center")

    ax.text(
        0.01,
        -0.20,
        "Note: ScienceQA contains both image-based and text-only questions.",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / f"fig4_client_{client_id}_image_text_distribution.png",
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / f"fig4_client_{client_id}_image_text_distribution.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


print("\nPaper-ready EDA plots saved to:", OUT_DIR.resolve())

print("\nRecommended paper figures:")
print("1. fig1_split_size_per_client.pdf")
print(f"2. fig2_client_vs_{GROUP_COL}_matrix.pdf")
print(f"3. fig3_client_0_{GROUP_COL}_distribution_by_split.pdf")
print("4. Optional: fig4_client_X_image_text_distribution.pdf")