import pandas as pd

df = pd.read_csv("fed_dscp_scored_examples.csv")
test = df[df["split"] == "test"]

print("=== Top‑1 Hallucination (No Conformal Prediction) ===\n")
print(f"{'Client':<10}{'Test Size':<12}{'Top‑1 Accuracy':<16}{'Hallucination Rate':<20}")
print("-" * 58)

for cid in sorted(test["client_id"].unique()):
    c_df = test[test["client_id"] == cid]
    n = len(c_df)
    acc = c_df["correct"].mean()          # fraction where top‑1 == gold
    hall = 1 - acc                         # fraction where top‑1 is wrong
    print(f"{cid:<10}{n:<12}{acc:<16.4f}{hall:<20.4f}")

# Overall
n_all = len(test)
acc_all = test["correct"].mean()
hall_all = 1 - acc_all
print("-" * 58)
print(f"{'Overall':<10}{n_all:<12}{acc_all:<16.4f}{hall_all:<20.4f}")