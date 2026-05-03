import math, numpy as np, pandas as pd

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

def get_edges(vals): return np.quantile(vals, [0.5])
def assign_bin(val, edges): return 0 if val <= edges[0] else 1

df = pd.read_csv(INPUT_CSV)
client_ids = sorted(df["client_id"].unique())

client_edges = {}
for cid in client_ids:
    val = df[(df.client_id==cid) & (df.split=="validation")][SIGNAL]
    client_edges[cid] = get_edges(val)
df["dscp_bin"] = df.apply(lambda row: assign_bin(row[SIGNAL], client_edges[row.client_id]), axis=1)

global_bin_qhat = {}
for b in range(N_BINS):
    scores = df[(df.split=="calibration") & (df.dscp_bin==b)]["score_correct"]
    global_bin_qhat[b] = conformal_qhat(scores) if len(scores)>0 else 0.5

def shrink_qhat(cid, b):
    cal = df[(df.client_id==cid) & (df.split=="calibration")]
    bin_cal = cal[cal["dscp_bin"]==b]
    n = len(bin_cal)
    if n >= MIN_CALIB_PER_BIN:
        local_q = conformal_qhat(bin_cal["score_correct"])
    else:
        local_q = conformal_qhat(cal["score_correct"])   # fallback to overall client quantile
    lam = n/(n+TAU_V2)
    return lam * local_q + (1-lam) * global_bin_qhat[b]

rows = []
for cid in client_ids:
    test = df[(df.client_id==cid) & (df.split=="test")]
    total = len(test)

    # No CP
    top1_acc = test["correct"].mean()
    hall_no_cp = 1.0 - top1_acc

    # Fed‑Shrink‑v2
    hall_cp = 0
    correct_commits = 0
    for _, row in test.iterrows():
        b = int(row["dscp_bin"])
        q = shrink_qhat(cid, b)
        ps = lac_set(row, q)
        if len(ps) == 1:
            if row["gold"] in ps:
                correct_commits += 1
            else:
                hall_cp += 1
    useful = correct_commits / total if total>0 else 0.0
    hall_cp_rate = hall_cp / total if total>0 else 0.0

    rows.append([cid, total, round(top1_acc,4), round(hall_no_cp,4), round(useful,4), round(hall_cp_rate,4)])

test_all = df[df.split=="test"]
overall_acc = test_all["correct"].mean()
overall_hall_no_cp = 1.0 - overall_acc

overall_correct_commits = 0
overall_hall_cp = 0
for _, row in test_all.iterrows():
    cid = int(row.client_id)
    b = int(row.dscp_bin)
    q = shrink_qhat(cid, b)
    ps = lac_set(row, q)
    if len(ps) == 1:
        if row["gold"] in ps:
            overall_correct_commits += 1
        else:
            overall_hall_cp += 1
overall_useful = overall_correct_commits / len(test_all)
overall_hall_cp_rate = overall_hall_cp / len(test_all)

rows.append(["Overall", len(test_all), round(overall_acc,4), round(overall_hall_no_cp,4), round(overall_useful,4), round(overall_hall_cp_rate,4)])

header = f"{'Client':<10}{'N':<6}{'Top‑1 Acc':>10}{'Hall (No CP)':>13}{'Useful (Fed‑Shrink‑v2)':>23}{'Hall (Fed‑Shrink‑v2)':>21}"
print(header)
print("-"*len(header))
for r in rows:
    print(f"{r[0]:<10}{r[1]:<6}{r[2]:>10.4f}{r[3]:>13.4f}{r[4]:>23.4f}{r[5]:>21.4f}")