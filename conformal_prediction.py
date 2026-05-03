import math, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "fed_dscp_scored_examples.csv"
ALPHA = 0.10
LETTERS = ["A","B","C","D"]
FEAT_COLS = ["entropy","max_prob","margin","energy"]
N_BINS = 2
MIN_CALIB_PER_BIN = 10
CVAE_EPOCHS = 300
AUGMENT_SIZE = 200
LATENT_DIM = 4
TAU_SHRINK = 5                # best tau for CVAE
TAU_V2 = 20                   # tau for Fed‑Shrink‑DSCP v2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    val = df[(df.client_id==cid) & (df.split=="validation")]["entropy"]
    client_edges[cid] = get_edges(val)

global_cal = df[df.split=="calibration"]
global_lac_qhat = conformal_qhat(global_cal["score_correct"])
global_aps_qhat = conformal_qhat(global_cal["aps_score_correct"])

local_lac_qhat = {}
local_aps_qhat = {}
dscp_qhat = {}          # Fed‑DSCP
for cid in client_ids:
    cal = df[(df.client_id==cid)&(df.split=="calibration")]
    local_lac_qhat[cid] = conformal_qhat(cal["score_correct"])
    local_aps_qhat[cid] = conformal_qhat(cal["aps_score_correct"])
    dscp_qhat[cid] = {}
    for b in range(N_BINS):
        bc = cal[cal["dscp_bin"]==b]
        dscp_qhat[cid][b] = conformal_qhat(bc["score_correct"]) if len(bc)>=MIN_CALIB_PER_BIN else local_lac_qhat[cid]


global_bin_qhat = {}
for b in range(N_BINS):
    scores = df[(df.split=="calibration") & (df.dscp_bin==b)]["score_correct"]
    global_bin_qhat[b] = conformal_qhat(scores) if len(scores)>0 else global_lac_qhat

def shrink_v2_qhat(cid, b):
    n = len(df[(df.client_id==cid)&(df.split=="calibration")&(df.dscp_bin==b)])
    lam = n/(n+TAU_V2)
    return lam * dscp_qhat[cid][b] + (1-lam) * global_bin_qhat[b]

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

# Training data
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
print("Training CVAE...")
for epoch in range(1, CVAE_EPOCHS+1):
    cvae.train()
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        recon, mu, logvar = cvae(y, x)
        loss = F.mse_loss(recon,y) + 0.001 * (-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp()))/y.size(0)
        loss.backward(); opt.step()
    if epoch%50==0: print(f"  epoch {epoch}")
cvae.eval()

# CVAE‑augmented quantiles
cvae_qhat = {}
for cid in client_ids:
    cvae_qhat[cid] = {}
    val = df[(df.client_id==cid)&(df.split=="validation")]
    for b in range(N_BINS):
        mask = val["entropy"].apply(lambda x: assign_bin(x, client_edges[cid])) == b
        bin_data = val[mask]
        if len(bin_data)==0:
            cvae_qhat[cid][b] = dscp_qhat[cid][b]; continue
        avg_feat = bin_data[FEAT_COLS].mean().values.astype(np.float32).reshape(1,-1)
        avg_feat = scaler.transform(avg_feat)
        with torch.no_grad():
            c = torch.tensor(avg_feat).to(DEVICE).expand(AUGMENT_SIZE,-1)
            z = torch.randn(AUGMENT_SIZE, LATENT_DIM).to(DEVICE)
            gen = cvae.decode(z, c).cpu().numpy().flatten()
        gen = gen * y_std + y_mean
        real = df[(df.client_id==cid)&(df.split=="calibration")&(df.dscp_bin==b)]["score_correct"].values
        combined = np.concatenate([real, gen]) if len(real)>0 else gen
        cvae_qhat[cid][b] = conformal_qhat(combined)

def cvae_shrink_qhat(cid, b):
    n = len(df[(df.client_id==cid)&(df.split=="calibration")&(df.dscp_bin==b)])
    lam = n/(n+TAU_SHRINK)
    return lam * dscp_qhat[cid][b] + (1-lam) * cvae_qhat[cid][b]

def evaluate(builder):
    rows = []
    for _, row in df[df.split=="test"].iterrows():
        cid = int(row.client_id); gold = row.gold; b = int(row.dscp_bin)
        q = builder(cid, b, row)
        pred = lac_set(row, q)
        committed = len(pred)==1
        correct_commit = committed and (gold in pred)
        rows.append({
            "covered": gold in pred, "set_size": len(pred),
            "committed": committed, "correct_commit": correct_commit,
            "hallucination": committed and not (gold in pred)
        })
    d = pd.DataFrame(rows)
    prec = d[d.committed==1]["correct_commit"].mean() if d.committed.sum()>0 else np.nan
    return {
        "coverage": d.covered.mean(),
        "useful_rate": d.correct_commit.mean(),
        "hallucination_rate": d.hallucination.mean(),
        "avg_set_size": d.set_size.mean(),
        "precision_commit": prec
    }

# Build methods
methods = {
    "Global‑LAC": lambda cid,b,row: global_lac_qhat,
    "Local‑LAC": lambda cid,b,row: local_lac_qhat[cid],
    "Local‑APS": lambda cid,b,row: local_aps_qhat[cid],
    "Fed‑DSCP": lambda cid,b,row: dscp_qhat[cid][b],
    "Fed‑Shrink‑v2": lambda cid,b,row: shrink_v2_qhat(cid,b),
    "Fed‑CVAE‑CP (τ=5)": lambda cid,b,row: cvae_shrink_qhat(cid,b),
}

print("\nEvaluating all methods...")
results = []
for name, func in methods.items():
    print(f"  {name}")
    m = evaluate(func)
    m["method"] = name
    results.append(m)

res_df = pd.DataFrame(results).round(4)
print("\n===== MAIN RESULT TABLE =====")
print(res_df[["method","coverage","useful_rate","hallucination_rate","avg_set_size","precision_commit"]].to_string(index=False))
res_df.to_csv("main_result.csv", index=False)
print("\nSaved: main_result.csv")