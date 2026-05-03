
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
CVAE_EPOCHS = 200
AUGMENT_SIZE = 200
LATENT_DIM = 4
TAUS = [1,5,10,20]          
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
df = pd.read_csv(INPUT_CSV)
client_ids = sorted(df["client_id"].unique())

def conformal_qhat(scores, alpha=ALPHA):
    scores = np.asarray(scores, dtype=float)
    if len(scores)==0: return 1.0
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

client_edges = {}
for cid in client_ids:
    val = df[(df.client_id==cid)&(df.split=="validation")]["entropy"]
    client_edges[cid] = get_edges(val)

dscp_qhat = {}
for cid in client_ids:
    cal = df[(df.client_id==cid)&(df.split=="calibration")]
    local_emp = conformal_qhat(cal["score_correct"])
    dscp_qhat[cid] = {}
    for b in range(N_BINS):
        bc = cal[cal["dscp_bin"]==b]
        dscp_qhat[cid][b] = conformal_qhat(bc["score_correct"]) if len(bc)>=MIN_CALIB_PER_BIN else local_emp

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
cvae.eval()

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

X_all = scaler.transform(df[df.split=="calibration"][FEAT_COLS].values.astype(np.float64))
y_all = df[df.split=="calibration"]["score_correct"].values.astype(np.float64)
h = np.std(X_all, axis=0) * (len(y_all)**(-1/(4+4))) + 1e-8
g = np.std(y_all) * (len(y_all)**(-1/5)) + 1e-8

def kde_quantile(x, n_samples=500):
    diff = (X_all - x.reshape(1,-1)) / h
    w = np.exp(-0.5 * np.sum(diff**2, axis=1))
    w /= w.sum()
    idx = np.random.choice(len(y_all), size=n_samples, p=w)
    samples = y_all[idx] + np.random.normal(0, g, n_samples)
    return np.quantile(samples, 1-ALPHA)

kde_qhat = {}
for cid in client_ids:
    kde_qhat[cid] = {}
    val = df[(df.client_id==cid)&(df.split=="validation")]
    for b in range(N_BINS):
        mask = val["entropy"].apply(lambda x: assign_bin(x, client_edges[cid])) == b
        bin_data = val[mask]
        if len(bin_data)==0:
            kde_qhat[cid][b] = dscp_qhat[cid][b]; continue
        avg_feat = bin_data[FEAT_COLS].mean().values.astype(np.float64).reshape(1,-1)
        avg_feat = scaler.transform(avg_feat).flatten()
        kde_qhat[cid][b] = kde_quantile(avg_feat)

def shrink(cid, b, aug_q, tau):
    n = len(df[(df.client_id==cid)&(df.split=="calibration")&(df.dscp_bin==b)])
    lam = n/(n+tau)
    return lam * dscp_qhat[cid][b] + (1-lam) * aug_q


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
methods = {}
methods["Fed‑DSCP"] = lambda cid,b,row: dscp_qhat[cid][b]
methods["CVAE‑no‑shrink"] = lambda cid,b,row: cvae_qhat[cid][b]
for tau in TAUS:
    methods[f"CVAE‑τ={tau}"] = lambda cid,b,row, tau=tau: shrink(cid, b, cvae_qhat[cid][b], tau)
methods["Fed‑CondKDE"] = lambda cid,b,row: kde_qhat[cid][b]
methods["Fed‑CondKDE‑Shrink"] = lambda cid,b,row: shrink(cid, b, kde_qhat[cid][b], 5)

# Run evaluation
print("\nEvaluating ablation variants...")
results = []
for name, func in methods.items():
    print(f"  {name}")
    m = evaluate(func)
    m["method"] = name
    results.append(m)

ablation_df = pd.DataFrame(results).round(4)
print("\n===== ABLATION STUDY =====")
print(ablation_df[["method","coverage","useful_rate","hallucination_rate","avg_set_size","precision_commit"]].to_string(index=False))
ablation_df.to_csv("ablation_result.csv", index=False)
print("\nSaved: ablation_result.csv")