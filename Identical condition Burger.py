import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam, LBFGS
from tqdm import tqdm
from scipy.integrate import odeint
import time

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

nu = 0.01 / np.pi
pi = torch.tensor(np.pi, device=device)
print(f"nu     : {nu:.6f}")

# ------------------------------------------------------------------
# Fourier Embedding  (ORIGINAL — unchanged)
# ------------------------------------------------------------------
class FourierEmbedding(nn.Module):
    def __init__(self, in_dim=2, n_freq=32, sigma=5.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * n_freq   # 64

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

# ------------------------------------------------------------------
# Pseudo-sequence generator  (ORIGINAL — unchanged)
# ------------------------------------------------------------------
def pseudo_sequence(xt, dx):
    """Input: xt=[N,2]  Output: [N,3,2]"""
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    return torch.stack([
        torch.cat([x - dx, t], dim=1),
        torch.cat([x,       t], dim=1),
        torch.cat([x + dx, t], dim=1),
    ], dim=1)

# ------------------------------------------------------------------
# Transformer Block  (ORIGINAL — unchanged, GELU feed-forward)
# ------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        xn = self.norm1(x)
        h, _ = self.attn(xn, xn, xn)
        x = x + h
        return x + self.ff(self.norm2(x))

# ------------------------------------------------------------------
# Fourier Trans-PINN  (ORIGINAL architecture — unchanged)
# Learnable log-weights REMOVED for fair comparison (fixed 1.0)
# ------------------------------------------------------------------
class FourierTransPINN(nn.Module):
    def __init__(self, d_model=64, nhead=4, n_blocks=2,
                 n_freq=32, sigma=5.0):
        super().__init__()
        self.fourier      = FourierEmbedding(2, n_freq, sigma)
        self.embed        = nn.Sequential(
            nn.Linear(self.fourier.out_dim, d_model), nn.Tanh())
        self.transformers = nn.ModuleList(
            [TransformerBlock(d_model, nhead) for _ in range(n_blocks)])
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256), nn.Tanh(),
            nn.Linear(256,     256), nn.Tanh(),
            nn.Linear(256,     128), nn.Tanh(),
            nn.Linear(128,       1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xt, dx):
        seq     = pseudo_sequence(xt, dx)   # [N, 3, 2]
        N, S, _ = seq.shape
        flat    = seq.reshape(N * S, 2)
        feat    = self.fourier(flat)         # [N*3, 64]
        h       = self.embed(feat).reshape(N, S, -1)   # [N, 3, 64]
        for blk in self.transformers:
            h = blk(h)
        return self.mlp(h[:, 1, :])          # center token [N, 1]

model = FourierTransPINN().to(device)
print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

# ------------------------------------------------------------------
# Training data  (EQUALIZED)
# ------------------------------------------------------------------
N_RES = 10_000
dx    = torch.tensor(2.0 / 200, device=device)

# Collocation points — random sampling, fixed throughout training (no RAR)
def new_colloc():
    pts = torch.rand(N_RES, 2, device=device)
    pts[:, 0] = pts[:, 0] * 2.0 - 1.0   # x: [-1, 1]
    return pts.requires_grad_(True)

res = new_colloc()

# IC  (t = 0)
x_ic      = torch.linspace(-1, 1, 512).reshape(-1, 1).to(device)
t_ic      = torch.zeros_like(x_ic)
ic        = torch.cat([x_ic, t_ic], dim=1)
u_ic_true = -torch.sin(pi * x_ic)

# BC  (x = ±1, all t)
t_bc = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)
bc_l = torch.cat([-torch.ones_like(t_bc), t_bc], dim=1)
bc_r = torch.cat([ torch.ones_like(t_bc), t_bc], dim=1)

# ------------------------------------------------------------------
# Loss function  (EQUALIZED — all weights fixed = 1.0)
# ------------------------------------------------------------------
def pinn_loss(res_pts):
    u    = model(res_pts, dx)
    g1   = torch.autograd.grad(u.sum(), res_pts, create_graph=True)[0]
    u_x  = g1[:, 0:1]
    u_t  = g1[:, 1:2]
    u_xx = torch.autograd.grad(
               u_x.sum(), res_pts, create_graph=True)[0][:, 0:1]

    f        = u_t + u * u_x - nu * u_xx
    loss_res = torch.mean(f ** 2)
    loss_ic  = torch.mean((model(ic, dx) - u_ic_true) ** 2)
    loss_bc  = (torch.mean(model(bc_l, dx) ** 2) +
                torch.mean(model(bc_r, dx) ** 2))

    # Equal weights = 1.0 for all terms
    return loss_res + loss_ic + loss_bc

# ------------------------------------------------------------------
# Training  (EQUALIZED total Adam steps = 10,000)
# ------------------------------------------------------------------
# The 3-stage structure is an internal scheduling decision for the
# attention mechanism — it does NOT add extra optimization passes.
# Total Adam budget = 10,000 steps = same as MLP and Trans-PINN.
# ------------------------------------------------------------------
start_time = time.time()

# Stage 1/3 : Adam warmup with OneCycleLR  (2,000 steps)
# Purpose   : prevent large gradient spikes in the attention layers
#             during the first few hundred steps. Not an extra stage —
#             it is the first 2,000 of the shared 10,000 Adam budget.
print("\nStage 1/3 — Adam warmup, OneCycle (2,000 steps) ...")
opt1   = Adam(model.parameters(), lr=3e-4)
sched1 = torch.optim.lr_scheduler.OneCycleLR(
    opt1, max_lr=1e-3, total_steps=2000,
    pct_start=0.15, anneal_strategy="cos"
)

for step in tqdm(range(2000), desc="Warmup"):
    opt1.zero_grad()
    loss = pinn_loss(res)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt1.step()
    sched1.step()
    if (step + 1) % 500 == 0:
        tqdm.write(f"  [{step+1:5d}] loss = {loss.item():.4e}")

torch.cuda.empty_cache()

# Stage 2/3 : Adam with CosineAnnealingLR  (8,000 steps)
# Purpose   : fine-tune with decaying LR after warmup.
# Steps here = 8,000  →  2,000 + 8,000 = 10,000 total Adam steps.
print("\nStage 2/3 — Adam cosine annealing (8,000 steps) ...")
opt2   = Adam(model.parameters(), lr=5e-4)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt2, T_max=8000, eta_min=1e-6
)

for step in tqdm(range(8000), desc="CosineAdam"):
    opt2.zero_grad()
    loss = pinn_loss(res)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt2.step()
    sched2.step()
    if (step + 1) % 2000 == 0:
        tqdm.write(f"  [{step+1:5d}] loss = {loss.item():.4e}  "
                   f"lr={sched2.get_last_lr()[0]:.1e}")

torch.cuda.empty_cache()

# Stage 3/3 : L-BFGS — max_iter=5000, strong_wolfe (EQUALIZED)
print("\nStage 3/3 — L-BFGS (max_iter=5000) ...")
res_final = new_colloc()
opt3 = LBFGS(
    model.parameters(), max_iter=5000, lr=1.0,
    tolerance_grad=1e-9, tolerance_change=1e-11,
    line_search_fn="strong_wolfe"
)
_n = [0]

def closure():
    opt3.zero_grad()
    loss = pinn_loss(res_final)
    loss.backward()
    _n[0] += 1
    if _n[0] % 1000 == 0:
        print(f"  L-BFGS {_n[0]:4d} | loss = {loss.item():.4e}")
    return loss

opt3.step(closure)
print(f"  L-BFGS done — {_n[0]} evals")
torch.cuda.empty_cache()

total_time = time.time() - start_time
print(f"\nTotal training time : {total_time:.1f} s")

# ------------------------------------------------------------------
# FDM ground truth
# ------------------------------------------------------------------
def get_fdm_truth(nu_val, nx=512, nt=200):
    xg  = np.linspace(-1, 1, nx)
    dxg = xg[1] - xg[0]
    tg  = np.linspace(0, 1, nt)
    u0  = -np.sin(np.pi * xg[1:-1])

    def rhs(u, t_val):
        uf   = np.concatenate(([0.0], u, [0.0]))
        u_xx = (uf[2:] - 2*uf[1:-1] + uf[:-2]) / dxg**2
        u_x  = (uf[2:] - uf[:-2]) / (2 * dxg)
        return -uf[1:-1] * u_x + nu_val * u_xx

    sol = odeint(rhs, u0, tg)
    u   = np.zeros((nt, nx))
    u[:, 1:-1] = sol
    return xg, tg, u

xg, tg, utrue = get_fdm_truth(nu)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
upred = np.zeros_like(utrue)
model.eval()
with torch.no_grad():
    for i in range(len(tg)):
        tt  = torch.full((len(xg), 1), tg[i], device=device)
        xx  = torch.tensor(xg, dtype=torch.float32,
                           device=device).reshape(-1, 1)
        inp = torch.cat([xx, tt], dim=1)
        upred[i] = model(inp, dx).cpu().numpy().flatten()

rl1 = np.sum(np.abs(utrue - upred)) / (np.sum(np.abs(utrue)) + 1e-12)
rl2 = np.sqrt(np.sum((utrue - upred)**2) /
              (np.sum(utrue**2) + 1e-12))

print(f"\n{'='*55}")
print(f"  Fourier Trans-PINN  —  FAIR COMPARISON RESULT")
print(f"  Relative L1 Error : {rl1:.6f}  ({rl1*100:.4f}%)")
print(f"  Relative L2 Error : {rl2:.6f}  ({rl2*100:.4f}%)")
print(f"{'='*55}")

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
kw = dict(aspect='auto', extent=[0, 1, -1, 1], origin='lower', cmap='turbo')

im0 = axes[0].imshow(utrue, **kw)
axes[0].set_title("FDM Ground Truth")
axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(upred, **kw)
axes[1].set_title("Fourier Trans-PINN (Fair)")
axes[1].set_xlabel("t")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(np.abs(utrue - upred),
                     aspect='auto', extent=[0, 1, -1, 1],
                     origin='lower', cmap='inferno')
axes[2].set_title("Absolute Error")
axes[2].set_xlabel("t")
plt.colorbar(im2, ax=axes[2], label="|Error|")

plt.suptitle(f"", fontsize=11)
plt.tight_layout()
plt.savefig("fair_FourierTransPINN_result.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: fair_FourierTransPINN_result.png")
