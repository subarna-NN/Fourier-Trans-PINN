import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam, LBFGS
from tqdm import tqdm
from scipy.integrate import odeint
warnings.filterwarnings("ignore")

# ================================================================
# 1.  Device
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42);  np.random.seed(42)
torch.cuda.empty_cache();  gc.collect()

free  = torch.cuda.mem_get_info()[0] / 1e9
total = torch.cuda.mem_get_info()[1] / 1e9
print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"VRAM total : {total:.2f} GB")
print(f"VRAM free  : {free:.2f} GB")

# ================================================================
# 2.  Burgers parameter
# ================================================================
nu = 0.01 / np.pi
pi = torch.tensor(np.pi, device=device)
print(f"nu         : {nu:.6f}")

# ================================================================
# 3.  Memory budget
# ================================================================
N_RES       = 3000    # R2: was 2000
N_RAR_PROBE = 6000
N_RAR_KEEP  = 1200

# ================================================================
# 4.  Fourier Embedding
# ================================================================
class FourierEmbedding(nn.Module):
    def __init__(self, in_dim=2, n_freq=32, sigma=5.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * n_freq          # 64

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj),
                          torch.cos(proj)], dim=-1)

# ================================================================
# 5.  Pseudo-sequence  
# ================================================================
def pseudo_sequence(xt, dx):
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    return torch.stack([
        torch.cat([x - dx, t], dim=1),
        torch.cat([x,      t], dim=1),
        torch.cat([x + dx, t], dim=1),
    ], dim=1)                              # [N, 3, 2]

# ================================================================
# 6.  Transformer Block  
# ================================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead,
                                            batch_first=True)
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
        x = x + self.ff(self.norm2(x))
        return x

# ================================================================
# 7.  Fourier Trans-PINN model
# ================================================================
class FourierTransPINN(nn.Module):
    def __init__(self, d_model=64, nhead=4, n_blocks=2,
                 n_freq=32, sigma=5.0):
        super().__init__()
        self.fourier = FourierEmbedding(2, n_freq, sigma)
        self.embed   = nn.Sequential(
            nn.Linear(self.fourier.out_dim, d_model), nn.Tanh())
        self.transformers = nn.ModuleList(
            [TransformerBlock(d_model, nhead) for _ in range(n_blocks)])
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256), nn.Tanh(),
            nn.Linear(256,     256), nn.Tanh(),
            nn.Linear(256,     128), nn.Tanh(),
            nn.Linear(128,       1),
        )
        # 3 learnable log-weights
        self.log_w_res = nn.Parameter(torch.tensor(0.0))
        self.log_w_ic  = nn.Parameter(torch.tensor(2.3))
        self.log_w_bc  = nn.Parameter(torch.tensor(1.6))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, xt, dx):
        seq     = pseudo_sequence(xt, dx)  # [N, 3, 2]
        N, S, _ = seq.shape
        flat    = seq.reshape(N * S, 2)
        feat    = self.fourier(flat)
        h       = self.embed(feat).reshape(N, S, -1)
        for blk in self.transformers:
            h = blk(h)
        return self.mlp(h[:, 1, :])        # centre token [N,1]

# ================================================================
# 8.  Build model + checkpoint trackers
# ================================================================
model = FourierTransPINN().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters : {n_params:,}")

# R1: initialise checkpoint
best_loss  = float('inf')
best_state = copy.deepcopy(model.state_dict())

# ================================================================
# 9.  Domain
# ================================================================
Nx, Nt = 200, 120
x  = torch.linspace(-1, 1, Nx)
t  = torch.linspace( 0, 1, Nt)
dx = x[1] - x[0]

# ================================================================
# 10.  Initial Condition  (512 pts)
# ================================================================
x_ic      = torch.linspace(-1, 1, 512).reshape(-1, 1).to(device)
t_ic      = torch.zeros_like(x_ic)
ic        = torch.cat([x_ic, t_ic], dim=1)
u_ic_true = -torch.sin(pi * x_ic)
print(f"IC points : {ic.shape[0]:,}")

# ================================================================
# 11.  Boundary Conditions  (Nt_bc=300)
# ================================================================
Nt_bc = 300
t_bc  = torch.linspace(0, 1, Nt_bc).reshape(-1, 1).to(device)
x_l   = -torch.ones_like(t_bc)
x_r   =  torch.ones_like(t_bc)
bc_l  = torch.cat([x_l, t_bc], dim=1).to(device)
bc_r  = torch.cat([x_r, t_bc], dim=1).to(device)
print(f"BC points : {bc_l.shape[0] + bc_r.shape[0]:,}  (left + right)")

# ================================================================
# 12.  Collocation 
# ================================================================
def new_res():
    r = torch.rand(N_RES, 2, device=device)
    r[:, 0] = r[:, 0] * 2 - 1   # x: [0,1] -> [-1,1]
    return r.requires_grad_(True)

res = new_res()

# ================================================================
# 13.  PDE Residual 
# ================================================================
def pde_residual(model, pts):
    u    = model(pts, dx)
    g1   = torch.autograd.grad(
               u.sum(), pts, create_graph=True)[0]
    u_x  = g1[:, 0:1]
    u_t  = g1[:, 1:2]
    u_xx = torch.autograd.grad(
               u_x.sum(), pts,
               create_graph=True)[0][:, 0:1]
    f = u_t + u * u_x - nu * u_xx
    return torch.mean(f ** 2)

# ================================================================
# 14.  Full PINN loss 
# ================================================================
def pinn_loss(model, res_pts):
    loss_res = pde_residual(model, res_pts)

    u_ic_pred = model(ic, dx)
    loss_ic   = torch.mean((u_ic_pred - u_ic_true) ** 2)

    loss_bc   = (torch.mean(model(bc_l, dx) ** 2) +
                 torch.mean(model(bc_r, dx) ** 2))

    wr = torch.exp(model.log_w_res)
    wi = torch.exp(model.log_w_ic)
    wb = torch.exp(model.log_w_bc)

    total = wr * loss_res + wi * loss_ic + wb * loss_bc
    return total, loss_res, loss_ic, loss_bc

# ================================================================
# 15.  RAR Resampling 
# ================================================================
def rar_resample(model):
    model.eval()
    torch.cuda.empty_cache();  gc.collect()

    h = 1e-3

    with torch.no_grad():
        pts = torch.rand(N_RAR_PROBE, 2, device=device)
        pts[:, 0] = pts[:, 0] * 2 - 1

        x_ = pts[:, 0:1]
        t_ = pts[:, 1:2]

        def u(xv, tv):
            return model(torch.cat([xv, tv], dim=1), dx)

        u0   = u(x_,   t_  )
        u_xp = u(x_+h, t_  )
        u_xm = u(x_-h, t_  )
        u_tp = u(x_,   t_+h)
        u_tm = u(x_,   t_-h)

        u_xx = (u_xp - 2*u0 + u_xm) / h**2
        u_x  = (u_xp - u_xm) / (2*h)
        u_t  = (u_tp - u_tm) / (2*h)

        mag  = (u_t + u0 * u_x - nu * u_xx).abs().squeeze()

    k       = min(N_RAR_KEEP, N_RAR_PROBE)
    top_idx = torch.topk(mag, k).indices
    refine  = pts[top_idx]

    n_fill = max(0, N_RES - k)
    fill   = torch.rand(n_fill, 2, device=device)
    fill[:, 0] = fill[:, 0] * 2 - 1

    combined = torch.cat([refine, fill], dim=0)[:N_RES]
    model.train()
    torch.cuda.empty_cache()
    return combined.detach().requires_grad_(True)

# ================================================================
# 16.  Training — 3-stage 
# ================================================================
loss_log = []

# ---- Stage 1: Warmup  (2000 steps) ----------------------------
print("\n" + "="*50)
print("STAGE 1/3  Warmup  (2000 steps)")
print("="*50)
torch.cuda.empty_cache()

opt1   = Adam(model.parameters(), lr=3e-4)
sched1 = torch.optim.lr_scheduler.OneCycleLR(
    opt1, max_lr=1e-3, total_steps=2000,
    pct_start=0.15, anneal_strategy="cos")

for step in tqdm(range(2000), desc="Warmup"):
    opt1.zero_grad()
    loss, r, ic_, bc_ = pinn_loss(model, res)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt1.step();  sched1.step()
    lv = loss.item()
    loss_log.append(lv)
    # R1: checkpoint during Stage 1 too
    if lv < best_loss:
        best_loss  = lv
        best_state = copy.deepcopy(model.state_dict())
    if (step + 1) % 400 == 0:
        tqdm.write(
            f"  [{step+1:5d}] total={lv:.3e} "
            f"res={r.item():.3e} ic={ic_.item():.3e} "
            f"bc={bc_.item():.3e}")

torch.cuda.empty_cache()

# ---- Stage 2: CosineAdam + RAR  (15000 steps) -----------------
# R3: 15000 steps (was 10000)
print("\n" + "="*50)
print("STAGE 2/3  CosineAdam + RAR  (15000 steps)")
print("="*50)

opt2   = Adam(model.parameters(), lr=5e-4)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt2, T_max=15000, eta_min=1e-6)

for step in tqdm(range(15000), desc="CosineAdam"):
    if step % 2000 == 0:
        res = rar_resample(model)

    opt2.zero_grad()
    loss, r, ic_, bc_ = pinn_loss(model, res)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt2.step();  sched2.step()

    lv = loss.item()
    loss_log.append(lv)

    # R1: save checkpoint whenever loss improves
    if lv < best_loss:
        best_loss  = lv
        best_state = copy.deepcopy(model.state_dict())

    if (step + 1) % 1000 == 0:
        tqdm.write(
            f"  [{step+1:5d}] total={lv:.3e} "
            f"res={r.item():.3e} ic={ic_.item():.3e} "
            f"bc={bc_.item():.3e} "
            f"lr={sched2.get_last_lr()[0]:.1e} "
            f"best={best_loss:.3e}")

torch.cuda.empty_cache()

# R1: Restore best checkpoint before L-BFGS
print(f"\n  Restoring best model checkpoint — loss = {best_loss:.3e}")
model.load_state_dict(best_state)
print("  Done.")

# ---- Stage 3: L-BFGS  (3000 iters) ----------------------------
# R4: max_iter 2000 -> 3000
print("\n" + "="*50)
print("STAGE 3/3  L-BFGS polish  (3000 iters)")
print("="*50)

res_final = new_res()
opt3 = LBFGS(model.parameters(),
             max_iter=3000, lr=1.0,
             tolerance_grad=1e-9, tolerance_change=1e-11,
             line_search_fn="strong_wolfe")
_n = [0]

def closure():
    opt3.zero_grad()
    loss, r, ic_, bc_ = pinn_loss(model, res_final)
    loss.backward()
    _n[0] += 1
    loss_log.append(loss.item())   # ← ADDED: records every L-BFGS eval
    if _n[0] % 200 == 0:
        print(f"  L-BFGS {_n[0]:4d} | "
              f"total={loss.item():.3e}  res={r.item():.3e}")
    return loss

opt3.step(closure)
print(f"  Done — {_n[0]} evals")
torch.cuda.empty_cache()

# ================================================================
# 17.  FDM Ground Truth
# ================================================================
print("\nComputing FDM ground truth (odeint) ...")

def get_fdm_truth(nu_val, nx=512, nt=200):
    xg  = np.linspace(-1, 1, nx)
    dxg = xg[1] - xg[0]
    tg  = np.linspace(0, 1, nt)
    u0  = -np.sin(np.pi * xg[1:-1])

    def rhs(u, t_val):
        uf   = np.concatenate(([0.0], u, [0.0]))
        u_xx = (uf[2:] - 2*uf[1:-1] + uf[:-2]) / dxg**2
        u_x  = (uf[2:] - uf[:-2]) / (2*dxg)
        return -uf[1:-1] * u_x + nu_val * u_xx

    sol = odeint(rhs, u0, tg)
    u   = np.zeros((nt, nx))
    u[:, 1:-1] = sol
    return xg, tg, u

xg, tg, utrue = get_fdm_truth(nu)
print("FDM done.")

# ================================================================
# 18.  Evaluation
# ================================================================
upred = np.zeros_like(utrue)
model.eval()
with torch.no_grad():
    for i in range(len(tg)):
        tt  = torch.full((len(xg), 1), tg[i], device=device)
        xx  = torch.tensor(xg, device=device,
                           dtype=torch.float32).reshape(-1, 1)
        inp = torch.cat([xx, tt], dim=1)
        upred[i] = model(inp, dx).cpu().numpy().flatten()

l2 = np.sqrt(np.sum((utrue-upred)**2) / (np.sum(utrue**2) + 1e-12))
l1 = np.sum(np.abs(utrue-upred)) / (np.sum(np.abs(utrue)) + 1e-12)

print(f"\n{'='*45}")
print(f"  Relative L2 Error : {l2:.6f}  ({l2*100:.4f}%)")
print(f"  Relative L1 Error : {l1:.6f}  ({l1*100:.4f}%)")
print(f"{'='*45}")

# ================================================================
# 19.  Plots
# ================================================================

# ----------------------------------------------------------------
# A: Space-time heatmaps 
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

kw = dict(aspect='auto', extent=[0, 1, -1, 1],
          origin='lower', cmap='RdBu_r')

im0 = axes[0].imshow(utrue, **kw)
axes[0].set_title("FDM Ground Truth")
axes[0].set_xlabel("t")
axes[0].set_ylabel("x")
axes[0].grid(False)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(upred, **kw)
axes[1].set_title("Fourier Trans-PINN")
axes[1].set_xlabel("t")
axes[1].grid(False)
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(np.abs(utrue - upred),
                     aspect='auto', extent=[0, 1, -1, 1],
                     origin='lower', cmap='Reds')
axes[2].set_title("Absolute Error")
axes[2].set_xlabel("t")
axes[2].grid(False)
plt.colorbar(im2, ax=axes[2], label="|Error|")

plt.tight_layout()
plt.savefig("heatmap_spacetime.png", dpi=600)
plt.show()
print("Saved: heatmap_spacetime.png")

# ----------------------------------------------------------------
# B1: Solution slices (t = 0.25, 0.50)
# ----------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
fig2.suptitle("", fontsize=12)

for ax, frac in zip(axes2, [0.25, 0.50]):
    idx = int(frac * (len(tg) - 1))
    
    ax.plot(xg, utrue[idx], 'k-',  lw=3,   label="FDM")                  # thicker
    ax.plot(xg, upred[idx], 'r--', lw=2.5, label="Fourier Trans-PINN")   # thicker
    
    ax.set_title(f"t = {tg[idx]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(fontsize=8)
    ax.grid(False)

plt.tight_layout()
plt.savefig("solution_slices_part1.png", dpi=600)
plt.show()


# ----------------------------------------------------------------
# B2: Solution slices (t = 0.75, 1.00)
# ----------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
fig3.suptitle("", fontsize=12)

for ax, frac in zip(axes3, [0.75, 1.00]):
    idx = int(frac * (len(tg) - 1))
    
    ax.plot(xg, utrue[idx], 'k-',  lw=3,   label="FDM")                  # thicker
    ax.plot(xg, upred[idx], 'r--', lw=2.5, label="Fourier Trans-PINN")   # thicker
    
    ax.set_title(f"t = {tg[idx]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(fontsize=8)
    ax.grid(False)

plt.tight_layout()
plt.savefig("solution_slices_part2.png", dpi=600)
plt.show()

print("Saved: solution_slices_part1.png & solution_slices_part2.png")
# ----------------------------------------------------------------
# C: L2 error over time
# ----------------------------------------------------------------
l2_t = [np.sqrt(np.sum((utrue[i] - upred[i])**2)
                / (np.sum(utrue[i]**2) + 1e-12)) for i in range(len(tg))]
plt.figure(figsize=(8, 4))
plt.semilogy(tg, l2_t, lw=1.8, color="steelblue")
plt.xlabel("time t")
plt.ylabel("relative L2 error")
plt.title("Fourier Trans-PINN — Burgers error over time")
plt.grid(False)
plt.tight_layout()
plt.savefig("l2_over_time.png", dpi=600)
plt.show()
print("Saved: l2_over_time.png")

# ----------------------------------------------------------------
# D: Training loss curve — full 3-stage 
# ----------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.semilogy(loss_log, lw=1.2, color="darkorange")
plt.axvline(2000,  color='gray', ls='--', lw=1, label="Stage 1 end")
plt.axvline(17000, color='gray', ls=':',  lw=1, label="Stage 2 end")
plt.xlabel("step")
plt.ylabel("total loss")
plt.title("")
plt.legend(fontsize=9)
plt.grid(False)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=600)
plt.show()
print("Saved: loss_curve.png")
# ----------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------
print("\nFinal learnable loss weights:")
print(f"  w_res = {torch.exp(model.log_w_res).item():.4f}")
print(f"  w_ic  = {torch.exp(model.log_w_ic ).item():.4f}")
print(f"  w_bc  = {torch.exp(model.log_w_bc ).item():.4f}")
print(f"\nBest Stage 2 loss (checkpoint) : {best_loss:.3e}")
print("\nAll done.")
