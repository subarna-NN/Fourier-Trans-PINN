import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
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
# 2.  Allen-Cahn parameter
#     CHANGED vs Wave: eps constant added
# ================================================================
eps = 0.01
pi  = torch.tensor(np.pi, device=device)
print(f"epsilon    : {eps}")

# ================================================================
# 3.  Memory budget  (identical to Fourier Trans-PINN Wave)
# ================================================================
N_RES       = 2000
N_RAR_PROBE = 6000
N_RAR_KEEP  = 1200

# ================================================================
# 4.  Fourier Embedding  (IDENTICAL)
# ================================================================
class FourierEmbedding(nn.Module):
    def __init__(self, in_dim=3, n_freq=32, sigma=5.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * n_freq          # 64

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj),
                          torch.cos(proj)], dim=-1)

# ================================================================
# 5.  Pseudo-sequence  (IDENTICAL)
# ================================================================
def pseudo_sequence(xt, dx):
    x = xt[:, 0:1];  y = xt[:, 1:2];  t = xt[:, 2:3]
    return torch.stack([
        torch.cat([x - dx, y, t], dim=1),
        torch.cat([x,      y, t], dim=1),
        torch.cat([x + dx, y, t], dim=1),
    ], dim=1)

# ================================================================
# 6.  Transformer Block  (IDENTICAL)
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
#     CHANGED vs Wave: log_w_ic_v removed (no velocity IC)
#                      3 loss weights instead of 4
# ================================================================
class FourierTransPINN(nn.Module):
    def __init__(self, d_model=64, nhead=4, n_blocks=2,
                 n_freq=32, sigma=5.0):
        super().__init__()
        self.fourier = FourierEmbedding(3, n_freq, sigma)
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
        # 3 learnable log-weights (Allen-Cahn has no velocity IC)
        self.log_w_res = nn.Parameter(torch.tensor(0.0))
        self.log_w_ic  = nn.Parameter(torch.tensor(2.3))   # ~10
        self.log_w_bc  = nn.Parameter(torch.tensor(1.6))   # ~5

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, xt, dx):
        seq     = pseudo_sequence(xt, dx)
        N, S, _ = seq.shape
        flat    = seq.reshape(N * S, 3)
        feat    = self.fourier(flat)
        h       = self.embed(feat).reshape(N, S, -1)
        for blk in self.transformers:
            h = blk(h)
        return self.mlp(h[:, 1, :])   # centre token

# ================================================================
# 8.  Build model
# ================================================================
model = FourierTransPINN().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters : {n_params:,}")

# ================================================================
# 9.  Domain
#     CHANGED vs Wave: Nx=Ny=41, Nt=60
#     (odeint is adaptive so Nt just sets evaluation points)
# ================================================================
Nx, Ny, Nt = 41, 41, 60
x = torch.linspace(0, 1, Nx)
y = torch.linspace(0, 1, Ny)
t = torch.linspace(0, 1, Nt)
dx = x[1] - x[0]

# ================================================================
# 10.  Initial Conditions  (81x81)
#      CHANGED vs Wave: amplitude 0.05, no velocity IC
# ================================================================
x_ic_g = torch.linspace(0, 1, 81)
y_ic_g = torch.linspace(0, 1, 81)
X_ic, Y_ic = torch.meshgrid(x_ic_g, y_ic_g, indexing="ij")
x_ic     = X_ic.reshape(-1, 1).to(device)
y_ic_d   = Y_ic.reshape(-1, 1).to(device)
t_ic     = torch.zeros_like(x_ic)
ic       = torch.cat([x_ic, y_ic_d, t_ic], dim=1).requires_grad_(True)
# Allen-Cahn IC: 0.05 * sin(pi*x) * sin(pi*y)
u_ic_true = 0.05 * torch.sin(pi * x_ic) * torch.sin(pi * y_ic_d)
print(f"IC points : {ic.shape[0]:,}")

# ================================================================
# 11.  Boundary Conditions  (Nt_bc=120, identical structure)
# ================================================================
Nt_bc = 120
t_bc  = torch.linspace(0, 1, Nt_bc).to(device)

def make_bc():
    pts = []
    yv = torch.linspace(0, 1, Ny).to(device)
    xv = torch.linspace(0, 1, Nx).to(device)
    Yb, Tb = torch.meshgrid(yv, t_bc, indexing="ij")
    X0 = torch.zeros_like(Yb);  X1 = torch.ones_like(Yb)
    pts += [torch.cat([X0.reshape(-1,1),Yb.reshape(-1,1),Tb.reshape(-1,1)],1),
            torch.cat([X1.reshape(-1,1),Yb.reshape(-1,1),Tb.reshape(-1,1)],1)]
    Xb, Tb = torch.meshgrid(xv, t_bc, indexing="ij")
    Y0 = torch.zeros_like(Xb);  Y1 = torch.ones_like(Xb)
    pts += [torch.cat([Xb.reshape(-1,1),Y0.reshape(-1,1),Tb.reshape(-1,1)],1),
            torch.cat([Xb.reshape(-1,1),Y1.reshape(-1,1),Tb.reshape(-1,1)],1)]
    return torch.cat(pts, 0)

bc = make_bc().to(device)
print(f"BC points : {bc.shape[0]:,}")

# ================================================================
# 12.  Collocation helpers  (identical)
# ================================================================
def new_res():
    return torch.rand(N_RES, 3, device=device).requires_grad_(True)

res = new_res()

# ================================================================
# 13.  PDE Residual  — Allen-Cahn specific
#
#  CHANGED vs Wave:
#  Wave:       3 grad calls (u_xx, u_yy, u_tt)
#  Allen-Cahn: 2 grad calls (u_xx, u_yy) only
#              u_t comes from first-order g1 (no second call)
#              Nonlinear term u^3 - u uses forward-pass output
#
#  Allen-Cahn PDE:
#    f = u_t - eps*(u_xx + u_yy) + u^3 - u = 0
# ================================================================
def pde_residual(model, pts):
    u   = model(pts, dx)                              # (N, 1)

    # First derivatives (u_x, u_y, u_t all from one call)
    g1  = torch.autograd.grad(
              u.sum(), pts, create_graph=True)[0]     # (N, 3)
    u_x = g1[:, 0:1]
    u_y = g1[:, 1:2]
    u_t = g1[:, 2:3]   # used directly in Allen-Cahn

    # Second derivatives — 2 calls (not 3 like Wave)
    # u_xx: retain graph so u_yy can still be computed
    u_xx = torch.autograd.grad(
               u_x.sum(), pts,
               create_graph=True,
               retain_graph=True)[0][:, 0:1]          # (N, 1)

    # u_yy: last second-deriv call, no retain needed
    u_yy = torch.autograd.grad(
               u_y.sum(), pts,
               create_graph=True)[0][:, 1:2]          # (N, 1)

    # Allen-Cahn residual: u_t - eps*(u_xx+u_yy) + u^3 - u
    f = u_t - eps * (u_xx + u_yy) + u**3 - u
    return torch.mean(f ** 2)

# ================================================================
# 14.  Full PINN loss — 3 terms (no velocity IC)
#
#  CHANGED vs Wave:
#  Wave:       4 terms (res, ic_disp, ic_vel, bc)
#  Allen-Cahn: 3 terms (res, ic, bc)
# ================================================================
def pinn_loss(model, res_pts):
    loss_res = pde_residual(model, res_pts)

    # IC displacement only (no velocity IC for Allen-Cahn)
    u_ic_pred = model(ic, dx)
    loss_ic   = torch.mean((u_ic_pred - u_ic_true) ** 2)

    # BC Dirichlet zero
    loss_bc   = torch.mean(model(bc, dx) ** 2)

    # Self-adaptive weights
    wr = torch.exp(model.log_w_res)
    wi = torch.exp(model.log_w_ic)
    wb = torch.exp(model.log_w_bc)

    total = wr * loss_res + wi * loss_ic + wb * loss_bc
    return total, loss_res, loss_ic, loss_bc

# ================================================================
# 15.  RAR Resampling  (finite-difference, no_grad — identical)
#      CHANGED: FD residual uses Allen-Cahn formula
# ================================================================
def rar_resample(model):
    model.eval()
    torch.cuda.empty_cache();  gc.collect()

    h = 1e-3

    with torch.no_grad():
        pts = torch.rand(N_RAR_PROBE, 3, device=device)
        x_  = pts[:, 0:1];  y_ = pts[:, 1:2];  t_ = pts[:, 2:3]

        def u(xv, yv, tv):
            return model(torch.cat([xv, yv, tv], dim=1), dx)

        u0   = u(x_, y_, t_)
        u_xp = u(x_+h, y_,   t_);  u_xm = u(x_-h, y_,   t_)
        u_yp = u(x_,   y_+h, t_);  u_ym = u(x_,   y_-h, t_)
        u_tp = u(x_,   y_,   t_+h)
        u_tm = u(x_,   y_,   t_-h)

        u_xx = (u_xp - 2*u0 + u_xm) / h**2
        u_yy = (u_yp - 2*u0 + u_ym) / h**2
        # u_t: central difference in time
        u_t  = (u_tp - u_tm) / (2*h)

        # Allen-Cahn residual for point selection
        mag = (u_t - eps*(u_xx+u_yy) + u0**3 - u0).abs().squeeze()

    k        = min(N_RAR_KEEP, N_RAR_PROBE)
    top_idx  = torch.topk(mag, k).indices
    refine   = pts[top_idx]
    fill     = torch.rand(max(0, N_RES - k), 3, device=device)
    combined = torch.cat([refine, fill], dim=0)[:N_RES]

    model.train()
    torch.cuda.empty_cache()
    return combined.detach().requires_grad_(True)

# ================================================================
# 16.  Training  — 3-stage pipeline  (identical structure)
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
    loss_log.append(loss.item())
    if (step + 1) % 400 == 0:
        tqdm.write(
            f"  [{step+1:5d}] total={loss.item():.3e} "
            f"res={r.item():.3e} ic={ic_.item():.3e} "
            f"bc={bc_.item():.3e}")

torch.cuda.empty_cache()

# ---- Stage 2: CosineAdam + RAR  (10000 steps) -----------------
print("\n" + "="*50)
print("STAGE 2/3  CosineAdam + RAR  (10000 steps)")
print("="*50)

opt2   = Adam(model.parameters(), lr=5e-4)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt2, T_max=10000, eta_min=1e-6)

for step in tqdm(range(10000), desc="CosineAdam"):
    if step % 2000 == 0:
        res = rar_resample(model)

    opt2.zero_grad()
    loss, r, ic_, bc_ = pinn_loss(model, res)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt2.step();  sched2.step()
    loss_log.append(loss.item())
    if (step + 1) % 1000 == 0:
        tqdm.write(
            f"  [{step+1:5d}] total={loss.item():.3e} "
            f"res={r.item():.3e} ic={ic_.item():.3e} "
            f"bc={bc_.item():.3e} "
            f"lr={sched2.get_last_lr()[0]:.1e}")

torch.cuda.empty_cache()

# ---- Stage 3: L-BFGS  (2000 iters) ----------------------------
print("\n" + "="*50)
print("STAGE 3/3  L-BFGS polish  (2000 iters)")
print("="*50)

res_final = new_res()
opt3 = LBFGS(model.parameters(),
             max_iter=2000, lr=1.0,
             tolerance_grad=1e-9, tolerance_change=1e-11,
             line_search_fn="strong_wolfe")
_n = [0]

def closure():
    opt3.zero_grad()
    loss, r, ic_, bc_ = pinn_loss(model, res_final)
    loss.backward()
    _n[0] += 1
    if _n[0] % 200 == 0:
        print(f"  L-BFGS {_n[0]:4d} | "
              f"total={loss.item():.3e}  res={r.item():.3e}")
    return loss

opt3.step(closure)
print(f"  Done — {_n[0]} evals")
torch.cuda.empty_cache()

# ================================================================
# 17.  FDM Ground Truth — Allen-Cahn via scipy.odeint
#      CHANGED vs Wave: odeint (your original correct method)
#      Resolution raised: Nx=Ny=41, Nt=60
# ================================================================
print("\nComputing FDM ground truth (odeint) ...")

xg = x.numpy();  yg = y.numpy();  tg = t.numpy()
dxv = float(dx);  dyv = dxv   # dx = dy for square grid

def allen_cahn_rhs(u_flat, t_val, eps_val, Nxx, Nyy, dxx, dyy):
    u = u_flat.reshape(Nxx, Nyy)
    # enforce BC
    u[0,:]=0; u[-1,:]=0; u[:,0]=0; u[:,-1]=0

    u_xx = (np.roll(u,-1,axis=0) - 2*u + np.roll(u,1,axis=0)) / dxx**2
    u_yy = (np.roll(u,-1,axis=1) - 2*u + np.roll(u,1,axis=1)) / dyy**2

    # zero second derivatives at boundaries
    u_xx[0,:]=0; u_xx[-1,:]=0; u_xx[:,0]=0; u_xx[:,-1]=0
    u_yy[0,:]=0; u_yy[-1,:]=0; u_yy[:,0]=0; u_yy[:,-1]=0

    du = eps_val*(u_xx + u_yy) - u**3 + u
    du[0,:]=0; du[-1,:]=0; du[:,0]=0; du[:,-1]=0
    return du.reshape(-1)

Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
u0     = 0.05 * np.sin(np.pi*Xg) * np.sin(np.pi*Yg)
u0[0,:]=0; u0[-1,:]=0; u0[:,0]=0; u0[:,-1]=0

utrue_flat = odeint(allen_cahn_rhs, u0.reshape(-1), tg,
                    args=(eps, Nx, Ny, dxv, dyv))
utrue = utrue_flat.reshape(Nt, Nx, Ny)
print("FDM done.")

# ================================================================
# 18.  Evaluation  (identical to Wave version)
# ================================================================
upred = np.zeros_like(utrue)
model.eval()
with torch.no_grad():
    for i in range(Nt):
        tt = torch.full((Nx*Ny, 1), tg[i], device=device)
        Xe, Ye = torch.meshgrid(
            torch.tensor(xg, device=device),
            torch.tensor(yg, device=device), indexing="ij")
        inp = torch.cat([Xe.reshape(-1,1),
                         Ye.reshape(-1,1), tt], dim=1)
        upred[i] = model(inp, dx).cpu().numpy().reshape(Nx, Ny)

l2 = np.sqrt(np.sum((utrue-upred)**2) / (np.sum(utrue**2) + 1e-12))
l1 = np.sum(np.abs(utrue-upred)) / (np.sum(np.abs(utrue)) + 1e-12)

print(f"\n{'='*45}")
print(f"  Relative L2 Error : {l2:.6f}  ({l2*100:.4f}%)")
print(f"  Relative L1 Error : {l1:.6f}  ({l1*100:.4f}%)")
print(f"{'='*45}")

# ================================================================
# 19.  Plots  (identical 4-plot system)
# ================================================================
xp, yp = np.meshgrid(xg, yg, indexing="ij")

# A: Final-time 3D surface
fig = plt.figure(figsize=(18, 5))
fig.suptitle(f"Fourier Trans-PINN | 2D Allen-Cahn | t={tg[-1]:.2f} | "
             f"L2={l2:.2e}  L1={l1:.2e}", fontsize=13)
for col, (data, title, cm) in enumerate([
        (utrue[-1], "FDM Ground Truth", "viridis"),
        (upred[-1], "Fourier Trans-PINN", "viridis"),
        (np.abs(utrue[-1]-upred[-1]), "Absolute Error", "hot")]):
    ax = fig.add_subplot(1, 3, col+1, projection="3d")
    ax.plot_surface(xp, yp, data, cmap=cm)
    ax.set_title(title);  ax.set_xlabel("x");  ax.set_ylabel("y")
plt.tight_layout()
plt.savefig("snapshot_final.png", dpi=150);  plt.show()
print("Saved: snapshot_final.png")

# B: Mid-time contour
mid = Nt // 2
fig2, axes = plt.subplots(1, 3, figsize=(16, 4))
fig2.suptitle(f"Mid-time  t={tg[mid]:.2f}", fontsize=12)
for ax, (data, title, cm) in zip(axes, [
        (utrue[mid], "FDM", "viridis"),
        (upred[mid], "Fourier Trans-PINN", "viridis"),
        (np.abs(utrue[mid]-upred[mid]), "Abs Error", "hot")]):
    im = ax.contourf(xp, yp, data, 50, cmap=cm)
    ax.set_title(title);  plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("snapshot_mid.png", dpi=150);  plt.show()
print("Saved: snapshot_mid.png")

# C: L2 over time
l2_t = [np.sqrt(np.sum((utrue[i]-upred[i])**2)
                / (np.sum(utrue[i]**2)+1e-12)) for i in range(Nt)]
plt.figure(figsize=(8, 4))
plt.semilogy(tg, l2_t, lw=1.8, color="steelblue")
plt.xlabel("time t");  plt.ylabel("relative L2 error")
plt.title("Fourier Trans-PINN — Allen-Cahn error over time")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("l2_over_time.png", dpi=150);  plt.show()
print("Saved: l2_over_time.png")

# D: Loss curve
plt.figure(figsize=(8, 4))
plt.semilogy(loss_log, lw=1.2, color="darkorange")
plt.xlabel("step");  plt.ylabel("total loss")
plt.title("Fourier Trans-PINN — Allen-Cahn training loss")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150);  plt.show()
print("Saved: loss_curve.png")

# Final weight summary
print("\nFinal learnable loss weights:")
print(f"  w_res = {torch.exp(model.log_w_res).item():.4f}")
print(f"  w_ic  = {torch.exp(model.log_w_ic ).item():.4f}")
print(f"  w_bc  = {torch.exp(model.log_w_bc ).item():.4f}")
print("\nAll done.") 