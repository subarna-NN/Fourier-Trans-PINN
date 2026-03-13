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
# 2.  Memory budget
# ================================================================
N_RES       = 2000   # collocation points
N_RAR_PROBE = 6000   # RAR probe points  (no_grad → safe)
N_RAR_KEEP  = 1200   # high-residual points kept
 
# ================================================================
# 3.  Fourier Embedding
# ================================================================
class FourierEmbedding(nn.Module):
    def __init__(self, in_dim=3, n_freq=32, sigma=5.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * n_freq
 
    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj),
                          torch.cos(proj)], dim=-1)
 
# ================================================================
# 4.  Pseudo-sequence  (3 tokens)
# ================================================================
def pseudo_sequence(xt, dx):
    x = xt[:, 0:1];  y = xt[:, 1:2];  t = xt[:, 2:3]
    return torch.stack([
        torch.cat([x - dx, y, t], dim=1),
        torch.cat([x,      y, t], dim=1),
        torch.cat([x + dx, y, t], dim=1),
    ], dim=1)
 
# ================================================================
# 5.  Transformer Block
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
# 6.  Trans-PINN v6
# ================================================================
class TransPINN(nn.Module):
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
        self.log_w_res  = nn.Parameter(torch.tensor(0.0))
        self.log_w_ic_d = nn.Parameter(torch.tensor(2.3))
        self.log_w_ic_v = nn.Parameter(torch.tensor(2.3))
        self.log_w_bc   = nn.Parameter(torch.tensor(1.6))
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
        return self.mlp(h[:, 1, :])
 
# ================================================================
# 7.  Build model
# ================================================================
model = TransPINN().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters : {n_params:,}")
 
# ================================================================
# 8.  Domain
# ================================================================
Nx, Ny, Nt = 41, 41, 120
x  = torch.linspace(0, 1, Nx)
y  = torch.linspace(0, 1, Ny)
t  = torch.linspace(0, 1, Nt)
dx = x[1] - x[0]
 
# ================================================================
# 9.  IC  (81×81)
# ================================================================
x_ic_g = torch.linspace(0, 1, 81)
y_ic_g = torch.linspace(0, 1, 81)
X_ic, Y_ic = torch.meshgrid(x_ic_g, y_ic_g, indexing="ij")
x_ic      = X_ic.reshape(-1, 1).to(device)
y_ic_d    = Y_ic.reshape(-1, 1).to(device)
t_ic      = torch.zeros_like(x_ic)
ic        = torch.cat([x_ic, y_ic_d, t_ic], dim=1).requires_grad_(True)
u_ic_true = torch.sin(np.pi * x_ic) * torch.sin(np.pi * y_ic_d)
print(f"IC points : {ic.shape[0]:,}")
 
# ================================================================
# 10.  BC  (Nt_bc=120)
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
# 11.  Collocation helpers
# ================================================================
def new_res():
    return torch.rand(N_RES, 3, device=device).requires_grad_(True)
 
res = new_res()
 
# ================================================================
# 12.  PDE residual  (3 correct separate grad calls)
# ================================================================
def pde_residual(model, pts):
    u    = model(pts, dx)
    g1   = torch.autograd.grad(
               u.sum(), pts, create_graph=True)[0]
    u_x  = g1[:, 0:1];  u_y = g1[:, 1:2];  u_t = g1[:, 2:3]
 
    u_xx = torch.autograd.grad(
               u_x.sum(), pts,
               create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(
               u_y.sum(), pts,
               create_graph=True, retain_graph=True)[0][:, 1:2]
    u_tt = torch.autograd.grad(
               u_t.sum(), pts,
               create_graph=True)[0][:, 2:3]
 
    return torch.mean((u_tt - u_xx - u_yy) ** 2)
 
# ================================================================
# 13.  Full PINN loss
# ================================================================
def pinn_loss(model, res_pts):
    loss_res  = pde_residual(model, res_pts)
 
    u_ic_pred = model(ic, dx)
    loss_ic_d = torch.mean((u_ic_pred - u_ic_true) ** 2)
 
    g_ic      = torch.autograd.grad(
                    u_ic_pred.sum(), ic, create_graph=True)[0]
    loss_ic_v = torch.mean(g_ic[:, 2:3] ** 2)
 
    loss_bc   = torch.mean(model(bc, dx) ** 2)
 
    wr  = torch.exp(model.log_w_res)
    wid = torch.exp(model.log_w_ic_d)
    wiv = torch.exp(model.log_w_ic_v)
    wb  = torch.exp(model.log_w_bc)
 
    total = wr*loss_res + wid*loss_ic_d + wiv*loss_ic_v + wb*loss_bc
    return total, loss_res, loss_ic_d, loss_ic_v, loss_bc
 
# ================================================================
# 14.  RAR Resampling
#
#  FIX vs v5:
#  Do NOT use autograd inside RAR at all.
#  Use finite-difference approximation of PDE residual instead.
#  This runs entirely under torch.no_grad() → zero graph memory
#  and zero RuntimeError risk.
#
#  FD residual approximation:
#    u_tt ≈ [u(x,y,t+h) - 2u(x,y,t) + u(x,y,t-h)] / h^2
#    u_xx ≈ [u(x+h,y,t) - 2u(x,y,t) + u(x-h,y,t)] / h^2
#    u_yy ≈ [u(x,y+h,t) - 2u(x,y,t) + u(x,y-h,t)] / h^2
#  This is a cheap, accurate residual proxy for point selection.
# ================================================================
def rar_resample(model):
    model.eval()
    torch.cuda.empty_cache();  gc.collect()
 
    h = 1e-3   # FD step size
 
    with torch.no_grad():
        pts = torch.rand(N_RAR_PROBE, 3, device=device)
        x_  = pts[:, 0:1];  y_ = pts[:, 1:2];  t_ = pts[:, 2:3]
 
        def u(xv, yv, tv):
            inp = torch.cat([xv, yv, tv], dim=1)
            return model(inp, dx)
 
        u0   = u(x_,       y_,       t_      )
        u_xp = u(x_ + h,   y_,       t_      )
        u_xm = u(x_ - h,   y_,       t_      )
        u_yp = u(x_,       y_ + h,   t_      )
        u_ym = u(x_,       y_ - h,   t_      )
        u_tp = u(x_,       y_,       t_ + h  )
        u_tm = u(x_,       y_,       t_ - h  )
 
        u_xx = (u_xp - 2*u0 + u_xm) / h**2
        u_yy = (u_yp - 2*u0 + u_ym) / h**2
        u_tt = (u_tp - 2*u0 + u_tm) / h**2
 
        mag  = (u_tt - u_xx - u_yy).abs().squeeze()
 
    k        = min(N_RAR_KEEP, N_RAR_PROBE)
    top_idx  = torch.topk(mag, k).indices
    refine   = pts[top_idx]
    fill     = torch.rand(max(0, N_RES - k), 3, device=device)
    combined = torch.cat([refine, fill], dim=0)[:N_RES]
 
    model.train()
    torch.cuda.empty_cache()
    return combined.detach().requires_grad_(True)
 
# ================================================================
# 15.  Training
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
    loss, r, id_, iv, bc_ = pinn_loss(model, res)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt1.step();  sched1.step()
    loss_log.append(loss.item())
    if (step + 1) % 400 == 0:
        tqdm.write(
            f"  [{step+1:5d}] total={loss.item():.3e} "
            f"res={r.item():.3e} ic_d={id_.item():.3e} "
            f"ic_v={iv.item():.3e} bc={bc_.item():.3e}")
 
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
    loss, r, id_, iv, bc_ = pinn_loss(model, res)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt2.step();  sched2.step()
    loss_log.append(loss.item())
    if (step + 1) % 1000 == 0:
        tqdm.write(
            f"  [{step+1:5d}] total={loss.item():.3e} "
            f"res={r.item():.3e} ic_d={id_.item():.3e} "
            f"ic_v={iv.item():.3e} bc={bc_.item():.3e} "
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
    loss, r, id_, iv, bc_ = pinn_loss(model, res_final)
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
# 16.  FDM Ground Truth
# ================================================================
print("\nComputing FDM ground truth ...")
xg = x.numpy();  yg = y.numpy();  tg = t.numpy()
dxv = float(dx);  dtv = float(t[1] - t[0])
 
u_fdm = np.zeros((Nt, Nx, Ny))
Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
u_fdm[0] = np.sin(np.pi * Xg) * np.sin(np.pi * Yg)
 
uxx = np.zeros_like(u_fdm[0]);  uyy = np.zeros_like(u_fdm[0])
uxx[1:-1,1:-1] = (u_fdm[0,2:,1:-1] - 2*u_fdm[0,1:-1,1:-1]
                  + u_fdm[0,:-2,1:-1]) / dxv**2
uyy[1:-1,1:-1] = (u_fdm[0,1:-1,2:] - 2*u_fdm[0,1:-1,1:-1]
                  + u_fdm[0,1:-1,:-2]) / dxv**2
u_fdm[1] = u_fdm[0].copy()
u_fdm[1,1:-1,1:-1] += 0.5*dtv**2*(uxx[1:-1,1:-1]+uyy[1:-1,1:-1])
u_fdm[1,[0,-1],:] = 0;  u_fdm[1,:,[0,-1]] = 0
 
for n in range(1, Nt-1):
    uxx = np.zeros_like(u_fdm[n]);  uyy = np.zeros_like(u_fdm[n])
    uxx[1:-1,1:-1] = (u_fdm[n,2:,1:-1] - 2*u_fdm[n,1:-1,1:-1]
                      + u_fdm[n,:-2,1:-1]) / dxv**2
    uyy[1:-1,1:-1] = (u_fdm[n,1:-1,2:] - 2*u_fdm[n,1:-1,1:-1]
                      + u_fdm[n,1:-1,:-2]) / dxv**2
    u_fdm[n+1,1:-1,1:-1] = (2*u_fdm[n,1:-1,1:-1]
                              - u_fdm[n-1,1:-1,1:-1]
                              + dtv**2*(uxx[1:-1,1:-1]+uyy[1:-1,1:-1]))
utrue = u_fdm
print("FDM done.")
 
# ================================================================
# 17.  Evaluation
# ================================================================
upred = np.zeros_like(utrue)
model.eval()
with torch.no_grad():
    for i in range(Nt):
        tt = torch.full((Nx*Ny, 1), tg[i], device=device)
        Xe, Ye = torch.meshgrid(torch.tensor(xg, device=device),
                                 torch.tensor(yg, device=device),
                                 indexing="ij")
        inp = torch.cat([Xe.reshape(-1,1),
                         Ye.reshape(-1,1), tt], dim=1)
        upred[i] = model(inp, dx).cpu().numpy().reshape(Nx, Ny)
 
l2 = np.sqrt(np.sum((utrue-upred)**2) / np.sum(utrue**2))
l1 = np.sum(np.abs(utrue-upred)) / np.sum(np.abs(utrue))
print(f"\n{'='*45}")
print(f"  Relative L2 Error : {l2:.6f}  ({l2*100:.4f}%)")
print(f"  Relative L1 Error : {l1:.6f}  ({l1*100:.4f}%)")
print(f"{'='*45}")
 
# ================================================================
# 18.  Plots
# ================================================================
xp, yp = np.meshgrid(xg, yg, indexing="ij")
 
fig = plt.figure(figsize=(18, 5))
fig.suptitle(f"Trans-PINN v6 | t={tg[-1]:.2f} | "
             f"L2={l2:.2e}  L1={l1:.2e}", fontsize=13)
for col, (data, title, cm) in enumerate([
        (utrue[-1], "FDM Ground Truth", "viridis"),
        (upred[-1], "Trans-PINN v6",    "viridis"),
        (np.abs(utrue[-1]-upred[-1]),   "Absolute Error", "hot")]):
    ax = fig.add_subplot(1, 3, col+1, projection="3d")
    ax.plot_surface(xp, yp, data, cmap=cm)
    ax.set_title(title);  ax.set_xlabel("x");  ax.set_ylabel("y")
plt.tight_layout()
plt.savefig("snapshot_final.png", dpi=150);  plt.show()
 
mid = Nt // 2
fig2, axes = plt.subplots(1, 3, figsize=(16, 4))
fig2.suptitle(f"Mid-time  t={tg[mid]:.2f}", fontsize=12)
for ax, (data, title, cm) in zip(axes, [
        (utrue[mid], "FDM",           "viridis"),
        (upred[mid], "Trans-PINN v6", "viridis"),
        (np.abs(utrue[mid]-upred[mid]), "Abs Error", "hot")]):
    im = ax.contourf(xp, yp, data, 50, cmap=cm)
    ax.set_title(title);  plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("snapshot_mid.png", dpi=150);  plt.show()
 
l2_t = [np.sqrt(np.sum((utrue[i]-upred[i])**2)
                / (np.sum(utrue[i]**2)+1e-12)) for i in range(Nt)]
plt.figure(figsize=(8, 4))
plt.semilogy(tg, l2_t, lw=1.8, color="steelblue")
plt.xlabel("time t");  plt.ylabel("relative L2 error")
plt.title("Trans-PINN v6 — error over time")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("l2_over_time.png", dpi=150);  plt.show()
 
plt.figure(figsize=(8, 4))
plt.semilogy(loss_log, lw=1.2, color="darkorange")
plt.xlabel("step");  plt.ylabel("total loss")
plt.title("Trans-PINN v6 — training loss")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150);  plt.show()
 
print("\nFinal learnable loss weights:")
print(f"  w_res  = {torch.exp(model.log_w_res ).item():.4f}")
print(f"  w_ic_d = {torch.exp(model.log_w_ic_d).item():.4f}")
print(f"  w_ic_v = {torch.exp(model.log_w_ic_v).item():.4f}")
print(f"  w_bc   = {torch.exp(model.log_w_bc  ).item():.4f}")
print("\nAll done.")