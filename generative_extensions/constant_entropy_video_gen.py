"""
Entropy-Conserving Video Generation — Hypothesis Test
======================================================

Core hypothesis: Consecutive video frames have near-constant entropy.
A model can learn to transform frame t → frame t+1 by reorganizing
structure on a constant-entropy manifold, without creating or destroying
information.

This script:
  1. Generates a synthetic video dataset (moving shapes)
  2. Measures entropy across consecutive frames to validate the hypothesis
  3. Trains an entropy-conserving flow model for next-frame prediction
  4. Compares entropy stability in autoregressive rollout
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG — edit these directly
# ============================================================
IMG_SIZE     = 64
BATCH_SIZE   = 32
LR           = 2e-4
EPOCHS       = 100
MODEL_DIM    = 64
NUM_VIDEOS   = 500       # number of synthetic video sequences
SEQ_LEN      = 20        # frames per sequence
LAMBDA_ENT   = 1.0       # weight for entropy conservation loss
SAVE_DIR     = Path("./outputs_video")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name()}")


# ============================================================
# 1. ENTROPY MEASURES
# ============================================================
def spectral_entropy(x: torch.Tensor) -> torch.Tensor:
    """
    Spectral entropy per image:
    H = -Σ P(k) log P(k)  where  P(k) = |û(k)|² / Σ|û(k)|²
    Returns shape [B].
    """
    fft = torch.fft.fft2(x)
    power = (fft.real ** 2 + fft.imag ** 2).sum(dim=1)   # [B, H, W]
    power_flat = power.reshape(power.shape[0], -1)         # [B, H*W]
    total = power_flat.sum(dim=1, keepdim=True).clamp(min=1e-10)
    p = power_flat / total
    log_p = torch.log(p.clamp(min=1e-10))
    return -(p * log_p).sum(dim=1)


def pixel_histogram_entropy(x: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """Pixel intensity histogram entropy per image. Returns shape [B]."""
    gray = x.mean(dim=1)
    entropies = []
    for i in range(gray.shape[0]):
        hist = torch.histc(gray[i], bins=bins, min=0.0, max=1.0)
        hist = hist / hist.sum().clamp(min=1e-10)
        H = -(hist * torch.log(hist.clamp(min=1e-10))).sum()
        entropies.append(H)
    return torch.stack(entropies)


def spatial_variance_entropy(x: torch.Tensor) -> torch.Tensor:
    """Spatial variance as entropy proxy. Returns shape [B]."""
    return x.var(dim=(1, 2, 3))


# ============================================================
# 2. SYNTHETIC VIDEO DATASET
# ============================================================
class SyntheticVideoDataset(Dataset):
    """
    Video sequences of shapes moving smoothly.
    Returns consecutive frame pairs (frame_t, frame_t+1).
    """

    def __init__(self, num_videos=NUM_VIDEOS, seq_len=SEQ_LEN, img_size=IMG_SIZE):
        self.img_size = img_size
        self.seq_len = seq_len
        self.pairs = []
        print(f"Generating {num_videos} synthetic videos x {seq_len} frames...")

        for vid in range(num_videos):
            rng = np.random.RandomState(vid)
            frames = self._generate_video(rng, seq_len)
            for t in range(len(frames) - 1):
                self.pairs.append((frames[t], frames[t + 1]))

        print(f"Total frame pairs: {len(self.pairs)}")

    def _generate_video(self, rng, seq_len):
        S = self.img_size
        num_shapes = rng.randint(2, 5)

        bg = rng.rand(3).astype(np.float32) * 0.3 + 0.1
        bg_drift = (rng.rand(3).astype(np.float32) - 0.5) * 0.005

        shapes = []
        for _ in range(num_shapes):
            shapes.append({
                "type": rng.choice(["circle", "rect", "triangle"]),
                "cx": rng.uniform(S * 0.2, S * 0.8),
                "cy": rng.uniform(S * 0.2, S * 0.8),
                "vx": rng.uniform(-1.5, 1.5),
                "vy": rng.uniform(-1.5, 1.5),
                "size": rng.uniform(S * 0.08, S * 0.25),
                "color": rng.rand(3).astype(np.float32) * 0.6 + 0.3,
                "color_drift": (rng.rand(3).astype(np.float32) - 0.5) * 0.003,
            })

        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        frames = []

        for t in range(seq_len):
            img = np.zeros((3, S, S), dtype=np.float32)
            current_bg = np.clip(bg + bg_drift * t, 0, 1)
            for c in range(3):
                img[c] = current_bg[c]

            for sh in shapes:
                cx = sh["cx"] + sh["vx"] * t
                cy = sh["cy"] + sh["vy"] * t
                cx = cx % (2 * S)
                cy = cy % (2 * S)
                if cx > S: cx = 2 * S - cx
                if cy > S: cy = 2 * S - cy

                color = np.clip(sh["color"] + sh["color_drift"] * t, 0, 1)
                size = sh["size"]

                if sh["type"] == "circle":
                    mask = ((xx - cx)**2 + (yy - cy)**2) < size**2
                elif sh["type"] == "rect":
                    mask = (np.abs(xx - cx) < size) & (np.abs(yy - cy) < size * 0.7)
                else:
                    mask = ((yy - cy + size) > 0) & \
                           (np.abs(xx - cx) < (yy - cy + size) * 0.6) & \
                           ((yy - cy) < size)

                for c in range(3):
                    img[c][mask] = color[c]

            img = np.clip(img, 0, 1)
            frames.append(torch.from_numpy(img))

        return frames

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ============================================================
# 3. FLOW PREDICTION NETWORK
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(32, ch), ch), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class FlowNet(nn.Module):
    """
    Given frame u_t, predict flow v such that u_{t+1} = u_t + v(u_t).
    Output layer initialized near zero so initial prediction = identity.
    """

    def __init__(self, in_ch=3, base=MODEL_DIM):
        super().__init__()
        self.enc_in = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc1 = ResBlock(base)
        self.down1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.enc2 = ResBlock(base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)
        self.enc3 = ResBlock(base * 4)

        self.mid = ResBlock(base * 4)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
        )
        self.dec2 = ResBlock(base * 2)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 2, base, 3, padding=1),
        )
        self.dec1 = ResBlock(base)

        self.skip2 = nn.Conv2d(base * 4, base * 2, 1)
        self.skip1 = nn.Conv2d(base * 2, base, 1)

        self.out = nn.Sequential(
            nn.GroupNorm(min(32, base), base), nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x):
        e0 = self.enc_in(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        m = self.mid(e3)

        d2 = self.up2(m)
        d2 = self.dec2(self.skip2(torch.cat([d2, e2], 1)))
        d1 = self.up1(d2)
        d1 = self.dec1(self.skip1(torch.cat([d1, e1], 1)))

        return self.out(d1)


# ============================================================
# 4. DATASET
# ============================================================
dataset = SyntheticVideoDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=0, drop_last=True,
                    pin_memory=(DEVICE.type == "cuda"))


# ============================================================
# 5. HYPOTHESIS TEST — Entropy across consecutive frames
# ============================================================
print("\n" + "=" * 65)
print("HYPOTHESIS TEST: Entropy conservation in consecutive frames")
print("=" * 65)

all_spectral_diffs = []
all_histogram_diffs = []
all_variance_diffs = []

test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
for batch_idx, (f_t, f_tp1) in enumerate(test_loader):
    if batch_idx >= 20:
        break
    f_t, f_tp1 = f_t.to(DEVICE), f_tp1.to(DEVICE)

    h_t = spectral_entropy(f_t)
    h_tp1 = spectral_entropy(f_tp1)
    all_spectral_diffs.append(((h_tp1 - h_t) / h_t.clamp(min=1e-6)).abs().cpu())

    hh_t = pixel_histogram_entropy(f_t)
    hh_tp1 = pixel_histogram_entropy(f_tp1)
    all_histogram_diffs.append(((hh_tp1 - hh_t) / hh_t.clamp(min=1e-6)).abs().cpu())

    v_t = spatial_variance_entropy(f_t)
    v_tp1 = spatial_variance_entropy(f_tp1)
    all_variance_diffs.append(((v_tp1 - v_t) / v_t.clamp(min=1e-6)).abs().cpu())

spectral_diffs = torch.cat(all_spectral_diffs)
histogram_diffs = torch.cat(all_histogram_diffs)
variance_diffs = torch.cat(all_variance_diffs)

print(f"\nRelative entropy change between consecutive frames:")
print(f"  Spectral entropy:  mean={spectral_diffs.mean():.6f}  "
      f"std={spectral_diffs.std():.6f}  max={spectral_diffs.max():.6f}")
print(f"  Histogram entropy: mean={histogram_diffs.mean():.6f}  "
      f"std={histogram_diffs.std():.6f}  max={histogram_diffs.max():.6f}")
print(f"  Spatial variance:  mean={variance_diffs.mean():.6f}  "
      f"std={variance_diffs.std():.6f}  max={variance_diffs.max():.6f}")
print(f"\n  -> Values close to 0 confirm the hypothesis!")

# Entropy over time for a single video
single_video_frames = []
for t in range(min(SEQ_LEN - 1, 19)):
    f_t, _ = dataset[t]
    single_video_frames.append(f_t)
single_video_frames.append(dataset[min(SEQ_LEN - 2, 18)][1])

sv_tensor = torch.stack(single_video_frames).to(DEVICE)
sv_entropy = spectral_entropy(sv_tensor).cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sv_entropy, "b-o", markersize=4, linewidth=2)
ax.set(xlabel="Frame", ylabel="Spectral Entropy",
       title="Spectral Entropy Across Consecutive Video Frames")
ax.grid(True, alpha=0.3)
rel_range = (sv_entropy.max() - sv_entropy.min()) / sv_entropy.mean() * 100
ax.text(0.02, 0.98, f"Relative range: {rel_range:.2f}%",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
fig.savefig(SAVE_DIR / "entropy_over_frames.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved entropy-over-frames plot")


# ============================================================
# 6. TRAINING
# ============================================================
print("\n" + "=" * 65)
print("TRAINING: Entropy-Conserving Flow Model")
print("=" * 65)

model = FlowNet().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {nparam:,}")

losses_recon = []
losses_entropy = []
losses_total = []

for epoch in range(EPOCHS):
    model.train()
    ep_recon, ep_ent, ep_total = 0, 0, 0

    for f_t, f_tp1 in loader:
        f_t, f_tp1 = f_t.to(DEVICE), f_tp1.to(DEVICE)

        flow = model(f_t)
        pred = (f_t + flow).clamp(0, 1)

        loss_recon = F.mse_loss(pred, f_tp1)

        H_input = spectral_entropy(f_t)
        H_pred = spectral_entropy(pred)
        loss_entropy = F.mse_loss(H_pred, H_input)

        loss = loss_recon + LAMBDA_ENT * loss_entropy

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        ep_recon += loss_recon.item()
        ep_ent += loss_entropy.item()
        ep_total += loss.item()

    sched.step()
    n = len(loader)
    losses_recon.append(ep_recon / n)
    losses_entropy.append(ep_ent / n)
    losses_total.append(ep_total / n)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
              f"recon {ep_recon/n:.6f} | "
              f"entropy {ep_ent/n:.6f} | "
              f"total {ep_total/n:.6f}")

    # ---- periodic visualization ----
    if (epoch + 1) % 20 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            # Multi-step autoregressive rollout
            start_frame = dataset[0][0].unsqueeze(0).to(DEVICE)
            rollout = [start_frame]
            u = start_frame
            for step in range(min(SEQ_LEN - 1, 15)):
                flow = model(u)
                u = (u + flow).clamp(0, 1)
                rollout.append(u)

            # Ground truth
            gt_frames = [dataset[0][0].unsqueeze(0)]
            for t in range(min(SEQ_LEN - 2, 14)):
                gt_frames.append(dataset[t][1].unsqueeze(0))
            gt_frames.append(dataset[min(SEQ_LEN - 2, 14)][1].unsqueeze(0))

            rollout_tensor = torch.cat(rollout[:16], dim=0)
            gt_tensor = torch.cat(gt_frames[:16], dim=0).to(DEVICE)

            comparison = torch.cat([gt_tensor, rollout_tensor], dim=0)
            grid = vutils.make_grid(comparison, nrow=min(16, len(rollout)), padding=2)
            fig, ax = plt.subplots(figsize=(20, 4))
            ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
            ax.axis("off")
            ax.set_title(f"Top: ground truth | Bottom: predicted rollout — "
                         f"epoch {epoch+1}")
            fig.savefig(SAVE_DIR / f"rollout_epoch_{epoch+1:04d}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Entropy comparison
            rollout_entropy = spectral_entropy(rollout_tensor).cpu().numpy()
            gt_entropy = spectral_entropy(gt_tensor).cpu().numpy()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(gt_entropy, "b-o", markersize=4, label="Ground truth", lw=2)
            ax.plot(rollout_entropy, "r-s", markersize=4, label="Predicted", lw=2)
            ax.set(xlabel="Frame", ylabel="Spectral Entropy",
                   title=f"Entropy: GT vs Predicted — epoch {epoch+1}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(SAVE_DIR / f"entropy_comparison_{epoch+1:04d}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  -> saved visualizations")


# ============================================================
# 7. FINAL ANALYSIS
# ============================================================
print("\n" + "=" * 65)
print("FINAL ANALYSIS")
print("=" * 65)

# Loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(losses_recon, "b-", label="Reconstruction", lw=1.5)
ax1.plot(losses_total, "k-", label="Total", lw=1.5, alpha=0.5)
ax1.set(xlabel="Epoch", ylabel="Loss", title="Training Losses")
ax1.set_yscale("log")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(losses_entropy, "r-", lw=1.5)
ax2.set(xlabel="Epoch", ylabel="Entropy Loss",
        title="Entropy Conservation Loss")
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3)
fig.savefig(SAVE_DIR / "loss_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Long autoregressive rollout
model.eval()
with torch.no_grad():
    u = dataset[0][0].unsqueeze(0).to(DEVICE)
    long_rollout = [u]
    entropies = [spectral_entropy(u).item()]

    for step in range(min(SEQ_LEN * 2, 40)):
        flow = model(u)
        u = (u + flow).clamp(0, 1)
        long_rollout.append(u)
        entropies.append(spectral_entropy(u).item())

    # Show sampled frames
    show_indices = list(range(0, len(long_rollout), max(1, len(long_rollout) // 16)))
    show_frames = torch.cat([long_rollout[i] for i in show_indices[:16]])
    grid = vutils.make_grid(show_frames, nrow=8, padding=2)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
    ax.set_title("Long autoregressive rollout (sampled frames)")
    fig.savefig(SAVE_DIR / "long_rollout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Entropy stability
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(entropies, "g-", lw=2)
    ax.axhline(entropies[0], color="k", ls="--", alpha=0.3, label="Initial entropy")
    ax.set(xlabel="Rollout step", ylabel="Spectral Entropy",
           title="Entropy Stability Over Long Autoregressive Rollout")
    ax.legend()
    ax.grid(True, alpha=0.3)
    drift = abs(entropies[-1] - entropies[0]) / entropies[0] * 100
    ax.text(0.02, 0.98, f"Entropy drift: {drift:.2f}%",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.savefig(SAVE_DIR / "entropy_long_rollout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nLong rollout entropy drift: {drift:.2f}%")
    print(f"Initial entropy: {entropies[0]:.4f}")
    print(f"Final entropy:   {entropies[-1]:.4f}")

# Save model
torch.save({
    "model_state_dict": model.state_dict(),
}, SAVE_DIR / "model_final.pt")

print(f"\nAll outputs saved to {SAVE_DIR}/")
print("Done!")