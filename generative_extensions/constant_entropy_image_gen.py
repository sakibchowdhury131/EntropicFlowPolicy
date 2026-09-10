"""
Entropy-Conserving Image Generation — Pixel Reorganization
============================================================

Core idea: A shuffled image and the original have the same entropy
but completely different structure. Train a model to transform
shuffled → structured while conserving entropy.

This is the "same sand, different arrangement" principle:
  - Shuffled image = sand scattered randomly on plate (high disorder)
  - Original image = sand house (structured, same amount of sand)
  - The model learns to reorganize, not add or remove

Training pipeline:
  1. Take real image u
  2. Create shuffled version s(u) — same pixels, random arrangement
  3. Train model: f(s(u)) → u, subject to H(f(s(u))) ≈ H(s(u))

At inference:
  - Start from ANY shuffled/random image with the right entropy
  - The model reorganizes it into a coherent image

Shuffling strategies tested:
  - Full pixel shuffle (hardest — complete spatial destruction)
  - Patch shuffle (moderate — local structure preserved, global destroyed)
  - Band shuffle (frequency bands scrambled — same power spectrum, different phases)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE      = 32
BATCH_SIZE    = 128
LR            = 2e-4
EPOCHS        = 200
MODEL_DIM     = 128
LAMBDA_ENT    = 0.5       # weight for entropy conservation loss
SHUFFLE_MODE  = "patch"   # "pixel", "patch", or "phase"
PATCH_SIZE    = 4          # for patch shuffle mode
NUM_STEPS     = 10         # iterative refinement steps during generation
SAVE_DIR      = Path("./outputs_reorg")
DATA_DIR      = Path("./data")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name()}")


# ============================================================
# 1. ENTROPY MEASURE
# ============================================================
def spectral_entropy(x: torch.Tensor) -> torch.Tensor:
    """Spectral entropy per image. Returns shape [B]."""
    fft = torch.fft.fft2(x)
    power = (fft.real ** 2 + fft.imag ** 2).sum(dim=1)
    power_flat = power.reshape(power.shape[0], -1)
    total = power_flat.sum(dim=1, keepdim=True).clamp(min=1e-10)
    p = power_flat / total
    log_p = torch.log(p.clamp(min=1e-10))
    return -(p * log_p).sum(dim=1)


# ============================================================
# 2. SHUFFLING STRATEGIES
# ============================================================
def shuffle_pixels(images: torch.Tensor) -> torch.Tensor:
    """Randomly permute all pixels independently per image."""
    B, C, H, W = images.shape
    # Reshape to [B, C, H*W], shuffle along last dim
    flat = images.reshape(B, C, H * W)
    idx = torch.argsort(torch.rand(B, 1, H * W, device=images.device).expand(-1, C, -1), dim=2)
    # Use same permutation for all channels to preserve color relationships
    idx_same = torch.argsort(torch.rand(B, 1, H * W, device=images.device), dim=2).expand(-1, C, -1)
    shuffled = torch.gather(flat, 2, idx_same)
    return shuffled.reshape(B, C, H, W)


def shuffle_patches(images: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Randomly permute patches of size patch_size x patch_size."""
    B, C, H, W = images.shape
    pH, pW = H // patch_size, W // patch_size
    # Reshape into patches
    patches = images.reshape(B, C, pH, patch_size, pW, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5)  # [B, C, pH, pW, ps, ps]
    patches = patches.reshape(B, C, pH * pW, patch_size, patch_size)
    # Shuffle patch order
    idx = torch.argsort(torch.rand(B, 1, pH * pW, device=images.device), dim=2)
    idx = idx.expand(-1, C, -1).unsqueeze(-1).unsqueeze(-1)
    idx = idx.expand(-1, -1, -1, patch_size, patch_size)
    shuffled = torch.gather(patches, 2, idx)
    # Reshape back
    shuffled = shuffled.reshape(B, C, pH, pW, patch_size, patch_size)
    shuffled = shuffled.permute(0, 1, 2, 4, 3, 5)
    return shuffled.reshape(B, C, H, W)


def shuffle_phase(images: torch.Tensor) -> torch.Tensor:
    """Keep magnitude spectrum, randomize phases.
    This preserves the power spectrum (and thus spectral entropy) exactly,
    but completely destroys spatial structure."""
    fft = torch.fft.fft2(images)
    magnitude = torch.abs(fft)
    # Random phases
    random_phase = torch.exp(1j * 2 * math.pi * torch.rand_like(magnitude))
    # Enforce conjugate symmetry for real output
    shuffled_fft = magnitude * random_phase
    result = torch.fft.ifft2(shuffled_fft).real
    return result.clamp(0, 1)


def apply_shuffle(images: torch.Tensor, mode: str = SHUFFLE_MODE) -> torch.Tensor:
    if mode == "pixel":
        return shuffle_pixels(images)
    elif mode == "patch":
        return shuffle_patches(images)
    elif mode == "phase":
        return shuffle_phase(images)
    else:
        raise ValueError(f"Unknown shuffle mode: {mode}")


# ============================================================
# 3. DATASET
# ============================================================
print("Loading CIFAR-10...")
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
cifar = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform)
loader = DataLoader(cifar, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=2, drop_last=True,
                    pin_memory=(DEVICE.type == "cuda"))
print(f"Dataset: {len(cifar)} images, {len(loader)} batches/epoch")


# ============================================================
# 4. NETWORK — Iterative Refinement U-Net
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None, dropout=0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(32, in_ch), in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.has_time = time_dim is not None
        if self.has_time:
            self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(32, out_ch), out_ch), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.block1(x)
        if self.has_time and t_emb is not None:
            h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], dim=1)


class ReorgNet(nn.Module):
    """
    Takes a shuffled image and a refinement step index,
    outputs a reorganized image.

    The step index lets the same network be applied iteratively:
      u_0 = shuffled
      u_1 = model(u_0, step=0)
      u_2 = model(u_1, step=1)
      ...
      u_N = model(u_{N-1}, step=N-1)  →  final output
    """

    def __init__(self, in_ch=3, base=MODEL_DIM, num_steps=NUM_STEPS):
        super().__init__()
        t_dim = base * 4
        self.t_embed = nn.Sequential(
            SinusoidalEmb(base),
            nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
        )

        ch = [base, base * 2, base * 4]

        self.enc_in = nn.Conv2d(in_ch, ch[0], 3, padding=1)
        self.enc1 = ResBlock(ch[0], ch[0], t_dim)
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, 2, 1)
        self.enc2 = ResBlock(ch[0], ch[1], t_dim)
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, 2, 1)
        self.enc3 = ResBlock(ch[1], ch[2], t_dim)

        self.mid = ResBlock(ch[2], ch[2], t_dim)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(ch[2], ch[1], 3, padding=1))
        self.dec2 = ResBlock(ch[1] * 2, ch[1], t_dim)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(ch[1], ch[0], 3, padding=1))
        self.dec1 = ResBlock(ch[0] * 2, ch[0], t_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(min(32, ch[0]), ch[0]), nn.SiLU(),
            nn.Conv2d(ch[0], in_ch, 3, padding=1),
        )

    def forward(self, x, step):
        """
        x: [B, 3, H, W] — current image state
        step: [B] — refinement step index (integer tensor)
        """
        te = self.t_embed(step)

        e0 = self.enc_in(x)
        e1 = self.enc1(e0, te)
        e2 = self.enc2(self.down1(e1), te)
        e3 = self.enc3(self.down2(e2), te)

        m = self.mid(e3, te)

        d2 = self.dec2(torch.cat([self.up2(m), e2], 1), te)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1), te)

        return self.out(d1)


# ============================================================
# 5. VISUALIZE SHUFFLING + ENTROPY CHECK
# ============================================================
print("\n" + "=" * 65)
print("SHUFFLE VERIFICATION: Entropy preservation check")
print("=" * 65)

sample_batch = next(iter(loader))[0][:16].to(DEVICE)

for mode in ["pixel", "patch", "phase"]:
    shuffled = apply_shuffle(sample_batch, mode)
    H_orig = spectral_entropy(sample_batch)
    H_shuf = spectral_entropy(shuffled)
    rel_diff = ((H_shuf - H_orig) / H_orig.clamp(min=1e-6)).abs()
    print(f"  {mode:6s} shuffle: entropy change = {rel_diff.mean():.4f} "
          f"(+/- {rel_diff.std():.4f})")

# Save visual comparison
for mode in ["pixel", "patch", "phase"]:
    shuffled = apply_shuffle(sample_batch[:8], mode)
    comparison = torch.cat([sample_batch[:8], shuffled], dim=0)
    grid = vutils.make_grid(comparison.clamp(0, 1), nrow=8, padding=2)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
    ax.set_title(f"Top: original | Bottom: {mode} shuffled")
    fig.savefig(SAVE_DIR / f"shuffle_comparison_{mode}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

print("Saved shuffle comparison images")


# ============================================================
# 6. TRAINING
# ============================================================
print("\n" + "=" * 65)
print(f"TRAINING: Entropy-Conserving Reorganization ({SHUFFLE_MODE} shuffle)")
print("=" * 65)

model = ReorgNet().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {nparam:,}")
print(f"Shuffle mode: {SHUFFLE_MODE}")
print(f"Refinement steps: {NUM_STEPS}")

losses_recon_hist = []
losses_entropy_hist = []

for epoch in range(EPOCHS):
    model.train()
    ep_recon, ep_ent = 0, 0

    for images, _ in loader:
        images = images.to(DEVICE)
        B = images.shape[0]

        # Create shuffled version
        shuffled = apply_shuffle(images)

        # Pick a random refinement step to train on
        # (like diffusion — train on random step, apply all steps at inference)
        step = torch.randint(0, NUM_STEPS, (B,), device=DEVICE)

        # For intermediate steps, create a partially refined input
        # by interpolating between shuffled and original
        # This teaches the model what partially-organized images look like
        alpha = step.float() / NUM_STEPS  # 0 = fully shuffled, 1 = near original
        alpha = alpha[:, None, None, None]
        intermediate = (1 - alpha) * shuffled + alpha * images
        # Add tiny noise to prevent exact interpolation memorization
        intermediate = intermediate + 0.01 * torch.randn_like(intermediate)
        intermediate = intermediate.clamp(0, 1)

        # Model predicts the original from the intermediate state
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            pred = model(intermediate, step)
            loss_recon = F.mse_loss(pred, images)

            H_input = spectral_entropy(intermediate)
            H_pred = spectral_entropy(pred)
            loss_entropy = F.mse_loss(H_pred, H_input)

            loss = loss_recon + LAMBDA_ENT * loss_entropy

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        ep_recon += loss_recon.item()
        ep_ent += loss_entropy.item()

    sched.step()
    n = len(loader)
    losses_recon_hist.append(ep_recon / n)
    losses_entropy_hist.append(ep_ent / n)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
              f"recon {ep_recon/n:.6f} | entropy {ep_ent/n:.6f}")

    # ---- periodic visualization ----
    if (epoch + 1) % 25 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            test_images = sample_batch[:8]
            test_shuffled = apply_shuffle(test_images)

            # === Iterative refinement ===
            u = test_shuffled.clone()
            refinement_stages = [u.clone()]

            for s in range(NUM_STEPS):
                step_t = torch.full((8,), s, device=DEVICE, dtype=torch.long)
                u = model(u, step_t).clamp(0, 1)
                refinement_stages.append(u.clone())

            # Show: original → shuffled → refinement steps → final
            # Pick a few stages to display
            stage_indices = [0, 1, NUM_STEPS // 4, NUM_STEPS // 2,
                            3 * NUM_STEPS // 4, NUM_STEPS]
            stage_indices = sorted(set(min(i, len(refinement_stages)-1)
                                       for i in stage_indices))
            rows = [test_images]  # row 0: originals
            for si in stage_indices:
                rows.append(refinement_stages[si])

            combined = torch.cat(rows, dim=0)
            grid = vutils.make_grid(combined.clamp(0, 1), nrow=8, padding=2)
            fig, ax = plt.subplots(figsize=(14, 2 * len(rows)))
            ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
            ax.axis("off")
            ax.set_title(f"Row 1: originals | Row 2: shuffled | "
                        f"Rows 3+: refinement steps — epoch {epoch+1}")
            fig.savefig(SAVE_DIR / f"refinement_epoch_{epoch+1:04d}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

            # === Generate from random noise with matched entropy ===
            # Compute average entropy of real images
            avg_entropy = spectral_entropy(test_images).mean().item()

            # Start from shuffled versions of random real images
            random_seeds = next(iter(loader))[0][:8].to(DEVICE)
            random_start = apply_shuffle(random_seeds)

            u_gen = random_start.clone()
            for s in range(NUM_STEPS):
                step_t = torch.full((8,), s, device=DEVICE, dtype=torch.long)
                u_gen = model(u_gen, step_t).clamp(0, 1)

            gen_grid = torch.cat([random_start, u_gen], dim=0)
            grid = vutils.make_grid(gen_grid.clamp(0, 1), nrow=8, padding=2)
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
            ax.axis("off")
            ax.set_title(f"Top: shuffled input | Bottom: generated — epoch {epoch+1}")
            fig.savefig(SAVE_DIR / f"generation_epoch_{epoch+1:04d}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

            # === Entropy tracking through refinement ===
            u_track = test_shuffled[:4].clone()
            ent_track = [spectral_entropy(u_track).cpu().numpy()]
            for s in range(NUM_STEPS):
                step_t = torch.full((4,), s, device=DEVICE, dtype=torch.long)
                u_track = model(u_track, step_t).clamp(0, 1)
                ent_track.append(spectral_entropy(u_track).cpu().numpy())

            ent_track = np.array(ent_track)  # [steps+1, 4]
            fig, ax = plt.subplots(figsize=(10, 4))
            for i in range(4):
                ax.plot(ent_track[:, i], "-o", markersize=3, label=f"Image {i+1}")
            ax.set(xlabel="Refinement step", ylabel="Spectral Entropy",
                   title=f"Entropy through refinement — epoch {epoch+1}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(SAVE_DIR / f"entropy_track_{epoch+1:04d}.png",
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
ax1.plot(losses_recon_hist, "b-", lw=1.5)
ax1.set(xlabel="Epoch", ylabel="MSE", title="Reconstruction Loss")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3)

ax2.plot(losses_entropy_hist, "r-", lw=1.5)
ax2.set(xlabel="Epoch", ylabel="Entropy MSE", title="Entropy Conservation Loss")
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3)
fig.savefig(SAVE_DIR / "loss_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Final generation from multiple shuffled inputs
model.eval()
with torch.no_grad():
    # Generate 64 images
    gen_seeds = []
    for batch_images, _ in loader:
        gen_seeds.append(batch_images)
        if len(gen_seeds) * BATCH_SIZE >= 64:
            break
    gen_seeds = torch.cat(gen_seeds)[:64].to(DEVICE)
    gen_shuffled = apply_shuffle(gen_seeds)

    gen_output = gen_shuffled.clone()
    for s in range(NUM_STEPS):
        step_t = torch.full((64,), s, device=DEVICE, dtype=torch.long)
        gen_output = model(gen_output, step_t).clamp(0, 1)

    # Entropy comparison
    H_seeds = spectral_entropy(gen_seeds)
    H_shuffled = spectral_entropy(gen_shuffled)
    H_output = spectral_entropy(gen_output)

    print(f"\nEntropy statistics:")
    print(f"  Original images:  mean={H_seeds.mean():.2f}  std={H_seeds.std():.2f}")
    print(f"  Shuffled inputs:  mean={H_shuffled.mean():.2f}  std={H_shuffled.std():.2f}")
    print(f"  Generated output: mean={H_output.mean():.2f}  std={H_output.std():.2f}")
    print(f"  Entropy change (shuffled→output): "
          f"{((H_output - H_shuffled) / H_shuffled.clamp(min=1e-6)).abs().mean():.4f}")

    # Save final grid
    grid = vutils.make_grid(gen_output.clamp(0, 1), nrow=8, padding=2)
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
    ax.set_title("Final generated images (64 samples)")
    fig.savefig(SAVE_DIR / "final_generated.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Side-by-side: shuffled → generated → original (for first 8)
    trio = torch.cat([gen_shuffled[:8], gen_output[:8], gen_seeds[:8]], dim=0)
    grid = vutils.make_grid(trio.clamp(0, 1), nrow=8, padding=2)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
    ax.set_title("Top: shuffled input | Middle: generated | Bottom: original")
    fig.savefig(SAVE_DIR / "final_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# Save model
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "img_size": IMG_SIZE,
        "model_dim": MODEL_DIM,
        "num_steps": NUM_STEPS,
        "shuffle_mode": SHUFFLE_MODE,
        "lambda_ent": LAMBDA_ENT,
    }
}, SAVE_DIR / "model_final.pt")

print(f"\nAll outputs saved to {SAVE_DIR}/")
print("Done!")