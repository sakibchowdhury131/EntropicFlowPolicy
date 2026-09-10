"""
Heat Dissipation Generative Model
==================================
Instead of adding/removing noise (diffusion), this model:
  Forward: blurs images via the heat equation (structure dissolves, entropy increases)
  Reverse: a neural network learns to restore structure (sharpen, reconstruct detail)

The key insight: structure naturally decays under local averaging (like shaking a sand house).
Generation = reversing that natural decay, guided by a learned model.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 32
IMG_CHANNELS = 3
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
NUM_TIMESTEPS = 30        # discretized degradation steps
ALPHA_MAX = 10         # max diffusion coefficient (blur strength)
MODEL_DIM = 64            # base channel width of U-Net
SAVE_DIR = Path("outputs")
SAVE_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")

# ============================================================
# 1. FORWARD PROCESS: Heat Equation Blurring
# ============================================================
class HeatDissipation:
    """
    Forward process via heat equation, computed in Fourier space.

    û(k, t) = û(k, 0) · exp(-σ² |k|² / 2)

    Periodic boundaries (exact mass conservation).
    Works for any sigma, no kernel size issues.
    """

    def __init__(self, num_timesteps=NUM_TIMESTEPS, alpha_max=ALPHA_MAX):
        self.num_timesteps = num_timesteps
        self.sigmas = torch.exp(torch.linspace(math.log(0.01), math.log(alpha_max), num_timesteps))
        self._filters = {}

    def _get_filter(self, sigma, H, W, device):
        key = (round(sigma, 6), H, W)
        if key not in self._filters:
            fy = torch.fft.fftfreq(H, device=device)
            fx = torch.fft.fftfreq(W, device=device)
            fy, fx = torch.meshgrid(fy, fx, indexing='ij')
            freq_sq = fy**2 + fx**2
            filt = torch.exp(-2 * (math.pi ** 2) * sigma**2 * freq_sq)
            self._filters[key] = filt
        return self._filters[key]

    def blur(self, x, t_index):
        """Apply heat equation blur at time step t_index."""
        sigma = self.sigmas[t_index].item()
        H, W = x.shape[-2], x.shape[-1]
        filt = self._get_filter(sigma, H, W, x.device)

        # FFT → multiply by Gaussian filter → IFFT
        x_freq = torch.fft.fft2(x)
        x_filtered = x_freq * filt[None, None, :, :]
        return torch.fft.ifft2(x_filtered).real

    def sample_timestep(self, batch_size):
        return torch.randint(0, self.num_timesteps, (batch_size,))


heat = HeatDissipation()

# ============================================================
# 2. U-NET ARCHITECTURE (Reverse Process Network)
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    """Encode timestep t as a sinusoidal positional embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(self.conv1(x))
        h = F.silu(h)
        # Add time embedding
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(self.conv2(h))
        h = F.silu(h)
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """
    U-Net that takes a blurred image u_t and timestep t,
    and predicts the original image u_0.

    f_θ(u_t, t) → u_0
    """
    def __init__(self, in_ch=IMG_CHANNELS, base_dim=MODEL_DIM):
        super().__init__()
        time_dim = base_dim * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder (downsampling)
        self.enc1 = ResBlock(in_ch, base_dim, time_dim)
        self.enc2 = ResBlock(base_dim, base_dim * 2, time_dim)
        self.enc3 = ResBlock(base_dim * 2, base_dim * 4, time_dim)

        self.down1 = nn.Conv2d(base_dim, base_dim, 4, 2, 1)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 2, 4, 2, 1)

        # Bottleneck
        self.mid = ResBlock(base_dim * 4, base_dim * 4, time_dim)

        # Decoder (upsampling)
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, 2, 1)

        self.dec3 = ResBlock(base_dim * 4 + base_dim * 4, base_dim * 4, time_dim)
        self.dec2 = ResBlock(base_dim * 2 + base_dim * 2, base_dim * 2, time_dim)
        self.dec1 = ResBlock(base_dim + base_dim, base_dim, time_dim)

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        # Encoder
        e1 = self.enc1(x, t_emb)           # [B, D, 32, 32]
        e2 = self.enc2(self.down1(e1), t_emb)  # [B, 2D, 16, 16]
        e3 = self.enc3(self.down2(e2), t_emb)  # [B, 4D, 8, 8]

        # Bottleneck
        m = self.mid(e3, t_emb)             # [B, 4D, 8, 8]

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([m, e3], dim=1), t_emb)   # [B, 4D, 8, 8]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)  # [B, 2D, 16, 16]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)  # [B, D, 32, 32]

        return self.out(d1)


# ============================================================
# 3. DATASET - Synthetic structured images
# ============================================================
print("Generating synthetic training dataset...")
print("(Geometric shapes with color gradients — structured images with clear detail)")

class SyntheticImageDataset(torch.utils.data.Dataset):
    """
    Generates images with geometric shapes, gradients, and edges.
    These have clear spatial structure that the heat equation will dissolve.
    """
    def __init__(self, num_images=10000, img_size=IMG_SIZE):
        self.num_images = num_images
        self.img_size = img_size
        # Pre-generate all images for consistency
        self.images = self._generate_all()

    def _generate_all(self):
        images = []
        for i in range(self.num_images):
            img = self._make_image(i)
            images.append(img)
        return torch.stack(images)

    def _make_image(self, seed):
        rng = np.random.RandomState(seed)
        img = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)

        # Background gradient
        bg_color = rng.rand(3) * 0.5
        direction = rng.choice(['h', 'v', 'd'])
        for c in range(3):
            if direction == 'h':
                grad = np.linspace(bg_color[c], bg_color[c] + 0.3, self.img_size)
                img[c] = grad[np.newaxis, :]
            elif direction == 'v':
                grad = np.linspace(bg_color[c], bg_color[c] + 0.3, self.img_size)
                img[c] = grad[:, np.newaxis]
            else:
                gx = np.linspace(0, 1, self.img_size)
                gy = np.linspace(0, 1, self.img_size)
                img[c] = bg_color[c] + 0.3 * (gx[np.newaxis, :] + gy[:, np.newaxis]) / 2

        # Add 2-5 geometric shapes
        num_shapes = rng.randint(2, 6)
        yy, xx = np.mgrid[0:self.img_size, 0:self.img_size].astype(np.float32)

        for _ in range(num_shapes):
            color = rng.rand(3) * 0.8 + 0.2
            cx, cy = rng.randint(4, self.img_size - 4, 2)
            shape_type = rng.choice(['circle', 'rect', 'triangle'])

            if shape_type == 'circle':
                r = rng.randint(3, self.img_size // 3)
                mask = ((xx - cx)**2 + (yy - cy)**2) < r**2
            elif shape_type == 'rect':
                w, h = rng.randint(3, self.img_size // 3, 2)
                mask = (np.abs(xx - cx) < w) & (np.abs(yy - cy) < h)
            else:  # triangle
                size = rng.randint(4, self.img_size // 3)
                mask = ((yy - cy + size) > 0) & (np.abs(xx - cx) < (yy - cy + size) * 0.7) & ((yy - cy) < size)

            for c in range(3):
                img[c][mask] = color[c]

        # Add some noise for texture
        img += rng.randn(3, self.img_size, self.img_size).astype(np.float32) * 0.02
        img = np.clip(img, 0, 1)
        return torch.from_numpy(img)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], 0  # 0 = dummy label

dataset = SyntheticImageDataset(num_images=5000)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True, pin_memory=False)

print(f"Dataset: {len(dataset)} synthetic images, {len(dataloader)} batches per epoch")

# ============================================================
# 4. TRAINING
# ============================================================
model = SimpleUNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {param_count:,}")

def save_visualization(images, filename, nrow=8, title=None):
    """Save a grid of images."""
    grid = vutils.make_grid(images.clamp(0, 1), nrow=nrow, padding=2, normalize=False)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_forward_process(dataloader):
    """Show the forward heat dissipation process on sample images."""
    batch = next(iter(dataloader))[0][:8].to(DEVICE)
    timesteps_to_show = [0, NUM_TIMESTEPS//6, NUM_TIMESTEPS//3, NUM_TIMESTEPS//2,
                         3*NUM_TIMESTEPS//4, NUM_TIMESTEPS-1]

    all_images = []
    for t_idx in timesteps_to_show:
        blurred = heat.blur(batch, t_idx)
        all_images.append(blurred)

    combined = torch.cat(all_images, dim=0)
    save_visualization(combined, SAVE_DIR / "forward_process.png", nrow=8,
                       title="Forward Process: Heat Dissipation (t=0 → t=T)")
    print(f"Saved forward process visualization")


# Visualize forward process
visualize_forward_process(dataloader)

print(f"\nStarting training for {NUM_EPOCHS} epochs...")
print("=" * 60)

losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)  # u_0: original images
        batch_size = images.shape[0]

        # Sample random timesteps
        t = heat.sample_timestep(batch_size)

        # Forward process: blur images to get u_t
        u_t = torch.stack([heat.blur(images[i:i+1], t[i].item()) for i in range(batch_size)])
        u_t = u_t.squeeze(1)

        # Predict original image from blurred version
        # Loss: || f_θ(u_t, t) - u_0 ||²
        t_device = t.to(DEVICE)
        u_0_pred = model(u_t, t_device)

        loss = F.mse_loss(u_0_pred, images)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Save sample generations periodically
    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            # === GENERATION: Reverse the heat equation ===
            # Start from near-flat (heavily blurred real images as seed)
            seed_images = next(iter(dataloader))[0][:16].to(DEVICE)

            # Start from maximum blur
            u = heat.blur(seed_images, NUM_TIMESTEPS - 1)

            # Also try starting from pure flat (global average color)
            flat_start = seed_images.mean(dim=(2, 3), keepdim=True).expand_as(seed_images)
            # Add tiny perturbation to break symmetry
            flat_start = flat_start + 0.01 * torch.randn_like(flat_start)
            u_flat = flat_start

            # Iterative reverse process
            steps_to_save = []
            for step in reversed(range(NUM_TIMESTEPS)):
                t_batch = torch.full((16,), step, device=DEVICE, dtype=torch.long)

                # From blurred seeds
                u_0_pred = model(u, t_batch)
                if step > 0:
                    # Re-blur the prediction to t-1, then use it as next input
                    u = heat.blur(u_0_pred.clamp(0, 1), step - 1)
                else:
                    u = u_0_pred

                # From flat start
                u_0_pred_flat = model(u_flat, t_batch)
                if step > 0:
                    u_flat = heat.blur(u_0_pred_flat.clamp(0, 1), step - 1)
                else:
                    u_flat = u_0_pred_flat

            # Save results
            save_visualization(
                u.clamp(0, 1),
                SAVE_DIR / f"gen_from_blur_epoch_{epoch+1:03d}.png",
                nrow=8, title=f"Generated from max-blur seeds (Epoch {epoch+1})"
            )
            save_visualization(
                u_flat.clamp(0, 1),
                SAVE_DIR / f"gen_from_flat_epoch_{epoch+1:03d}.png",
                nrow=8, title=f"Generated from flat start (Epoch {epoch+1})"
            )

            # Show at multiple detail levels (stop reversal at different points)
            detail_levels = []
            u_detail = heat.blur(seed_images[:8], NUM_TIMESTEPS - 1)
            detail_stops = [NUM_TIMESTEPS-1, NUM_TIMESTEPS*3//4, NUM_TIMESTEPS//2,
                           NUM_TIMESTEPS//4, 0]

            for stop_t in reversed(range(NUM_TIMESTEPS)):
                t_batch = torch.full((8,), stop_t, device=DEVICE, dtype=torch.long)
                u_0_pred = model(u_detail, t_batch)
                if stop_t > 0:
                    u_detail = heat.blur(u_0_pred.clamp(0, 1), stop_t - 1)
                else:
                    u_detail = u_0_pred
                if stop_t in detail_stops:
                    detail_levels.append(u_detail.clamp(0, 1))

            detail_combined = torch.cat(detail_levels, dim=0)
            save_visualization(
                detail_combined,
                SAVE_DIR / f"detail_levels_epoch_{epoch+1:03d}.png",
                nrow=8, title=f"Detail levels: abstract → full detail (Epoch {epoch+1})"
            )

        print(f"  → Saved generation samples at epoch {epoch+1}")

# ============================================================
# 5. SAVE TRAINING CURVE
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(losses, 'b-', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Heat Dissipation Model - Training Loss')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig(SAVE_DIR / "training_loss.png", dpi=150)
plt.close()
print(f"\nSaved training loss curve")

# ============================================================
# 6. FINAL GENERATION + ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("FINAL GENERATION & ANALYSIS")
print("=" * 60)

model.eval()
with torch.no_grad():
    # Get a batch for analysis
    real_images = next(iter(dataloader))[0][:16].to(DEVICE)

    # 1) Show reconstruction quality at different blur levels
    blur_levels = [2, NUM_TIMESTEPS//4, NUM_TIMESTEPS//2, 3*NUM_TIMESTEPS//4, NUM_TIMESTEPS-1]
    recon_rows = [real_images[:8]]  # first row: originals

    for t_idx in blur_levels:
        blurred = heat.blur(real_images[:8], t_idx)
        t_batch = torch.full((8,), t_idx, device=DEVICE, dtype=torch.long)
        recon = model(blurred, t_batch).clamp(0, 1)
        recon_rows.append(recon)

    recon_combined = torch.cat(recon_rows, dim=0)
    save_visualization(
        recon_combined, SAVE_DIR / "reconstruction_quality.png", nrow=8,
        title="Row 1: originals | Rows 2-6: reconstructions from increasing blur"
    )

    # 2) Entropy analysis - verify mass conservation
    print("\nMass conservation check:")
    for t_idx in [0, NUM_TIMESTEPS//3, NUM_TIMESTEPS//2, NUM_TIMESTEPS-1]:
        blurred = heat.blur(real_images[:4], t_idx)
        orig_mass = real_images[:4].sum(dim=(1,2,3)).mean().item()
        blur_mass = blurred.sum(dim=(1,2,3)).mean().item()
        print(f"  t={t_idx:2d}: original mass={orig_mass:.2f}, blurred mass={blur_mass:.2f}, "
              f"ratio={blur_mass/orig_mass:.4f}")

    # 3) Frequency analysis - show that high freq dies first
    def compute_freq_energy(img):
        """Compute energy at different frequency bands."""
        fft = torch.fft.fft2(img)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        h, w = img.shape[-2:]
        cy, cx = h // 2, w // 2

        # Define frequency bands
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist = torch.sqrt((y - cy).float()**2 + (x - cx).float()**2).to(img.device)
        max_dist = math.sqrt(cy**2 + cx**2)

        bands = []
        for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
            mask = (dist >= lo * max_dist) & (dist < hi * max_dist)
            energy = magnitude[:, :, mask].mean().item()
            bands.append(energy)
        return bands

    print("\nFrequency energy by band (low→high freq):")
    print(f"  {'t':>3s} | {'Band 1':>8s} {'Band 2':>8s} {'Band 3':>8s} {'Band 4':>8s} {'Band 5':>8s}")
    print(f"  " + "-" * 50)
    for t_idx in [0, NUM_TIMESTEPS//6, NUM_TIMESTEPS//3, NUM_TIMESTEPS//2, NUM_TIMESTEPS-1]:
        blurred = heat.blur(real_images[:4], t_idx)
        bands = compute_freq_energy(blurred)
        bands_str = " ".join(f"{b:8.2f}" for b in bands)
        print(f"  {t_idx:3d} | {bands_str}")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'img_size': IMG_SIZE,
        'img_channels': IMG_CHANNELS,
        'model_dim': MODEL_DIM,
        'num_timesteps': NUM_TIMESTEPS,
        'alpha_max': ALPHA_MAX,
    }
}, SAVE_DIR / "heat_dissipation_model.pt")

print(f"\nModel saved to {SAVE_DIR / 'heat_dissipation_model.pt'}")
print(f"All outputs saved to {SAVE_DIR}/")
print("\nDone!")