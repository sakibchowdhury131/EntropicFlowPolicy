# Heat Dissipation Generative Models — Project Context

## Overview

This project explores a novel generative modeling paradigm based on the **heat equation** instead of noise-based diffusion. The core insight comes from a physical analogy: a sand house on a shaking plate gradually loses structure — not because sand is removed, but because it's *reorganized* toward maximum entropy. We reverse this process with a learned model.

## Core Idea

### Standard Diffusion
- Forward: add i.i.d. Gaussian noise to data
- Reverse: learn to predict and remove noise
- All frequencies corrupted equally

### Heat Dissipation (Ours)
- Forward: blur data via the heat equation (spatially/temporally correlated smoothing)
- Reverse: learn to restore structure from blurred data
- High frequencies die first, low frequencies last → **natural coarse-to-fine hierarchy**
- **Mass/energy is conserved** — no information is added or removed, only reorganized

### Key Mathematical Formulation

**Forward process (Fourier space):**
```
û(k, t) = û(k, 0) · exp(-2π²σ²|k|²)
```

**Sigma schedule (sqrt for uniform attenuation):**
```python
self.sigmas = torch.sqrt(torch.linspace(0.01**2, alpha_max**2, num_timesteps))
```

**Reverse process:**
```
For t = T-1 down to 0:
    a_0_pred = model(a_t, t, condition)
    if t > 0:
        a_{t-1} = heat_blur(a_0_pred, t-1)  # re-blur to one level less
    else:
        a_0 = a_0_pred  # final output
```

**Key implementation detail:** Use FFT-based blurring with periodic boundaries for exact mass conservation. Spatial-domain convolution fails for large sigma on small signals.

## Three Research Directions

### 1. Heat Dissipation for Image Generation (Implemented, Validated)
- Forward: 2D heat equation blurs images spatially
- Trained on synthetic shapes (CIFAR-10 ready)
- **Status:** Working proof of concept. Reconstruction works well. Generation from flat start produces recognizable shapes but needs more training. Checkerboard artifacts fixed by replacing ConvTranspose2d with Upsample+Conv.
- **Key finding:** The sigma schedule matters enormously. Use sqrt schedule so frequency bands decay uniformly across timesteps. Linear-in-sigma is actually quadratic-in-attenuation and kills high frequencies too early.
- **Not novel:** "Cold Diffusion" (Bansal et al., 2022) and "Blurring Diffusion Models" (Hoogeboom & Salimans, 2023) cover similar ground.

### 2. Entropy-Conserving Generation (Novel, Theoretical)
- Instead of traversing entropy gradients, move along **constant-entropy manifolds**
- A phase-shuffled image has identical spectral entropy but random structure
- Train model to reorganize phase-shuffled → coherent while conserving entropy
- **This is the most novel direction** — unexplored in literature
- Connects to Riemannian flow matching with constant-entropy level sets as the manifold
- For video: consecutive frames naturally have near-constant entropy, making this a natural fit

### 3. Heat Dissipation Policy for Robotics (Implemented, Ready for Testing)
- Apply 1D heat equation to action trajectories (blur along time axis)
- Drop-in replacement for Diffusion Policy on PushT task
- Same dataset, environment, network architecture (ConditionalUnet1D)
- **Potential advantages:**
  - Smoother trajectories (low-freq structure preserved by construction)
  - Coarse-to-fine planning (can inspect rough plan at intermediate steps)
  - Fewer inference steps needed (hierarchy means early steps matter most)
  - Natural safety constraint (can't generate high-frequency jitter)
- **Status:** Code ready, needs training and evaluation against Diffusion Policy baseline

## File Structure

```
project/
├── heat_dissipation_train.py          # Image generation (synthetic/CIFAR-10)
├── entropy_conserving_video.py        # Video frame prediction with entropy constraint
├── entropy_conserving_image_gen.py    # Image generation via pixel reorganization
├── heat_dissipation_policy_pusht.py   # Robotics policy for PushT task
└── CLAUDE.md                          # This file
```

## What Has Been Validated

1. **Mass conservation** — FFT-based blurring preserves total pixel/action mass exactly (ratio = 1.0000)
2. **Frequency hierarchy** — High frequencies die exponentially faster than low frequencies, confirmed by band energy analysis
3. **Reconstruction** — Model successfully recovers images from moderate blur levels
4. **Coarse-to-fine generation** — Detail levels visualization shows blobs → shapes → edges → texture
5. **Sigma schedule** — Sqrt schedule produces uniform attenuation across timesteps; linear schedule wastes most timesteps on already-flat signals

## What Needs Work

### Immediate (Engineering)
- Train heat policy on PushT and compare score vs Diffusion Policy (target: 0.7-0.9)
- Train image model on CIFAR-10 for 200+ epochs with sqrt schedule, MODEL_DIM=128, attention
- Measure actual inference speed comparison (wall clock time per action chunk)
- Implement trajectory smoothness metrics (jerk, acceleration variance) for robotics comparison

### Medium-term (Research)
- Implement **hard** entropy conservation via projection onto constant-entropy manifold tangent plane
- Derive the gradient of spectral entropy ∇H and the projection operator
- Test whether fewer reverse steps (10-20 vs 50-100) maintain quality — the hierarchy should allow this
- Try anisotropic diffusion: different blur rates for different action dimensions / image regions
- Add stochastic noise injection during reverse process for sample diversity

### Long-term (Paper)
- The strongest paper is NOT about image generation (Cold Diffusion exists)
- Best framing: **"Heat Dissipation Policies: Hierarchical Trajectory Generation with Coarse-to-Fine Planning"**
- Compare against Diffusion Policy on PushT, ToolHang, and real robot tasks
- Show: (a) comparable task success, (b) smoother trajectories, (c) fewer inference steps, (d) inspectable coarse plans
- Alternative paper: **"Entropy-Conserving Generative Flows"** — the constant-entropy manifold idea is genuinely novel but needs more theoretical development

## Technical Details to Remember

### Common Bugs
- **Boundary padding:** Spatial-domain convolution with zero padding breaks mass conservation. Always use FFT.
- **Checkerboard artifacts:** ConvTranspose2d causes grid patterns. Replace with `nn.Upsample(mode='nearest') + nn.Conv2d`.
- **Sigma schedule:** Linear-in-sigma is quadratic-in-exponent. Use `sqrt(linspace(σ_min², σ_max², T))` for uniform information destruction.
- **Flat start generation:** If ALPHA_MAX is too small, the forward process never reaches true flatness. The model never sees flat inputs during training, so generation from flat fails. Verify with spatial variance check.

### Key Hyperparameters
- **Image generation:** IMG_SIZE=32, ALPHA_MAX=10, NUM_TIMESTEPS=30, sqrt sigma schedule, MODEL_DIM=128
- **Robotics policy:** pred_horizon=16, obs_horizon=2, action_horizon=8, num_heat_steps=50, alpha_max=6.0
- **Training:** AdamW, lr=1e-4 (images) or 2e-4, cosine LR schedule, EMA for inference, gradient clipping=1.0

### Architecture
The same ConditionalUnet1D from Diffusion Policy works for heat dissipation. The only change is:
- Diffusion: network predicts **noise ε** → `clean = noisy - ε`
- Heat: network predicts **clean signal a_0** directly → `a_0 = model(a_t, t, obs)`

### Dependencies
```
torch, torchvision, matplotlib, numpy
zarr, numcodecs, gdown           # for PushT dataset
pygame, pymunk, gym, shapely     # for PushT environment
diffusers                        # for EMAModel (optional, can implement manually)
```

## Novelty Assessment

| Direction | Novel? | Why / Why Not |
|-----------|--------|---------------|
| Heat equation for images | No | Cold Diffusion (2022), Blurring Diffusion Models (2023) |
| Constant-entropy manifold flows | Yes | No prior work on generative flows constrained to entropy level sets |
| Heat equation for robot actions | Partially | Cold Diffusion exists but not applied to action generation; the coarse-to-fine planning angle is new |
| Entropy conservation for video | Yes | No prior work using entropy conservation as the defining constraint for temporal consistency |

## Reference Papers
- Cold Diffusion (Bansal et al., 2022) — deterministic degradations as forward process
- Blurring Diffusion Models (Hoogeboom & Salimans, 2023) — blur-based forward process
- Diffusion Policy (Chi et al., RSS 2023) — diffusion for robot action generation
- Score-Based Generative Models via SDEs (Song et al., 2021) — SDE framework
- Riemannian Flow Matching (Chen & Lipman, 2023) — flows on manifolds
