# Entropic Flow Policy (EFP)

**Findings, plots, and rollout videos: [live results page](https://claude.ai/code/artifact/b035f3a5-932d-4362-9e3e-6947c9284289)**

An imitation-learning framework that replaces the reward function with a single
information-theoretic quantity: the log-count of trajectories still reachable
from the current state to the goal ("goal-reachability entropy"). A good policy
is one that always moves toward lower entropy. This repo also contains a second,
related line of work — a **Heat Dissipation Policy** that replaces diffusion
policies' Gaussian noise with the heat equation as the forward corruption process.

```
H(s, s_g) = log |{ τ : s →τ→ s_g, |τ| ≤ H }|
```

## Headline results

- **Push-T, adversarial start:** EFP trained with failure-demonstration
  supervision reaches 64%/56% success (Standard/Perturb conditions) while every
  baseline (plain EFP, Diffusion Policy, BC, MDN-BC) scores ≤24%/0%. Failure
  demos are not an optimization — without them EFP's entropy guidance actively
  hurts performance (0% in both conditions, worse than the Diffusion Policy
  baseline).
- **Heat Dissipation Policy:** trained the naive way, it never learns
  (eval score flat at 0.000 for 100 epochs). Randomizing the DC/zero-frequency
  offset of each training action chunk ("DC augmentation") fixes this —
  best eval score 0.857, and on 16 held-out seeds it's competitive with
  standard Diffusion Policy (mean score 0.912 vs 0.982, wins 6/16 seeds).

Full charts, per-seed tables, and gif/video rollouts for every method and
condition are on the [results page](https://claude.ai/code/artifact/b035f3a5-932d-4362-9e3e-6947c9284289).

## Repository layout

| Path | Contents |
|---|---|
| `entropic_flow_policy.py` | Exp 1 — 3-link planar arm reaching, full EFP pipeline |
| `efp_hard_v2.py` … `efp_hard_v6.py` | Exp 2 — 2D multi-modal obstacle navigation, iterated task design |
| `efp_pusht.py` | Exp 3 — EFP (with/without failure supervision) on the `gym-pusht` benchmark |
| `efp_pusht_real.py` | EFP vs. Diffusion Policy on the official Push-T human-demo dataset |
| `efp_guided_bc.py` | Ablation — entropy-gradient guidance applied on top of plain BC |
| `viz_pusht.py`, `viz_real.py`, `plot_trajectories.py` | Rollout/entropy-field visualization utilities |
| `heat-dissipation-policy-pushT.py` | Heat Dissipation Policy implementation for Push-T |
| `compare_dp_heat_pusht.py` | Main training script — Diffusion Policy vs. Heat Policy, side by side |
| `eval_configs.py` | Post-hoc DP-vs-Heat evaluation across diverse seeds, renders comparison videos |
| `eval_novel.py` | Generalization eval — DC-augmented Heat vs. DP on unseen/hard configurations |
| `plot_dc_vs_noise.py` | Sensitivity analysis for the DC-augmentation ablation |
| `scores.csv` | Per-epoch loss/score log for the DP-vs-Heat training run |
| `docs/research_handoff.md` | Original theory write-up: formalism, Exp 1 & 2 derivations |
| `docs/heat_dissipation_policy.md` | Heat-equation generative modeling: theory, math, three research directions |
| `generative_extensions/` | Side project — heat-dissipation and constant-entropy image/video generation |

Trained checkpoints, the raw Push-T dataset, and full-resolution videos/gifs are
not tracked in git (see `.gitignore`) — they're large and regenerable from these
scripts. Curated rollout clips are embedded directly in the results page instead.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
# for Push-T experiments: pip install zarr numcodecs gdown pygame pymunk gym shapely diffusers
```

## Background

Developed collaboratively with Claude (Anthropic), starting from the idea that
entropy of remaining feasible trajectories — not accumulated reward — is enough
of a signal to learn from, since demonstrations already trace out the empirical
shape of that entropy funnel. See `docs/research_handoff.md` for the full
derivation and the two experiments that motivated the Push-T task design.
