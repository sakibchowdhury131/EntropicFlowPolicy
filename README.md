# Entropic Flow Policy (EFP)

**Findings, plots, and rollout videos: [live results page](https://claude.ai/code/artifact/b035f3a5-932d-4362-9e3e-6947c9284289)**

An imitation-learning framework that replaces the reward function with a single
information-theoretic quantity: the log-count of trajectories still reachable
from the current state to the goal ("goal-reachability entropy"). A good policy
is one that always moves toward lower entropy.

```
H(s, s_g) = log |{ τ : s →τ→ s_g, |τ| ≤ H }|
```

A related, code-independent line of work — the **Heat Dissipation Policy**,
which replaces diffusion policies' Gaussian noise with the heat equation as the
forward corruption process — lives in a separate repo:
[HeatDissipationPolicy](https://github.com/sakibchowdhury131/HeatDissipationPolicy).
The two share only a benchmark (Push-T) and a thesis (entropy contraction as the
learning signal); neither imports or depends on the other.

## Headline result

**Push-T, adversarial start:** EFP trained with failure-demonstration
supervision reaches 64%/56% success (Standard/Perturb conditions) while every
baseline (plain EFP, Diffusion Policy, BC, MDN-BC) scores ≤24%/0%. Failure
demos are not an optimization — without them EFP's entropy guidance actively
hurts performance (0% in both conditions, worse than the Diffusion Policy
baseline).

Full charts, per-condition tables, and gif rollouts for every method are on the
[results page](https://claude.ai/code/artifact/b035f3a5-932d-4362-9e3e-6947c9284289).

## Repository layout

| Path | Contents |
|---|---|
| `entropic_flow_policy.py` | Exp 1 — 3-link planar arm reaching, full EFP pipeline |
| `efp_hard_v2.py` … `efp_hard_v6.py` | Exp 2 — 2D navigation around road-block obstacles, iterated task design (open gap → closed gap → fixed start zone) |
| `efp_pusht.py` | Exp 3 — EFP (with/without failure supervision) on the `gym-pusht` benchmark |
| `efp_pusht_real.py` | EFP vs. Diffusion Policy on the official Push-T human-demo dataset |
| `efp_guided_bc.py` | Ablation — entropy-gradient guidance applied on top of plain BC |
| `viz_pusht.py`, `viz_real.py` | Rollout and entropy-field visualization utilities |
| `docs/research_handoff.md` | Original theory write-up: formalism, Exp 1 & 2 derivations |

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
