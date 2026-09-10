"""
Entropic Flow Policy — Push-T Benchmark (gym-pusht)
====================================================
Core claim: failure supervision in the entropy estimator improves policy
quality over baselines that only see success demonstrations.

Environment : gym-pusht (pymunk T-block physics)
Observation : 5-D [agent_x, agent_y, block_x, block_y, block_angle]
Norm. state : 6-D [ax/512, ay/512, bx/512, by/512, cos(θ), sin(θ)]
Action      : 2-D position target [0,512]² (normalised to [0,1]² for nets)
Goal        : T-block centred at (256,256) with angle π/4

Expert modes:
  LEFT  — detour left  (x≈100), approach behind block, push toward goal
  RIGHT — detour right (x≈410), symmetric detour

Failure demos: agent descends directly above block → pushes block DOWN (away
from goal at y=256), labelled H_FAILURE = log(200) ≈ 5.30.

Baselines
  EFP (w/ failure)  : flow ODE + entropy gradient (failure-supervised Hθ)
  EFP (no failure)  : flow ODE + entropy gradient (success-only Hθ)
  Diffusion Policy  : same VelocityNet, no guidance at inference
  BC (Mean)         : deterministic MLP regression
  MDN-BC            : Gaussian mixture BC
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
import gym_pusht  # registers gym_pusht/PushT-v0 with gymnasium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(0)
torch.manual_seed(0)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG          = 512
GOAL_X       = 256.0
GOAL_Y       = 256.0
GOAL_ANGLE   = math.pi / 4
GOAL_VEC     = np.array([GOAL_X/IMG, GOAL_Y/IMG,
                          math.cos(GOAL_ANGLE), math.sin(GOAL_ANGLE)],
                         dtype=np.float32)          # 4-D goal

BLK_ORIGIN_X = 256.0
BLK_ORIGIN_Y = 350.0   # far below goal → initial coverage ≈ 0.0
BLK_INIT_ANG = math.pi / 4

STATE_DIM    = 6
GOAL_DIM     = 4
ACT_DIM      = 2
HIDDEN       = 256
K_MDN        = 3
N_FLOW_STEPS = 10
T_DIM        = 16           # sinusoidal time embedding dimension
H_FAILURE    = math.log(200.0)   # ≈ 5.30, well above max success label
SUCCESS_THR  = 0.35              # coverage threshold for "success"
GUIDANCE_SCALE = 1.5             # entropy gradient step (overrides module-level)

# ── Time embedding ─────────────────────────────────────────────────────────────
def t_embed(t, dim=T_DIM):
    """t: (B,) float tensor → (B, dim) sinusoidal embedding."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], dtype=torch.float32)
    t = t.float().view(-1, 1)                          # (B, 1)
    half = dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(half, dtype=torch.float32) / half)  # (half,)
    args  = t * freqs                                  # (B, half)
    return torch.cat([args.sin(), args.cos()], dim=-1) # (B, dim)

# ── Networks ───────────────────────────────────────────────────────────────────
class EntropyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + GOAL_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),                nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )
    def forward(self, sg):                             # sg: (B, 10)
        return self.net(sg).squeeze(-1)                # (B,)

class DynNet(nn.Module):
    """Residual one-step dynamics: s_{t+1} = f(s_t, a_t) + s_t."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),               nn.ReLU(),
            nn.Linear(HIDDEN, STATE_DIM),
        )
    def forward(self, s, a):                           # s:(B,6) a:(B,2)
        return self.net(torch.cat([s, a], dim=-1)) + s # residual

class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        inp = STATE_DIM + GOAL_DIM + ACT_DIM + T_DIM
        self.net = nn.Sequential(
            nn.Linear(inp,    HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, ACT_DIM),
        )
    def forward(self, s, g, a_t, t_emb):
        return self.net(torch.cat([s, g, a_t, t_emb], dim=-1))

class BCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + GOAL_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),                nn.ReLU(),
            nn.Linear(HIDDEN, ACT_DIM),
        )
    def forward(self, sg):
        return self.net(sg)

class MDNNet(nn.Module):
    def __init__(self, K=K_MDN):
        super().__init__()
        self.K = K
        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM + GOAL_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),                nn.ReLU(),
        )
        self.head_pi  = nn.Linear(HIDDEN, K)
        self.head_mu  = nn.Linear(HIDDEN, K * ACT_DIM)
        self.head_lsig = nn.Linear(HIDDEN, K * ACT_DIM)

    def forward(self, sg):
        h      = self.backbone(sg)
        pi     = torch.softmax(self.head_pi(h), dim=-1)                   # (B, K)
        mu     = self.head_mu(h).view(-1, self.K, ACT_DIM)                # (B, K, 2)
        sigma  = self.head_lsig(h).view(-1, self.K, ACT_DIM).exp().clamp(1e-4, 2.0)
        return pi, mu, sigma                                               # (B,K) (B,K,2) (B,K,2)

# ── Environment helpers ────────────────────────────────────────────────────────
def make_env():
    return gym.make("gym_pusht/PushT-v0", obs_type="state")

def raw_to_state(obs):
    """5-D raw obs → 6-D normalised state [0,1]."""
    ax, ay, bx, by, ba = obs
    return np.array([ax/IMG, ay/IMG, bx/IMG, by/IMG,
                     math.cos(ba), math.sin(ba)], dtype=np.float32)

def reset_env(env, agent_x=256, agent_y=50,
              bx_off=0.0, by_off=0.0, ba_off=0.0):
    """Reset env with block near canonical start, return raw obs."""
    vec = [float(agent_x), float(agent_y),
           BLK_ORIGIN_X + bx_off,
           BLK_ORIGIN_Y + by_off,
           BLK_INIT_ANG + ba_off]
    obs, info = env.reset(options={"reset_to_state": vec})
    return obs, info

# ── Scripted expert ────────────────────────────────────────────────────────────
def _walk_to(env, obs, target, max_steps=150):
    """Move agent to pixel target. Yields (s, a_norm, s_next) transitions."""
    transitions, max_cov = [], 0.0
    target = np.array(target, dtype=np.float64)
    for _ in range(max_steps):
        if np.linalg.norm(obs[:2] - target) < 8:
            break
        action = np.clip(target, 0, IMG).astype(np.float32)
        next_obs, reward, done, truncated, _info = env.step(action)
        max_cov = max(max_cov, float(reward))
        transitions.append((raw_to_state(obs),
                             action / IMG,
                             raw_to_state(next_obs)))
        obs = next_obs
        if done or truncated:
            break
    return obs, transitions, max_cov

def run_expert_episode(env, mode='LEFT'):
    """
    Route AROUND the block via the bottom of the arena, then push upward.
    Block starts below goal (CoM ≈ 288,363), goal is above (CoM ≈ 288,269).
    The T-block dynamics require a lateral x-offset on approach so that
    the push translates the block upward without excessive sideways drift:
      LEFT  mode: detour x=80,  approach x_off=+20 (right of CoM)
      RIGHT mode: detour x=430, approach x_off=−10 (left  of CoM)
    """
    obs, _ = reset_env(
        env,
        agent_x = 256 + np.random.uniform(-8, 8),
        agent_y = 50  + np.random.uniform(-5, 5),
        bx_off  = np.random.uniform(-8, 8),
        by_off  = np.random.uniform(-8, 8),
        ba_off  = np.random.uniform(-0.1, 0.1),
    )

    block_com = obs[2:4].copy()         # observed CoM (pixels)
    detour_x  = 80.0 if mode == 'LEFT' else 430.0
    x_off     = +20  if mode == 'LEFT' else -10    # calibrated lateral offset
    ax        = block_com[0] + x_off    # approach x
    ay        = block_com[1] + 85       # approach y (85 px below CoM)
    push_y    = 250.0                   # push target y (above goal CoM ≈ 269)
    bottom_y  = 490.0                   # routing depth (well below block)

    all_tr, total_cov = [], 0.0
    phases = [
        np.array([detour_x, 50.0]),      # 1. top of detour column
        np.array([detour_x, bottom_y]),  # 2. descend in safe column
        np.array([ax, bottom_y]),        # 3. move below block to approach x
        np.array([ax, ay]),              # 4. rise to approach position
        np.array([ax, push_y]),          # 5. push upward through goal
    ]
    for tgt in phases:
        obs, tr, cov = _walk_to(env, obs, tgt)
        all_tr.extend(tr)
        total_cov = max(total_cov, cov)

    return all_tr, total_cov

def run_failure_episode(env):
    """
    Failure mode: agent starts directly above block CoM and descends straight
    down, pushing block further DOWN (away from goal at y=256).
    Goal is ABOVE initial block (block CoM≈363), so pushing DOWN = failure.
    Labelled H_FAILURE.
    """
    obs, _ = reset_env(
        env,
        agent_x = 288 + np.random.uniform(-8, 8),   # above block CoM x
        agent_y = 50  + np.random.uniform(-5, 5),
        bx_off  = np.random.uniform(-5, 5),
        by_off  = np.random.uniform(-5, 5),
        ba_off  = np.random.uniform(-0.05, 0.05),
    )
    target = np.array([obs[0], 490.0])               # descend toward bottom
    obs, transitions, max_cov = _walk_to(env, obs, target, max_steps=300)
    return transitions, max_cov

# ── Demo generation ────────────────────────────────────────────────────────────
def generate_demos(n_success=200, n_failure=100):
    print("[1] Generating success demonstrations...")
    env = make_env()
    success_demos, left_n, right_n = [], 0, 0
    modes = (['LEFT']  * (n_success // 2) +
             ['RIGHT'] * (n_success - n_success // 2))
    np.random.shuffle(modes)

    for mode in modes:
        tr, cov = run_expert_episode(env, mode)
        if cov > 0.15:                         # discard degenerate episodes
            success_demos.append((tr, cov))
            if mode == 'LEFT': left_n += 1
            else:              right_n += 1

    env.close()
    avg_cov = np.mean([c for _, c in success_demos])
    print(f"  {len(success_demos)} demos  "
          f"(LEFT={left_n}, RIGHT={right_n}, avg_cov={avg_cov:.3f})")

    print("[2] Generating failure demonstrations...")
    env = make_env()
    failure_demos = []
    for _ in range(n_failure):
        tr, _ = run_failure_episode(env)
        if tr:
            failure_demos.append(tr)
    env.close()
    total_fail_tr = sum(len(t) for t in failure_demos)
    print(f"  {len(failure_demos)} failure eps  ({total_fail_tr} transitions)")

    return success_demos, failure_demos

# ── Dataset building ───────────────────────────────────────────────────────────
def build_datasets(success_demos, failure_demos):
    print("[3] Building datasets...")
    S, A, SN, H_labels = [], [], [], []
    for demo_tr, _ in success_demos:
        T = len(demo_tr)
        for t_idx, (s, a, sn) in enumerate(demo_tr):
            S.append(s); A.append(a); SN.append(sn)
            H_labels.append(math.log(T - t_idx + 1))

    S  = torch.tensor(S,  dtype=torch.float32)
    A  = torch.tensor(A,  dtype=torch.float32)
    SN = torch.tensor(SN, dtype=torch.float32)
    H  = torch.tensor(H_labels, dtype=torch.float32)
    G  = torch.tensor(np.tile(GOAL_VEC, (len(S), 1)), dtype=torch.float32)

    # Pre-sample flow noise (paired with each success transition)
    NOISE = torch.randn_like(A)

    # Failure: (s, a, s_next) for dynamics + (s) for entropy
    SF_list, HF_list = [], []
    SF_s, SF_a, SF_sn = [], [], []
    for tr in failure_demos:
        for s, a, sn in tr:
            SF_list.append(s); HF_list.append(H_FAILURE)
            SF_s.append(s); SF_a.append(a); SF_sn.append(sn)
    SF   = torch.tensor(SF_list, dtype=torch.float32)  # for entropy
    HF   = torch.tensor(HF_list, dtype=torch.float32)
    GF   = torch.tensor(np.tile(GOAL_VEC, (len(SF), 1)), dtype=torch.float32)
    SF_S = torch.tensor(SF_s,  dtype=torch.float32)    # for dynamics
    SF_A = torch.tensor(SF_a,  dtype=torch.float32)
    SF_SN= torch.tensor(SF_sn, dtype=torch.float32)

    print(f"  Success: {len(S)} transitions")
    print(f"  Failure: {len(SF)} transitions  (H_FAILURE={H_FAILURE:.2f})")
    return S, A, SN, H, G, NOISE, SF, HF, GF, SF_S, SF_A, SF_SN

# ── Training ───────────────────────────────────────────────────────────────────
def _loader(ds, bs=256):
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

def train_entropy(net, S, H, G, SF, HF, GF,
                  use_failure=True, epochs=80, lr=3e-4, bs=256):
    opt = optim.Adam(net.parameters(), lr=lr)
    ds_s = TensorDataset(torch.cat([S, G], -1), H)
    ld_s = _loader(ds_s, bs)

    if use_failure and len(SF):
        ds_f  = TensorDataset(torch.cat([SF, GF], -1), HF)
        ld_f  = _loader(ds_f, bs)
        f_it  = iter(ld_f)

    for ep in range(1, epochs + 1):
        total = 0.0
        for sg, h in ld_s:
            pred_s = net(sg)
            loss   = ((pred_s - h) ** 2).mean()

            if use_failure and len(SF):
                try:
                    sgf, hf = next(f_it)
                except StopIteration:
                    f_it = iter(ld_f); sgf, hf = next(f_it)
                pred_f   = net(sgf)
                loss_f   = ((pred_f - hf) ** 2).mean()
                # Margin: failure prediction > max success prediction + 1
                margin   = torch.clamp(pred_s.detach().max() + 1.0 - pred_f, min=0).mean()
                loss     = loss + loss_f + 0.1 * margin

            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 40 == 0:
            print(f"  Entropy {ep}/{epochs}: {total/len(ld_s):.4f}")

def train_dynamics(net, S, A, SN, epochs=40, lr=3e-4, bs=256):
    opt = optim.Adam(net.parameters(), lr=lr)
    ld  = _loader(TensorDataset(S, A, SN), bs)
    for ep in range(1, epochs + 1):
        total = 0.0
        for s, a, sn in ld:
            loss = ((net(s, a) - sn) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 10 == 0:
            print(f"  Dynamics {ep}/{epochs}: {total/len(ld):.4f}")

def train_flow(net, S, G, A, NOISE, epochs=150, lr=3e-4, bs=256):
    opt = optim.Adam(net.parameters(), lr=lr)
    ld  = _loader(TensorDataset(S, G, A, NOISE), bs)
    for ep in range(1, epochs + 1):
        total = 0.0
        for s, g, a_clean, noise in ld:
            B = s.shape[0]
            t_val = torch.rand(B)
            t_bc  = t_val.unsqueeze(-1)
            a_t   = (1 - t_bc) * noise + t_bc * a_clean
            v_tgt = a_clean - noise
            te    = t_embed(t_val)
            loss  = ((net(s, g, a_t, te) - v_tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 60 == 0:
            print(f"  Flow {ep}/{epochs}: {total/len(ld):.4f}")

def train_bc(net, S, G, A, epochs=80, lr=3e-4, bs=256):
    opt = optim.Adam(net.parameters(), lr=lr)
    ld  = _loader(TensorDataset(torch.cat([S, G], -1), A), bs)
    for ep in range(1, epochs + 1):
        total = 0.0
        for sg, a in ld:
            loss = ((net(sg) - a) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 20 == 0:
            print(f"  BC {ep}/{epochs}: {total/len(ld):.4f}")

def mdn_nll(pi, mu, sigma, target):
    """NLL for K-component Gaussian mixture over ACT_DIM-D actions."""
    t = target.unsqueeze(1).expand_as(mu)              # (B, K, 2)
    log_p = (-0.5 * ((t - mu) / sigma).pow(2).sum(-1)  # (B, K)
             - sigma.log().sum(-1)
             - ACT_DIM * 0.5 * math.log(2 * math.pi))
    return -torch.logsumexp(log_p + pi.log(), dim=-1).mean()

def train_mdn(net, S, G, A, epochs=80, lr=3e-4, bs=256):
    opt = optim.Adam(net.parameters(), lr=lr)
    ld  = _loader(TensorDataset(torch.cat([S, G], -1), A), bs)
    for ep in range(1, epochs + 1):
        total = 0.0
        for sg, a in ld:
            pi, mu, sigma = net(sg)
            loss = mdn_nll(pi, mu, sigma, a)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 20 == 0:
            print(f"  MDN {ep}/{epochs}: {total/len(ld):.4f}")

# ── Inference ──────────────────────────────────────────────────────────────────
GUIDANCE_SCALE = 1.5   # entropy gradient step size per ODE step

def _infer_flow(state_np, vel_net, entropy_net=None, dyn_net=None, guidance=0.0):
    """Run flow ODE (10 Euler steps). With guidance>0 adds entropy gradient."""
    s = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)  # (1,6)
    g = torch.tensor(GOAL_VEC,  dtype=torch.float32).unsqueeze(0)  # (1,4)
    a = torch.randn(1, ACT_DIM)
    dt = 1.0 / N_FLOW_STEPS

    vel_net.eval()
    if entropy_net is not None: entropy_net.eval()
    if dyn_net     is not None: dyn_net.eval()

    for i in range(N_FLOW_STEPS):
        t_val = i / N_FLOW_STEPS
        te    = t_embed(torch.tensor([t_val]))              # (1, T_DIM)
        with torch.no_grad():
            v = vel_net(s, g, a, te)

        if guidance > 0 and entropy_net is not None and dyn_net is not None:
            with torch.enable_grad():
                a_d    = a.detach().requires_grad_(True)
                s_next = dyn_net(s, a_d)
                h_val  = entropy_net(torch.cat([s_next, g], dim=-1))
                h_val.backward()
            grad     = a_d.grad.data
            gnorm    = grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            grad_n   = grad / gnorm
            a = (a.detach() + v.detach() * dt - guidance * grad_n * dt)
        else:
            a = (a.detach() + v.detach() * dt)

    action_norm = a.squeeze(0).detach().numpy()
    return np.clip(action_norm * IMG, 0, IMG).astype(np.float32)

@torch.no_grad()
def _infer_bc(state_np, bc_net):
    bc_net.eval()
    s  = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
    g  = torch.tensor(GOAL_VEC,  dtype=torch.float32).unsqueeze(0)
    a  = bc_net(torch.cat([s, g], dim=-1)).squeeze(0).numpy()
    return np.clip(a * IMG, 0, IMG).astype(np.float32)

@torch.no_grad()
def _infer_mdn(state_np, mdn_net):
    mdn_net.eval()
    s       = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
    g       = torch.tensor(GOAL_VEC,  dtype=torch.float32).unsqueeze(0)
    pi, mu, sigma = mdn_net(torch.cat([s, g], dim=-1))
    k       = torch.multinomial(pi, 1).item()
    a       = (mu[0, k] + sigma[0, k] * torch.randn(ACT_DIM)).numpy()
    return np.clip(a * IMG, 0, IMG).astype(np.float32)

# ── Evaluation ─────────────────────────────────────────────────────────────────
_rng_eval = np.random.RandomState(99)

# Standard: agent at top-centre (256, 50), block at standard starting position
EVAL_STD = [
    (256 + _rng_eval.uniform(-5, 5),   # agent_x
     50  + _rng_eval.uniform(-3, 3),   # agent_y
     _rng_eval.uniform(-8, 8),         # bx_off
     _rng_eval.uniform(-8, 8),         # by_off
     _rng_eval.uniform(-0.1, 0.1))     # ba_off
    for _ in range(25)
]

# Perturbation: agent starts ABOVE the block (288, 280) — challenging because
# going straight down (toward goal direction) pushes block further from goal.
# EFP w/ failure supervision should avoid this failure mode via entropy gradient.
EVAL_PERTURB = [
    (288 + _rng_eval.uniform(-8, 8),   # agent_x (above block CoM x=288)
     280 + _rng_eval.uniform(-10, 10), # agent_y (above block CoM y≈363)
     _rng_eval.uniform(-8, 8),         # bx_off
     _rng_eval.uniform(-8, 8),         # by_off
     _rng_eval.uniform(-0.1, 0.1))     # ba_off
    for _ in range(25)
]

def evaluate(policy_fn, label, eval_set=EVAL_STD, max_steps=350):
    env = make_env()
    coverages, successes = [], []
    for init in eval_set:
        ax, ay, bx, by, ba = init
        obs, _ = reset_env(env, agent_x=ax, agent_y=ay,
                           bx_off=bx, by_off=by, ba_off=ba)
        max_cov = 0.0
        for _ in range(max_steps):
            action = policy_fn(raw_to_state(obs))
            obs, reward, done, truncated, _info = env.step(action)
            max_cov = max(max_cov, float(reward))
            if done or truncated:
                break
        coverages.append(max_cov)
        successes.append(float(max_cov > SUCCESS_THR))
    env.close()
    mean_cov  = np.mean(coverages)
    mean_succ = np.mean(successes)
    print(f"    {label:<24} coverage={mean_cov:.3f}  "
          f"success={int(mean_succ*100)}%")
    return mean_cov, mean_succ

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ENTROPIC FLOW POLICY — Push-T Benchmark")
    print("  EFP (w/ failure) | EFP (no failure) | DP | BC | MDN-BC")
    print("=" * 60)
    print()

    # ── 1-2. Data ──────────────────────────────────────────────────────────────
    success_demos, failure_demos = generate_demos(n_success=300, n_failure=120)
    S, A, SN, H_lab, G, NOISE, SF, HF, GF, SF_S, SF_A, SF_SN = build_datasets(
        success_demos, failure_demos)

    # ── 3. Entropy estimators ──────────────────────────────────────────────────
    print("[4] Training entropy estimator (with failure supervision)...")
    enet_full = EntropyNet()
    train_entropy(enet_full, S, H_lab, G, SF, HF, GF,
                  use_failure=True, epochs=120)

    print("[5] Training entropy estimator (no failure — ablation)...")
    enet_nofail = EntropyNet()
    train_entropy(enet_nofail, S, H_lab, G, SF, HF, GF,
                  use_failure=False, epochs=120)

    # ── 4. Dynamics (success + failure for broader coverage) ──────────────────
    print("[6] Training dynamics model...")
    dyn_net = DynNet()
    # Combine success + failure transitions so dynamics model predicts both
    # "push toward goal" and "push away from goal" outcomes
    S_dyn  = torch.cat([S,    SF_S],  dim=0)
    A_dyn  = torch.cat([A,    SF_A],  dim=0)
    SN_dyn = torch.cat([SN,   SF_SN], dim=0)
    train_dynamics(dyn_net, S_dyn, A_dyn, SN_dyn, epochs=40)

    # ── 5. Flow matching (shared VelocityNet) ──────────────────────────────────
    print("[7] Training flow matching (VelocityNet — shared)...")
    vel_net = VelocityNet()
    train_flow(vel_net, S, G, A, NOISE, epochs=300)

    # ── 6. BC ──────────────────────────────────────────────────────────────────
    print("[8] Training BC...")
    bc_net = BCNet()
    train_bc(bc_net, S, G, A, epochs=80)

    # ── 7. MDN-BC ──────────────────────────────────────────────────────────────
    print("[9] Training MDN-BC...")
    mdn_net = MDNNet()
    train_mdn(mdn_net, S, G, A, epochs=80)

    # ── 7b. Save models ────────────────────────────────────────────────────────
    _mdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(_mdir, exist_ok=True)
    torch.save(enet_full.state_dict(),   f'{_mdir}/enet_full.pt')
    torch.save(enet_nofail.state_dict(), f'{_mdir}/enet_nofail.pt')
    torch.save(dyn_net.state_dict(),     f'{_mdir}/dyn_net.pt')
    torch.save(vel_net.state_dict(),     f'{_mdir}/vel_net.pt')
    torch.save(bc_net.state_dict(),      f'{_mdir}/bc_net.pt')
    torch.save(mdn_net.state_dict(),     f'{_mdir}/mdn_net.pt')
    print(f"[models saved → {_mdir}/]")

    # ── 8. Evaluation ──────────────────────────────────────────────────────────
    print("\n[10] Evaluation...")

    # Build policy functions (capture nets in closures)
    _vel, _ef, _en, _dyn, _bc, _mdn = vel_net, enet_full, enet_nofail, dyn_net, bc_net, mdn_net
    _G  = GUIDANCE_SCALE
    policies = [
        ("EFP (w/ failure)",
         lambda s: _infer_flow(s, _vel, _ef,   _dyn, guidance=_G)),
        ("EFP (no failure)",
         lambda s: _infer_flow(s, _vel, _en,   _dyn, guidance=_G)),
        ("Diffusion Policy",
         lambda s: _infer_flow(s, _vel, guidance=0.0)),
        ("BC (Mean)",
         lambda s: _infer_bc(s, _bc)),
        ("MDN-BC",
         lambda s: _infer_mdn(s, _mdn)),
    ]

    results_std    = {}
    results_perturb = {}

    print("\n  === Standard (agent starts top-centre) ===")
    for label, fn in policies:
        cov, succ = evaluate(fn, label, eval_set=EVAL_STD)
        results_std[label] = (cov, succ)

    print("\n  === Perturb (agent starts above block, pushing DOWN = failure) ===")
    for label, fn in policies:
        cov, succ = evaluate(fn, label, eval_set=EVAL_PERTURB)
        results_perturb[label] = (cov, succ)

    # ── 9. Plot ────────────────────────────────────────────────────────────────
    print("\n[11] Plotting...")
    labels = [l for l, _ in policies]
    colors = ['#2ecc71', '#27ae60', '#3498db', '#e74c3c', '#9b59b6']
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (results, title_sfx) in enumerate([
        (results_std,    'Standard'),
        (results_perturb, 'Perturb (agent above block)'),
    ]):
        covs  = [results[l][0] for l in labels]
        succs = [results[l][1] for l in labels]

        axes[row, 0].bar(x, covs, color=colors, edgecolor='black', linewidth=0.8)
        axes[row, 0].axhline(SUCCESS_THR, color='grey', linestyle='--', linewidth=1,
                             label=f'threshold={SUCCESS_THR}')
        axes[row, 0].set_xticks(x); axes[row, 0].set_xticklabels(labels, rotation=18, ha='right')
        axes[row, 0].set_ylim(0, 0.7); axes[row, 0].set_ylabel('Mean Max Coverage')
        axes[row, 0].set_title(f'[{title_sfx}] Coverage')
        axes[row, 0].legend(fontsize=8)

        axes[row, 1].bar(x, [s*100 for s in succs], color=colors, edgecolor='black', linewidth=0.8)
        axes[row, 1].set_xticks(x); axes[row, 1].set_xticklabels(labels, rotation=18, ha='right')
        axes[row, 1].set_ylim(0, 105); axes[row, 1].set_ylabel(f'Success % [cov>{SUCCESS_THR}]')
        axes[row, 1].set_title(f'[{title_sfx}] Success Rate')

    plt.suptitle('Entropic Flow Policy — Push-T Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = '/home/sakib/working_dir/entropyFlowPolicy/efp_pusht_results.png'
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")

if __name__ == '__main__':
    main()
