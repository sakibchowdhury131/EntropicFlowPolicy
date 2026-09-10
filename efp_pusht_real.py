"""
efp_pusht_real.py — EFP vs original Diffusion Policy on the official Push-T
human-demonstration dataset (pusht_cchi_v7_replay.zarr.zip).

All models are trained on the same 206 human-demo episodes (25,650
transitions).  EFP additionally trains on synthetically generated failure
demonstrations.  Evaluation re-uses the Standard / Perturb conditions from
efp_pusht.py for a direct apples-to-apples comparison.

Methods
-------
  EFP (w/ failure) : flow ODE + entropy gradient, failure-supervised Hθ
  EFP (no failure) : same, but Hθ trained on success demos only
  Diffusion Policy : ConditionalUnet1D + DDPM (original notebook design)
  BC (Mean)        : deterministic MLP
  MDN-BC           : Gaussian mixture BC

Usage
-----
  python efp_pusht_real.py              # full training (default)
  python efp_pusht_real.py --dp_epochs 50   # faster DP training
"""

import math, os, collections, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import zarr.storage
import zarr
import gymnasium as gym
import gym_pusht          # noqa: F401
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from efp_pusht import (
    EntropyNet, DynNet, BCNet, MDNNet,
    _infer_bc, _infer_mdn,
    raw_to_state, reset_env, make_env,
    train_entropy, train_dynamics, train_bc, train_mdn,
    GOAL_VEC, H_FAILURE, IMG, ACT_DIM, STATE_DIM, GOAL_DIM,
    SUCCESS_THR,
    EVAL_STD, EVAL_PERTURB,
    run_failure_episode,
)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
ZARR_PATH  = os.path.join(BASE, 'pusht_cchi_v7_replay.zarr.zip')
MODELS_DIR = os.path.join(BASE, 'models_real')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DP hyperparams (from original notebook)
OBS_HORIZON    = 2
PRED_HORIZON   = 16
ACTION_HORIZON = 8
NUM_DDPM_ITERS = 100
DP_BS          = 256
DP_LR          = 1e-4
DP_WEIGHT_DECAY = 1e-6

# EFP-UNet hyperparams
EFP_N_FLOW_STEPS = 10   # Euler ODE steps at inference
EFP_GUIDANCE     = 0.3  # entropy gradient scale

# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_zarr():
    store = zarr.storage.ZipStore(ZARR_PATH, mode='r')
    ds    = zarr.open(store=store, mode='r')
    obs   = ds['data']['state'][:]          # (N, 5)
    act   = ds['data']['action'][:]         # (N, 2)
    ends  = ds['meta']['episode_ends'][:]   # (n_eps,)
    return obs, act, ends


def get_stats(obs_raw, act_raw):
    """Min-max stats for DP normalisation, computed from dataset."""
    def _stats(arr):
        arr = arr.reshape(-1, arr.shape[-1])
        return {'min': arr.min(0), 'max': arr.max(0)}
    return {'obs': _stats(obs_raw), 'action': _stats(act_raw)}


def norm(data, stats):
    """Normalise to [-1, 1] using pre-computed stats."""
    ndata = (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
    return (ndata * 2 - 1).astype(np.float32)


def unnorm(ndata, stats):
    """Inverse of norm()."""
    return ((ndata + 1) / 2) * (stats['max'] - stats['min']) + stats['min']


def ep_boundaries(ends):
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts, ends))


def build_efp_tensors(obs_raw, act_raw, ends):
    """
    Extract single-step (s6d, a_norm, s_next_6d, H_label) for EFP training.
    """
    S, A, SN, Hl = [], [], [], []
    for ep_s, ep_e in ep_boundaries(ends):
        T = ep_e - ep_s
        for t in range(T - 1):
            i = ep_s + t
            s  = raw_to_state(obs_raw[i])
            a  = (act_raw[i] / IMG).astype(np.float32)
            sn = raw_to_state(obs_raw[i + 1])
            h  = math.log(T - t)
            S.append(s); A.append(a); SN.append(sn); Hl.append(h)

    S  = torch.tensor(np.stack(S),  dtype=torch.float32)
    A  = torch.tensor(np.stack(A),  dtype=torch.float32)
    SN = torch.tensor(np.stack(SN), dtype=torch.float32)
    H  = torch.tensor(Hl,           dtype=torch.float32)
    G  = torch.tensor(np.tile(GOAL_VEC, (len(S), 1)), dtype=torch.float32)
    return S, A, SN, H, G


def add_failure_tensors(n_failure=120):
    """Generate synthetic failure demonstrations using gym_pusht."""
    print(f"[failure] Generating {n_failure} failure episodes...")
    env = make_env()
    SF_s, SF_a, SF_sn = [], [], []
    eps = 0
    while eps < n_failure:
        tr, _ = run_failure_episode(env)
        if tr:
            for s, a, sn in tr:
                SF_s.append(s); SF_a.append(a); SF_sn.append(sn)
            eps += 1
    env.close()
    SF_S = torch.tensor(np.stack(SF_s),  dtype=torch.float32)
    SF_A = torch.tensor(np.stack(SF_a),  dtype=torch.float32)
    SF_SN= torch.tensor(np.stack(SF_sn), dtype=torch.float32)
    SF   = SF_S.clone()
    HF   = torch.full((len(SF_S),), H_FAILURE)
    GF   = torch.tensor(np.tile(GOAL_VEC, (len(SF), 1)), dtype=torch.float32)
    print(f"  {len(SF_s)} failure transitions  (H_FAILURE={H_FAILURE:.2f})")
    return SF, HF, GF, SF_S, SF_A, SF_SN


# ── Diffusion Policy model (from original notebook) ───────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half  = self.dim // 2
        emb   = math.log(10000) / (half - 1)
        emb   = torch.exp(torch.arange(half, device=device) * -emb)
        emb   = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, inp, out, ks, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, ks, padding=ks // 2),
            nn.GroupNorm(n_groups, out),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)


class CondResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, cond_dim, ks=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_c, out_c, ks, n_groups),
            Conv1dBlock(out_c, out_c, ks, n_groups),
        ])
        self.out_c = out_c
        self.cond_enc = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_c * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.res_conv = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, cond):
        out   = self.blocks[0](x)
        emb   = self.cond_enc(cond).reshape(cond.shape[0], 2, self.out_c, 1)
        out   = emb[:, 0] * out + emb[:, 1]
        out   = self.blocks[1](out)
        return out + self.res_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim,
                 dsed=256, down_dims=(256, 512, 1024), ks=5, n_groups=8):
        super().__init__()
        all_dims  = [input_dim] + list(down_dims)
        cond_dim  = dsed + global_cond_dim
        in_out    = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim   = all_dims[-1]

        self.diff_emb = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed),
        )
        self.mid = nn.ModuleList([
            CondResBlock1D(mid_dim, mid_dim, cond_dim, ks, n_groups),
            CondResBlock1D(mid_dim, mid_dim, cond_dim, ks, n_groups),
        ])
        self.down = nn.ModuleList()
        for i, (di, do) in enumerate(in_out):
            is_last = i >= len(in_out) - 1
            self.down.append(nn.ModuleList([
                CondResBlock1D(di, do, cond_dim, ks, n_groups),
                CondResBlock1D(do, do, cond_dim, ks, n_groups),
                nn.Conv1d(do, do, 3, 2, 1) if not is_last else nn.Identity(),
            ]))
        self.up = nn.ModuleList()
        for i, (di, do) in enumerate(reversed(in_out[1:])):
            is_last = i >= len(in_out) - 1
            self.up.append(nn.ModuleList([
                CondResBlock1D(do * 2, di, cond_dim, ks, n_groups),
                CondResBlock1D(di, di, cond_dim, ks, n_groups),
                nn.ConvTranspose1d(di, di, 4, 2, 1) if not is_last else nn.Identity(),
            ]))
        self.final = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], ks),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )
        n = sum(p.numel() for p in self.parameters())
        print(f"  ConditionalUnet1D: {n/1e6:.1f}M parameters")

    def forward(self, sample, timestep, global_cond=None):
        x  = sample.moveaxis(-1, -2)     # (B, C, T)
        ts = torch.as_tensor(timestep, device=x.device).expand(x.shape[0]).long()
        gf = self.diff_emb(ts)
        if global_cond is not None:
            gf = torch.cat([gf, global_cond], -1)
        h  = []
        for res1, res2, ds in self.down:
            x = res2(res1(x, gf), gf); h.append(x); x = ds(x)
        for m in self.mid:
            x = m(x, gf)
        for res1, res2, us in self.up:
            x = res2(res1(torch.cat([x, h.pop()], 1), gf), gf); x = us(x)
        return self.final(x).moveaxis(-1, -2)   # (B, T, C)


# ── DP Dataset ─────────────────────────────────────────────────────────────────

class DPDataset(Dataset):
    """Sliding-window (nobs_seq, naction_seq) pairs for DDPM training."""
    def __init__(self, obs_raw, act_raw, ends, stats):
        self.samples = []
        for ep_s, ep_e in ep_boundaries(ends):
            T       = ep_e - ep_s
            nobs    = norm(obs_raw[ep_s:ep_e], stats['obs'])
            nact    = norm(act_raw[ep_s:ep_e], stats['action'])
            for t in range(T - 1):
                # obs window: pad left with first obs if needed
                o_start = max(0, t - OBS_HORIZON + 1)
                oc = nobs[o_start:t + 1]
                if len(oc) < OBS_HORIZON:
                    oc = np.concatenate([np.repeat(oc[:1], OBS_HORIZON - len(oc), 0), oc])
                # action window: pad right with last action if needed
                ac = nact[t:min(T, t + PRED_HORIZON)]
                if len(ac) < PRED_HORIZON:
                    ac = np.concatenate([ac, np.repeat(ac[-1:], PRED_HORIZON - len(ac), 0)])
                self.samples.append((oc.astype(np.float32), ac.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        o, a = self.samples[idx]
        return torch.from_numpy(o), torch.from_numpy(a)


# ── EFP-UNet Dataset ───────────────────────────────────────────────────────────

class EFPSequenceDataset(Dataset):
    """Sliding-window (obs_6d_seq, act_norm_seq) pairs for UNet CFM training.

    Unlike DPDataset (which uses raw 5D obs + min-max norm), this uses 6D
    states from raw_to_state() and /512 action normalisation to stay
    consistent with the EFP entropy / dynamics models.
    """
    def __init__(self, obs_raw, act_raw, ends):
        self.samples = []
        for ep_s, ep_e in ep_boundaries(ends):
            T      = ep_e - ep_s
            states = np.stack([raw_to_state(obs_raw[ep_s + t]) for t in range(T)])
            acts   = (act_raw[ep_s:ep_e] / IMG).astype(np.float32)
            for t in range(T - 1):
                o_start = max(0, t - OBS_HORIZON + 1)
                oc = states[o_start:t + 1]
                if len(oc) < OBS_HORIZON:
                    oc = np.concatenate(
                        [np.repeat(oc[:1], OBS_HORIZON - len(oc), 0), oc])
                ac = acts[t:min(T, t + PRED_HORIZON)]
                if len(ac) < PRED_HORIZON:
                    ac = np.concatenate(
                        [ac, np.repeat(ac[-1:], PRED_HORIZON - len(ac), 0)])
                self.samples.append((oc.astype(np.float32), ac.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        o, a = self.samples[idx]
        return torch.from_numpy(o), torch.from_numpy(a)


# ── DP Training ────────────────────────────────────────────────────────────────

def train_dp(model, dp_dataset, num_epochs=100, lr=DP_LR, wd=DP_WEIGHT_DECAY):
    loader = DataLoader(dp_dataset, batch_size=DP_BS, shuffle=True,
                        num_workers=0, drop_last=False)
    scheduler = DDPMScheduler(
        num_train_timesteps=NUM_DDPM_ITERS,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    ema = EMAModel(parameters=model.parameters(), power=0.75)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lr_sched = get_scheduler(
        'cosine', optimizer=opt,
        num_warmup_steps=500,
        num_training_steps=len(loader) * num_epochs,
    )
    model.train()
    for ep in range(1, num_epochs + 1):
        total = 0.0
        for nobs, nact in loader:
            nobs = nobs.to(DEVICE)          # (B, obs_horizon, 5)
            nact = nact.to(DEVICE)          # (B, pred_horizon, 2)
            B    = nobs.shape[0]
            obs_cond = nobs.flatten(start_dim=1)   # (B, obs_horizon*5)
            noise     = torch.randn_like(nact)
            t_samp    = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (B,), device=DEVICE).long()
            noisy     = scheduler.add_noise(nact, noise, t_samp)
            pred      = model(noisy, t_samp, global_cond=obs_cond)
            loss      = nn.functional.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step(); lr_sched.step()
            ema.step(model.parameters())
            total += loss.item()
        if ep % max(1, num_epochs // 5) == 0 or ep == num_epochs:
            print(f"  DP {ep}/{num_epochs}: {total/len(loader):.4f}")
    ema.copy_to(model.parameters())
    return model


# ── EFP-UNet Training (Conditional Flow Matching) ─────────────────────────────

def train_efp_unet_flow(net, dataset, epochs=300, lr=1e-4, wd=1e-6):
    """Train UNet velocity field with CFM loss on action sequences.

    CFM interpolation: a_t = (1-t)*noise + t*a_clean
    Velocity target:   v = a_clean - noise
    t is sampled uniformly in [0,1] and mapped to [0,999] for UNet's
    SinusoidalPosEmb (same integer range used by DDPM's timestep emb).
    """
    loader = DataLoader(dataset, batch_size=DP_BS, shuffle=True,
                        num_workers=0, drop_last=False)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lr_sched = get_scheduler(
        'cosine', optimizer=opt,
        num_warmup_steps=500,
        num_training_steps=len(loader) * epochs,
    )
    net.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for obs_seq, act_clean in loader:
            obs_seq   = obs_seq.to(DEVICE)     # (B, OBS_HORIZON, STATE_DIM)
            act_clean = act_clean.to(DEVICE)   # (B, PRED_HORIZON, ACT_DIM)
            B = obs_seq.shape[0]
            obs_cond = obs_seq.flatten(start_dim=1)          # (B, OBS_HORIZON*STATE_DIM)
            noise    = torch.randn_like(act_clean)
            t_cont   = torch.rand(B, device=DEVICE)          # (B,) uniform [0,1]
            t_int    = (t_cont * 999).long()                 # integer timestep for UNet
            t_exp    = t_cont[:, None, None]                 # (B,1,1) for broadcasting
            a_t      = (1 - t_exp) * noise + t_exp * act_clean
            v_tgt    = act_clean - noise
            pred_v   = net(a_t, t_int, global_cond=obs_cond)
            loss     = nn.functional.mse_loss(pred_v, v_tgt)
            opt.zero_grad(); loss.backward(); opt.step(); lr_sched.step()
            total += loss.item()
        if ep % max(1, epochs // 5) == 0 or ep == epochs:
            print(f"  Flow {ep}/{epochs}: {total/len(loader):.4f}")
    return net


# ── DP Inference (stateful — maintains obs queue + action chunk) ───────────────

class DPPolicy:
    """Wraps the trained DP model with obs-stacking and action chunking."""
    def __init__(self, model, stats, num_ddpm_iters=NUM_DDPM_ITERS):
        self.model  = model.eval()
        self.stats  = stats
        self.sched  = DDPMScheduler(
            num_train_timesteps=num_ddpm_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True, prediction_type='epsilon',
        )
        self.obs_deque  = None
        self.action_buf = None
        self.buf_idx    = 0

    def reset(self, obs_raw):
        self.obs_deque  = collections.deque(
            [obs_raw.copy()] * OBS_HORIZON, maxlen=OBS_HORIZON)
        self.action_buf = None
        self.buf_idx    = 0

    def __call__(self, obs_raw):
        self.obs_deque.append(obs_raw.copy())

        if self.action_buf is None or self.buf_idx >= ACTION_HORIZON:
            obs_seq   = np.stack(self.obs_deque)                  # (2, 5)
            nobs      = norm(obs_seq, self.stats['obs'])           # (2, 5)
            nobs_t    = torch.from_numpy(nobs).unsqueeze(0).to(DEVICE)  # (1,2,5)
            obs_cond  = nobs_t.flatten(start_dim=1)               # (1,10)

            with torch.no_grad():
                nact = torch.randn(1, PRED_HORIZON, ACT_DIM, device=DEVICE)
                self.sched.set_timesteps(NUM_DDPM_ITERS)
                for k in self.sched.timesteps:
                    pred = self.model(nact, k, global_cond=obs_cond)
                    nact = self.sched.step(pred, k, nact).prev_sample
                nact_np = nact[0].cpu().numpy()   # (pred_horizon, 2)

            # unnormalise; dataset has actions from t=0 so no leading skip needed
            act_pred = unnorm(nact_np, self.stats['action'])       # (pred_horizon, 2)
            self.action_buf = act_pred[:ACTION_HORIZON]            # (action_horizon, 2)
            self.buf_idx    = 0

        action = self.action_buf[self.buf_idx].astype(np.float32)
        self.buf_idx += 1
        return np.clip(action, 0, IMG)


# ── EFP-UNet Inference (stateful — obs queue + action chunk + entropy guidance) ─

class EFPUNetPolicy:
    """UNet-based flow policy for EFP.

    Uses Euler ODE with EFP_N_FLOW_STEPS to map noise → action sequence,
    then applies a single entropy-gradient correction to the first action
    before executing ACTION_HORIZON steps before replanning.
    """
    def __init__(self, vel_unet, entropy_net, dyn_net, guidance=EFP_GUIDANCE):
        self.vel_unet   = vel_unet.eval()
        self.enet       = entropy_net.eval() if entropy_net is not None else None
        # dyn_net and enet are kept on CPU (small MLP, fast enough)
        self.dyn        = dyn_net.eval().cpu()
        self.udev       = next(vel_unet.parameters()).device  # device of UNet
        self.guidance   = guidance
        self.obs_deque  = None
        self.action_buf = None
        self.buf_idx    = 0

    def reset(self, obs_raw):
        s0 = raw_to_state(obs_raw)
        self.obs_deque  = collections.deque(
            [s0.copy()] * OBS_HORIZON, maxlen=OBS_HORIZON)
        self.action_buf = None
        self.buf_idx    = 0

    def __call__(self, obs_raw):
        s = raw_to_state(obs_raw)
        self.obs_deque.append(s.copy())

        if self.action_buf is None or self.buf_idx >= ACTION_HORIZON:
            obs_seq  = np.stack(self.obs_deque)               # (OBS_HORIZON, STATE_DIM)
            obs_cond = torch.tensor(
                obs_seq.flatten(), dtype=torch.float32
            ).unsqueeze(0).to(self.udev)                      # (1, OBS_HORIZON*STATE_DIM)

            # Euler ODE on UNet device: noise → clean action sequence
            a  = torch.randn(1, PRED_HORIZON, ACT_DIM, device=self.udev)
            dt = 1.0 / EFP_N_FLOW_STEPS
            self.vel_unet.eval()
            with torch.no_grad():
                for i in range(EFP_N_FLOW_STEPS):
                    t_int = torch.tensor(
                        [int(i / EFP_N_FLOW_STEPS * 999)], device=self.udev)
                    v = self.vel_unet(a, t_int, global_cond=obs_cond)
                    a = a + v * dt

            a_np = a.squeeze(0).detach().cpu().numpy()        # (PRED_HORIZON, ACT_DIM)

            # Entropy guidance on CPU (dyn/enet are CPU models)
            if self.guidance > 0 and self.enet is not None:
                s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)       # (1,6) CPU
                g_t = torch.tensor(GOAL_VEC, dtype=torch.float32).unsqueeze(0)  # (1,4) CPU
                a0  = torch.tensor(a_np[:1], dtype=torch.float32).requires_grad_(True)
                s_next = self.dyn(s_t, a0)
                h_val  = self.enet(torch.cat([s_next, g_t], dim=-1))
                h_val.backward()
                grad   = a0.grad.data
                gnorm  = grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                a_np[0] -= (self.guidance * (grad / gnorm)).squeeze(0).numpy()

            # /512 → pixel space, clip to valid range
            act_pred = np.clip(a_np * IMG, 0, IMG).astype(np.float32)
            self.action_buf = act_pred[:ACTION_HORIZON]
            self.buf_idx    = 0

        action = self.action_buf[self.buf_idx].astype(np.float32)
        self.buf_idx += 1
        return action


# ── Evaluation (supports both stateless and stateful policies) ─────────────────

def evaluate_policy(policy_fn, label, eval_set=EVAL_STD, max_steps=350):
    env = make_env()
    coverages, successes = [], []
    for init in eval_set:
        ax, ay, bx, by, ba = init
        obs, _ = reset_env(env, agent_x=ax, agent_y=ay,
                           bx_off=bx, by_off=by, ba_off=ba)
        if hasattr(policy_fn, 'reset'):
            policy_fn.reset(obs)     # stateful policies need obs-deque initialisation
        max_cov = 0.0
        for _ in range(max_steps):
            action = policy_fn(obs)
            obs, reward, done, truncated, _ = env.step(action)
            max_cov = max(max_cov, float(reward))
            if done or truncated:
                break
        coverages.append(max_cov)
        successes.append(float(max_cov > SUCCESS_THR))
    env.close()
    mc   = np.mean(coverages)
    succ = np.mean(successes)
    print(f"    {label:<26} coverage={mc:.3f}  success={int(succ*100)}%")
    return mc, succ


# ── Model persistence ──────────────────────────────────────────────────────────

def save_all(enet_full, enet_nofail, dyn_net, efp_vel_unet, bc_net, mdn_net, dp_model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, net in [
        ('enet_full',    enet_full),    ('enet_nofail',  enet_nofail),
        ('dyn_net',      dyn_net),      ('efp_vel_unet', efp_vel_unet),
        ('bc_net',       bc_net),       ('mdn_net',      mdn_net),
        ('dp_model',     dp_model),
    ]:
        torch.save(net.state_dict(), f'{MODELS_DIR}/{name}.pt')
    print(f"[saved all models → {MODELS_DIR}/]")


def load_all():
    def _load(cls, name, *args, **kwargs):
        net = cls(*args, **kwargs)
        net.load_state_dict(torch.load(f'{MODELS_DIR}/{name}.pt',
                                       map_location='cpu'))
        return net
    enet_full    = _load(EntropyNet,  'enet_full')
    enet_nofail  = _load(EntropyNet,  'enet_nofail')
    dyn_net      = _load(DynNet,      'dyn_net')
    bc_net       = _load(BCNet,       'bc_net')
    mdn_net      = _load(MDNNet,      'mdn_net')
    efp_vel_unet = ConditionalUnet1D(
        input_dim=ACT_DIM,
        global_cond_dim=OBS_HORIZON * STATE_DIM).to(DEVICE)
    efp_vel_unet.load_state_dict(
        torch.load(f'{MODELS_DIR}/efp_vel_unet.pt', map_location=DEVICE))
    dp_model = ConditionalUnet1D(
        input_dim=ACT_DIM,
        global_cond_dim=OBS_HORIZON * 5).to(DEVICE)
    dp_model.load_state_dict(
        torch.load(f'{MODELS_DIR}/dp_model.pt', map_location=DEVICE))
    print(f"[loaded all models ← {MODELS_DIR}/]")
    return enet_full, enet_nofail, dyn_net, efp_vel_unet, bc_net, mdn_net, dp_model


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(results_std, results_perturb, policy_labels, out_path):
    colors = ['#2ecc71', '#27ae60', '#95a5a6', '#3498db', '#e74c3c', '#9b59b6']
    x = np.arange(len(policy_labels))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, (results, title_sfx) in enumerate([
        (results_std,     'Standard'),
        (results_perturb, 'Perturb (agent above block)'),
    ]):
        covs  = [results[l][0] for l in policy_labels]
        succs = [results[l][1] for l in policy_labels]
        axes[row, 0].bar(x, covs, color=colors, edgecolor='black', linewidth=0.8)
        axes[row, 0].axhline(SUCCESS_THR, color='grey', ls='--', lw=1,
                              label=f'threshold={SUCCESS_THR}')
        axes[row, 0].set_xticks(x)
        axes[row, 0].set_xticklabels(policy_labels, rotation=18, ha='right')
        axes[row, 0].set_ylim(0, 0.7); axes[row, 0].set_ylabel('Mean Max Coverage')
        axes[row, 0].set_title(f'[{title_sfx}] Coverage'); axes[row, 0].legend(fontsize=8)
        axes[row, 1].bar(x, [s * 100 for s in succs], color=colors,
                          edgecolor='black', linewidth=0.8)
        axes[row, 1].set_xticks(x)
        axes[row, 1].set_xticklabels(policy_labels, rotation=18, ha='right')
        axes[row, 1].set_ylim(0, 105); axes[row, 1].set_ylabel(f'Success % [cov>{SUCCESS_THR}]')
        axes[row, 1].set_title(f'[{title_sfx}] Success Rate')
    plt.suptitle('EFP vs DP — Push-T (real human-demo dataset)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot saved → {out_path}]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dp_epochs',    type=int, default=100)
    ap.add_argument('--efp_flow_ep',  type=int, default=300)
    ap.add_argument('--efp_ent_ep',   type=int, default=120)
    ap.add_argument('--n_failure',    type=int, default=120)
    ap.add_argument('--load',         action='store_true',
                    help='skip training, load saved models from models_real/')
    args = ap.parse_args()

    print("=" * 60)
    print("EFP vs DP — Push-T (real human-demo dataset)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    if args.load and os.path.exists(f'{MODELS_DIR}/efp_vel_unet.pt'):
        enet_full, enet_nofail, dyn_net, efp_vel_unet, bc_net, mdn_net, dp_model = load_all()
        obs_raw, act_raw, ends = load_zarr()
        stats = get_stats(obs_raw, act_raw)
    else:
        # ── 1. Load dataset ────────────────────────────────────────────────────
        print("\n[1] Loading zarr dataset...")
        obs_raw, act_raw, ends = load_zarr()
        stats = get_stats(obs_raw, act_raw)
        lengths = np.diff(np.concatenate([[0], ends]))
        print(f"  {len(ends)} episodes, {len(obs_raw)} transitions, "
              f"ep_len={lengths.min()}–{lengths.max()} (mean {lengths.mean():.1f})")

        # ── 2. Build EFP training tensors ──────────────────────────────────────
        print("\n[2] Building EFP dataset from zarr...")
        S, A, SN, H, G = build_efp_tensors(obs_raw, act_raw, ends)
        print(f"  {len(S)} success transitions")

        # ── 3. Failure demos ───────────────────────────────────────────────────
        print(f"\n[3] Generating {args.n_failure} failure demonstrations...")
        SF, HF, GF, SF_S, SF_A, SF_SN = add_failure_tensors(args.n_failure)

        # ── 4. EFP: entropy estimators ─────────────────────────────────────────
        print(f"\n[4] Training entropy estimator (with failure, {args.efp_ent_ep} epochs)...")
        enet_full = EntropyNet()
        train_entropy(enet_full, S, H, G, SF, HF, GF,
                      use_failure=True, epochs=args.efp_ent_ep)

        print(f"\n[5] Training entropy estimator (no failure, {args.efp_ent_ep} epochs)...")
        enet_nofail = EntropyNet()
        train_entropy(enet_nofail, S, H, G, SF, HF, GF,
                      use_failure=False, epochs=args.efp_ent_ep)

        # ── 5. Dynamics ────────────────────────────────────────────────────────
        print("\n[6] Training dynamics model...")
        dyn_net = DynNet()
        train_dynamics(dyn_net,
                       torch.cat([S, SF_S]),
                       torch.cat([A, SF_A]),
                       torch.cat([SN, SF_SN]),
                       epochs=40)

        # ── 6. Flow matching (UNet VelocityNet) ───────────────────────────────
        print(f"\n[7] Training EFP-UNet velocity field ({args.efp_flow_ep} epochs)...")
        efp_vel_unet = ConditionalUnet1D(
            input_dim=ACT_DIM,
            global_cond_dim=OBS_HORIZON * STATE_DIM,
        ).to(DEVICE)
        efp_seq_dataset = EFPSequenceDataset(obs_raw, act_raw, ends)
        print(f"  EFP dataset: {len(efp_seq_dataset)} windows")
        train_efp_unet_flow(efp_vel_unet, efp_seq_dataset, epochs=args.efp_flow_ep)

        # ── 7. BC / MDN-BC ─────────────────────────────────────────────────────
        print("\n[8] Training BC...")
        bc_net = BCNet()
        train_bc(bc_net, S, G, A, epochs=80)

        print("\n[9] Training MDN-BC...")
        mdn_net = MDNNet()
        train_mdn(mdn_net, S, G, A, epochs=80)

        # ── 8. Diffusion Policy (UNet1D + DDPM + EMA) ─────────────────────────
        print(f"\n[10] Training Diffusion Policy ({args.dp_epochs} epochs on GPU)...")
        dp_model = ConditionalUnet1D(
            input_dim=ACT_DIM,
            global_cond_dim=OBS_HORIZON * obs_raw.shape[-1],
        ).to(DEVICE)
        dp_dataset = DPDataset(obs_raw, act_raw, ends, stats)
        print(f"  DP dataset: {len(dp_dataset)} windows")
        train_dp(dp_model, dp_dataset, num_epochs=args.dp_epochs)

        # ── 9. Save ────────────────────────────────────────────────────────────
        save_all(enet_full, enet_nofail, dyn_net, efp_vel_unet, bc_net, mdn_net, dp_model)

    # ── 10. Build policy functions ─────────────────────────────────────────────
    dp_policy = DPPolicy(dp_model, stats)

    policies = [
        ("EFP (w/ failure)",  EFPUNetPolicy(efp_vel_unet, enet_full,   dyn_net)),
        ("EFP (no failure)",  EFPUNetPolicy(efp_vel_unet, enet_nofail, dyn_net)),
        ("EFP (no guidance)", EFPUNetPolicy(efp_vel_unet, None,        dyn_net, guidance=0.0)),
        ("Diffusion Policy",  dp_policy),
        ("BC (Mean)",         lambda o, _b=bc_net:  _infer_bc(raw_to_state(o), _b)),
        ("MDN-BC",            lambda o, _m=mdn_net: _infer_mdn(raw_to_state(o), _m)),
    ]
    labels = [l for l, _ in policies]

    # ── 11. Evaluate ───────────────────────────────────────────────────────────
    print("\n[11] Evaluation...")

    results_std     = {}
    results_perturb = {}

    print("\n  === Standard (agent starts top-centre) ===")
    for label, fn in policies:
        mc, succ = evaluate_policy(fn, label, EVAL_STD)
        results_std[label] = (mc, succ)

    print("\n  === Perturb (agent starts above block) ===")
    for label, fn in policies:
        mc, succ = evaluate_policy(fn, label, EVAL_PERTURB)
        results_perturb[label] = (mc, succ)

    # ── 12. Plot ───────────────────────────────────────────────────────────────
    print("\n[12] Plotting...")
    plot_results(results_std, results_perturb, labels,
                 os.path.join(BASE, 'efp_pusht_real_results.png'))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Policy':<26} {'STD cov':>8} {'STD%':>6} {'PERTURB cov':>12} {'PERTURB%':>9}")
    print("-" * 65)
    for l in labels:
        sc, ss = results_std[l]
        pc, ps = results_perturb[l]
        print(f"{l:<26} {sc:>8.3f} {int(ss*100):>5}%  {pc:>12.3f} {int(ps*100):>8}%")


if __name__ == '__main__':
    main()
