"""
eval_novel.py
=============
Evaluate DC-augmented Heat Dissipation vs Diffusion Policy on novel
configurations not seen during training.

Two test groups:
  1. Random large batch  — 50 random seeds, statistical comparison
  2. Hard hand-crafted   — T-block in corners / extremes / unusual angles

Produces:
  novel_eval_summary.png   — score distributions + trajectory grid
  novel_eval_hard.png      — side-by-side trajectories on hard cases

Usage
-----
  python eval_novel.py
"""

import collections, copy, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_dp_heat_pusht import (
    PushTEnv, ConditionalUnet1D, HeatDissipation1D,
    _normalize, _unnormalize, _ema_copy,
    OBS_DIM, ACTION_DIM, OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON,
    NUM_DDPM_STEPS, EVAL_MAX_STEPS, DEVICE,
)

BASE  = os.path.dirname(os.path.abspath(__file__))
CKPT  = os.path.join(BASE, 'checkpoints', 'epoch_100.pt')
ENV_SIZE = 512


# ── eval with trajectory capture ─────────────────────────────────────────────

def run_dp(net, stats, seed=None, init_state=None, max_steps=EVAL_MAX_STEPS):
    scheduler = DDPMScheduler(num_train_timesteps=NUM_DDPM_STEPS,
                              beta_schedule='squaredcos_cap_v2',
                              clip_sample=True, prediction_type='epsilon')
    env = PushTEnv()
    if init_state is not None:
        env.reset_to_state = init_state
    if seed is not None:
        env.seed(seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    traj, rewards = [obs[:2].copy()], []
    done, step_idx, action_buf, buf_idx = False, 0, None, 0

    with torch.no_grad():
        while not done:
            if action_buf is None or buf_idx >= ACTION_HORIZON:
                nobs     = _normalize(np.stack(obs_deque), stats['obs'])
                obs_cond = torch.from_numpy(nobs).unsqueeze(0).flatten(start_dim=1).to(DEVICE)
                nact     = torch.randn(1, PRED_HORIZON, ACTION_DIM, device=DEVICE)
                scheduler.set_timesteps(NUM_DDPM_STEPS)
                for k in scheduler.timesteps:
                    pred = net(nact, k, global_cond=obs_cond)
                    nact = scheduler.step(pred, k, nact).prev_sample
                action_buf = _unnormalize(nact[0].cpu().numpy(), stats['action'])
                buf_idx = 0
            obs, reward, done, _, _ = env.step(action_buf[buf_idx])
            obs_deque.append(obs); rewards.append(reward)
            traj.append(obs[:2].copy())
            buf_idx += 1; step_idx += 1
            if step_idx >= max_steps: done = True

    return np.array(traj), float(max(rewards)) if rewards else 0.0


def run_heat(net, heat, stats, seed=None, init_state=None, max_steps=EVAL_MAX_STEPS):
    env = PushTEnv()
    if init_state is not None:
        env.reset_to_state = init_state
    if seed is not None:
        env.seed(seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    traj, rewards = [obs[:2].copy()], []
    done, step_idx, action_buf, buf_idx = False, 0, None, 0
    episode_dc = torch.rand(1, 1, ACTION_DIM, device=DEVICE)

    with torch.no_grad():
        while not done:
            if action_buf is None or buf_idx >= ACTION_HORIZON:
                nobs     = _normalize(np.stack(obs_deque), stats['obs'])
                obs_cond = torch.from_numpy(nobs).unsqueeze(0).flatten(start_dim=1).to(DEVICE)
                nact = episode_dc.expand(1, PRED_HORIZON, ACTION_DIM).clone()
                for k in reversed(range(heat.T)):
                    t_b  = torch.full((1,), k, device=DEVICE, dtype=torch.long)
                    pred = net(nact, t_b, global_cond=obs_cond)
                    nact = heat.blur(pred.clamp(0, 1), k-1) if k > 0 else pred
                action_buf = _unnormalize(nact[0].cpu().numpy(), stats['action'])
                buf_idx = 0
            obs, reward, done, _, _ = env.step(action_buf[buf_idx])
            obs_deque.append(obs); rewards.append(reward)
            traj.append(obs[:2].copy())
            buf_idx += 1; step_idx += 1
            if step_idx >= max_steps: done = True

    return np.array(traj), float(max(rewards)) if rewards else 0.0


# ── drawing helpers ───────────────────────────────────────────────────────────

def draw_goal(ax):
    cx, cy, angle = 256, 256, np.pi / 4
    scale = 30; L = 4
    pts = np.array([(-L*scale/2,0),(L*scale/2,0),(L*scale/2,scale),
                    (-scale/2,scale),(-scale/2,L*scale),(scale/2,L*scale),
                    (scale/2,scale),(-L*scale/2,scale)], dtype=float)
    pts[:,1] -= (scale + L*scale) / 2
    c,s = np.cos(angle), np.sin(angle)
    pts = pts @ np.array([[c,-s],[s,c]]).T + [cx,cy]
    ax.fill(pts[:,0], pts[:,1], color='#90EE90', alpha=0.45, zorder=1)
    ax.plot(np.append(pts[:,0],pts[0,0]),
            np.append(pts[:,1],pts[0,1]), color='#228B22', lw=1.2, zorder=2)

def draw_block_start(ax, state, color, alpha=0.6):
    """Draw a small square marker showing where the T-block starts."""
    bx, by = state[2], state[3]
    ax.scatter(bx, by, s=80, marker='s', color=color,
               edgecolors='white', linewidths=0.6, alpha=alpha, zorder=6)

def setup_ax(ax, title='', dark=True):
    bg = '#111111' if dark else '#f5f5f5'
    ax.set_xlim(0, ENV_SIZE); ax.set_ylim(ENV_SIZE, 0)
    ax.set_aspect('equal'); ax.set_facecolor(bg)
    if title: ax.set_title(title, fontsize=9, fontweight='bold',
                            color='white' if dark else 'black', pad=4)
    ax.tick_params(colors='#888', labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#555')
    draw_goal(ax)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # ── load models ───────────────────────────────────────────────────────
    print("Loading checkpoint …")
    ckpt  = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    stats = ckpt['stats']

    gcond = OBS_HORIZON * OBS_DIM
    dp_net   = ConditionalUnet1D(ACTION_DIM, gcond).to(DEVICE)
    heat_net = ConditionalUnet1D(ACTION_DIM, gcond).to(DEVICE)
    dp_ema   = EMAModel(parameters=dp_net.parameters(),   power=0.75)
    heat_ema = EMAModel(parameters=heat_net.parameters(), power=0.75)
    dp_net.load_state_dict(ckpt['dp_net']); heat_net.load_state_dict(ckpt['heat_net'])
    dp_ema.load_state_dict(ckpt['dp_ema']); heat_ema.load_state_dict(ckpt['heat_ema'])
    dp_eval   = _ema_copy(dp_net,   dp_ema)
    heat_eval = _ema_copy(heat_net, heat_ema)
    heat      = HeatDissipation1D(100, 10.0)

    # ── Group 1: 50 random seeds ──────────────────────────────────────────
    rng   = np.random.RandomState(42)
    seeds = rng.randint(1_000_000, 9_999_999, size=50).tolist()  # high range, far from training seeds

    print(f"\n── Group 1: 50 random seeds ──")
    dp_rand, heat_rand = [], []
    for i, s in enumerate(seeds):
        _, ds = run_dp(dp_eval,   stats, seed=s)
        _, hs = run_heat(heat_eval, heat, stats, seed=s)
        dp_rand.append(ds); heat_rand.append(hs)
        print(f'  [{i+1:2d}/50]  seed={s}  dp={ds:.3f}  heat={hs:.3f}')

    dp_rand   = np.array(dp_rand)
    heat_rand = np.array(heat_rand)

    # ── Group 2: hand-crafted hard configs ────────────────────────────────
    # state = [agent_x, agent_y, block_x, block_y, block_angle]
    # goal  = T at (256, 256, π/4)
    hard_configs = [
        ('Top-left corner',    [256, 256,  80,  80, 0.0]),
        ('Top-right corner',   [256, 256, 430,  80, 0.0]),
        ('Bottom-left corner', [256, 256,  80, 430, np.pi]),
        ('Bottom-right corner',[256, 256, 430, 430, np.pi]),
        ('Max dist (TL→BR)',   [430, 430,  80,  80, np.pi/2]),
        ('Upright angle',      [256, 400, 256, 100, 0.0]),
        ('Upside-down',        [256, 400, 256, 400, np.pi]),
        ('Agent far left',     [ 60, 256, 400, 256, np.pi/4]),
        ('Agent far right',    [450, 256, 100, 256, np.pi/4]),
        ('Block at goal angle',[100, 400, 256, 256, 0.0]),   # in position, wrong angle
        ('Block 90° off',      [100, 300, 300, 200, np.pi/4 + np.pi/2]),
        ('Block 180° off',     [400, 400, 200, 300, np.pi/4 + np.pi]),
    ]

    print(f"\n── Group 2: {len(hard_configs)} hand-crafted hard configs ──")
    hard_results = []
    for name, state in hard_configs:
        state_arr = np.array(state, dtype=float)
        dt, ds = run_dp(dp_eval,   stats, init_state=state_arr)
        ht, hs = run_heat(heat_eval, heat, stats, init_state=state_arr)
        hard_results.append((name, state_arr, dt, ds, ht, hs))
        print(f'  {name:<25s}  dp={ds:.3f}  heat={hs:.3f}')

    # ── Plot 1: score distributions (random batch) ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')

    # histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(dp_rand,   bins=bins, alpha=0.7, color='steelblue',  label='Diffusion Policy', density=True)
    ax.hist(heat_rand, bins=bins, alpha=0.7, color='darkorange', label='Heat Dissipation', density=True)
    ax.set_xlabel('Score', fontsize=10); ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Score distribution — 50 novel random seeds', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.97,
            f'DP:   mean={dp_rand.mean():.3f}  success={( dp_rand>0.9).mean()*100:.0f}%\n'
            f'Heat: mean={heat_rand.mean():.3f}  success={(heat_rand>0.9).mean()*100:.0f}%',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='#eee', alpha=0.8))

    # scatter: DP vs Heat per config
    ax = axes[1]
    ax.scatter(dp_rand, heat_rand, alpha=0.6, s=40, color='purple')
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='equal')
    ax.set_xlabel('DP score', fontsize=10); ax.set_ylabel('Heat score', fontsize=10)
    ax.set_title('Head-to-head per config', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    n_dp_wins = (dp_rand > heat_rand).sum()
    n_heat_wins = (heat_rand > dp_rand).sum()
    ax.text(0.03, 0.97,
            f'DP wins:   {n_dp_wins}/50\nHeat wins: {n_heat_wins}/50\nTie:       {50-n_dp_wins-n_heat_wins}/50',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='#eee', alpha=0.8))

    # hard configs bar chart
    ax = axes[2]
    names = [r[0] for r in hard_results]
    dp_h  = [r[3] for r in hard_results]
    heat_h= [r[5] for r in hard_results]
    x = np.arange(len(names))
    w = 0.38
    ax.barh(x - w/2, dp_h,   w, color='steelblue',  alpha=0.8, label='DP')
    ax.barh(x + w/2, heat_h, w, color='darkorange', alpha=0.8, label='Heat')
    ax.set_yticks(x); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Score', fontsize=9)
    ax.set_title('Hard hand-crafted configs', fontsize=10, fontweight='bold')
    ax.axvline(0.95, color='green', lw=1, ls='--', alpha=0.5, label='success threshold')
    ax.legend(fontsize=8); ax.set_xlim(0, 1.05); ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'novel_eval_summary.png'), dpi=160, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → novel_eval_summary.png")

    # ── Plot 2: trajectories on hard configs (dark theme) ─────────────────
    n  = len(hard_configs)
    nc = 4; nr = int(np.ceil(n / nc))
    fig = plt.figure(figsize=(nc * 4, nr * 4), facecolor='#1a1a1a')
    fig.suptitle('Hard configs — DP (blue) vs Heat (orange) trajectories\n'
                 '■ = T-block start   ● = agent start   ★ = agent end',
                 color='white', fontsize=11, y=1.01)

    for idx, (name, state, dt, ds, ht, hs) in enumerate(hard_results):
        ax = fig.add_subplot(nr, nc, idx + 1)
        setup_ax(ax, f'{name}\nDP={ds:.3f}  Heat={hs:.3f}', dark=True)

        # T-block start marker
        draw_block_start(ax, state, 'white')

        # DP trajectory
        ax.plot(dt[:,0], dt[:,1], color='steelblue', alpha=0.8, lw=1.3, zorder=3)
        ax.scatter(dt[0,0],  dt[0,1],  color='steelblue', s=25, zorder=5,
                   marker='o', edgecolors='white', linewidths=0.5)
        ax.scatter(dt[-1,0], dt[-1,1], color='steelblue', s=40, zorder=5,
                   marker='*', edgecolors='white', linewidths=0.4)

        # Heat trajectory
        ax.plot(ht[:,0], ht[:,1], color='darkorange', alpha=0.8, lw=1.3, zorder=3)
        ax.scatter(ht[0,0],  ht[0,1],  color='darkorange', s=25, zorder=5,
                   marker='o', edgecolors='white', linewidths=0.5)
        ax.scatter(ht[-1,0], ht[-1,1], color='darkorange', s=40, zorder=5,
                   marker='*', edgecolors='white', linewidths=0.4)

        ax.tick_params(colors='#666', labelsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'novel_eval_hard.png'),
                dpi=160, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"Saved → novel_eval_hard.png")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  RANDOM BATCH (50 seeds)")
    print(f"    DP  : mean={dp_rand.mean():.3f}  median={np.median(dp_rand):.3f}  "
          f"success(>0.9)={( dp_rand>0.9).mean()*100:.0f}%")
    print(f"    Heat: mean={heat_rand.mean():.3f}  median={np.median(heat_rand):.3f}  "
          f"success(>0.9)={(heat_rand>0.9).mean()*100:.0f}%")
    print(f"  HARD CONFIGS ({len(hard_configs)} cases)")
    print(f"    DP  : mean={np.mean(dp_h):.3f}  "
          f"success={(np.array(dp_h)>0.9).mean()*100:.0f}%")
    print(f"    Heat: mean={np.mean(heat_h):.3f}  "
          f"success={(np.array(heat_h)>0.9).mean()*100:.0f}%")


if __name__ == '__main__':
    main()
