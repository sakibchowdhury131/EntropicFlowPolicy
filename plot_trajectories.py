"""
plot_trajectories.py
====================
Three-panel figure:
  Left  : all agent trajectories from the training dataset
  Centre: trajectories taken by Diffusion Policy across eval configs
  Right : trajectories taken by Heat Dissipation Policy across eval configs

Only successful eval runs (score >= 0.5) are shown in the eval panels.

Usage
-----
  python plot_trajectories.py
  python plot_trajectories.py --n_configs 16 --success_thresh 0.5
"""

import argparse, collections, copy, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import torch
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_dp_heat_pusht import (
    PushTEnv, ConditionalUnet1D, HeatDissipation1D,
    _normalize, _unnormalize, _ema_copy,
    OBS_DIM, ACTION_DIM, OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON,
    NUM_DDPM_STEPS, EVAL_MAX_STEPS, DEVICE,
)

BASE     = os.path.dirname(os.path.abspath(__file__))
ZARR_PATH = os.path.join(BASE, 'pusht_cchi_v7_replay.zarr.zip')
CKPT     = os.path.join(BASE, 'checkpoints', 'epoch_100.pt')
OUT      = os.path.join(BASE, 'trajectories.png')

ENV_SIZE = 512   # PushT workspace is 512x512


# ── eval with trajectory capture ─────────────────────────────────────────────

def eval_dp_traj(net, stats, seed, max_steps=EVAL_MAX_STEPS):
    scheduler = DDPMScheduler(num_train_timesteps=NUM_DDPM_STEPS,
                              beta_schedule='squaredcos_cap_v2',
                              clip_sample=True, prediction_type='epsilon')
    env = PushTEnv()
    env.seed(seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    rewards, traj = [], [obs[:2].copy()]   # agent xy positions
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
            obs_deque.append(obs)
            rewards.append(reward)
            traj.append(obs[:2].copy())   # agent position from obs
            buf_idx += 1; step_idx += 1
            if step_idx >= max_steps: done = True

    return np.array(traj), float(max(rewards)) if rewards else 0.0


def eval_heat_traj(net, heat, stats, seed, max_steps=EVAL_MAX_STEPS):
    env = PushTEnv()
    env.seed(seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    rewards, traj = [], [obs[:2].copy()]
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
            obs_deque.append(obs)
            rewards.append(reward)
            traj.append(obs[:2].copy())
            buf_idx += 1; step_idx += 1
            if step_idx >= max_steps: done = True

    return np.array(traj), float(max(rewards)) if rewards else 0.0


# ── drawing helpers ───────────────────────────────────────────────────────────

def draw_goal(ax):
    """Draw the goal T-shape outline at (256,256) angle=π/4."""
    import matplotlib.patches as patches
    from matplotlib.transforms import Affine2D
    cx, cy, angle = 256, 256, np.pi / 4
    scale = 30
    L = 4
    # horizontal bar
    w1, h1 = L * scale, scale
    # vertical stem
    w2, h2 = scale, L * scale

    def rotated_rect(ax, x, y, w, h, angle, cx, cy, **kw):
        corners = np.array([[-w/2, -h/2], [w/2, -h/2],
                             [w/2,  h/2], [-w/2,  h/2]])
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        corners = corners @ R.T + np.array([cx, cy])
        ax.fill(corners[:, 0], corners[:, 1], **kw)

    # horizontal bar centred at (cx, cy + scale/2)
    bx = cx + (scale/2) * np.sin(angle)
    by = cy - (scale/2) * np.cos(angle)
    # just draw the full T outline as a polygon
    verts_bar  = [(-L*scale/2, 0), (L*scale/2, 0),
                  (L*scale/2, scale), (-L*scale/2, scale)]
    verts_stem = [(-scale/2, scale), (-scale/2, L*scale),
                  (scale/2, L*scale), (scale/2, scale)]
    all_verts  = verts_bar + verts_stem[::-1]
    pts = np.array(all_verts, dtype=float)
    # centre of mass offset
    pts[:, 0] -= 0
    pts[:, 1] -= (scale + L*scale) / 2
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T + np.array([cx, cy])
    ax.fill(pts[:, 0], pts[:, 1], color='#90EE90', alpha=0.5, zorder=1)
    ax.plot(np.append(pts[:, 0], pts[0, 0]),
            np.append(pts[:, 1], pts[0, 1]),
            color='#228B22', lw=1.2, zorder=2)


def setup_ax(ax, title):
    ax.set_xlim(0, ENV_SIZE)
    ax.set_ylim(ENV_SIZE, 0)   # y-axis flipped (pygame convention)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f8f8')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('x (px)', fontsize=8)
    ax.set_ylabel('y (px)', fontsize=8)
    ax.tick_params(labelsize=7)
    # walls
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#555')
    draw_goal(ax)


def colored_lines(ax, trajs, cmap_name, alpha=0.55, lw=0.9):
    """Draw trajectories with colour cycling and faint start/end markers."""
    cmap = plt.get_cmap(cmap_name)
    n = len(trajs)
    for i, traj in enumerate(trajs):
        color = cmap(i / max(n - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=color,
                alpha=alpha, lw=lw, zorder=3)
        ax.scatter(traj[0, 0],  traj[0, 1],  color=color,
                   s=12, zorder=4, marker='o', edgecolors='none')
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color,
                   s=18, zorder=4, marker='*', edgecolors='none')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',            default=CKPT)
    ap.add_argument('--n_configs',       type=int,   default=16)
    ap.add_argument('--success_thresh',  type=float, default=0.5)
    ap.add_argument('--heat_steps',      type=int,   default=100)
    ap.add_argument('--alpha_max',       type=float, default=10.0)
    args = ap.parse_args()

    # ── 1. Training trajectories from zarr ───────────────────────────────
    print("Loading training trajectories …")
    store    = zarr.storage.ZipStore(ZARR_PATH, mode='r')
    root     = zarr.open(store=store, mode='r')
    actions  = root['data']['action'][:]        # (N, 2)  raw pixel coords
    ep_ends  = root['meta']['episode_ends'][:]  # episode boundaries

    train_trajs = []
    prev = 0
    for end in ep_ends:
        train_trajs.append(actions[prev:end])
        prev = end
    print(f"  {len(train_trajs)} episodes, {len(actions)} total steps")

    # ── 2. Load models ───────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    stats = ckpt['stats']

    global_cond_dim = OBS_HORIZON * OBS_DIM
    dp_net   = ConditionalUnet1D(ACTION_DIM, global_cond_dim).to(DEVICE)
    heat_net = ConditionalUnet1D(ACTION_DIM, global_cond_dim).to(DEVICE)
    dp_ema   = EMAModel(parameters=dp_net.parameters(),   power=0.75)
    heat_ema = EMAModel(parameters=heat_net.parameters(), power=0.75)

    dp_net.load_state_dict(ckpt['dp_net'])
    heat_net.load_state_dict(ckpt['heat_net'])
    dp_ema.load_state_dict(ckpt['dp_ema'])
    heat_ema.load_state_dict(ckpt['heat_ema'])

    dp_eval   = _ema_copy(dp_net,   dp_ema)
    heat_eval = _ema_copy(heat_net, heat_ema)
    heat      = HeatDissipation1D(num_timesteps=args.heat_steps,
                                  alpha_max=args.alpha_max)

    # ── 3. Run eval configs ───────────────────────────────────────────────
    rng   = np.random.RandomState(0)
    seeds = rng.randint(0, 1_000_000, size=args.n_configs).tolist()

    dp_trajs, heat_trajs = [], []
    print(f"\nRunning {args.n_configs} eval configs …")
    for i, seed in enumerate(seeds):
        dt, ds = eval_dp_traj(dp_eval,   stats, seed=seed)
        ht, hs = eval_heat_traj(heat_eval, heat, stats, seed=seed)
        if ds >= args.success_thresh:
            dp_trajs.append(dt)
        if hs >= args.success_thresh:
            heat_trajs.append(ht)
        print(f'  [{i+1:2d}/{args.n_configs}]  seed={seed:<8d}  '
              f'dp={ds:.3f} ({"kept" if ds>=args.success_thresh else "skip"})  '
              f'heat={hs:.3f} ({"kept" if hs>=args.success_thresh else "skip"})')

    print(f"\n  DP   kept: {len(dp_trajs)}/{args.n_configs}")
    print(f"  Heat kept: {len(heat_trajs)}/{args.n_configs}")

    # ── 4. Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.patch.set_facecolor('white')

    # ── panel 1: training data ────────────────────────────────────────────
    setup_ax(axes[0], f'Training Data\n({len(train_trajs)} episodes)')
    colored_lines(axes[0], train_trajs, 'viridis', alpha=0.25, lw=0.6)

    # ── panel 2: DP eval ─────────────────────────────────────────────────
    setup_ax(axes[1], f'Diffusion Policy — Eval\n'
                      f'({len(dp_trajs)}/{args.n_configs} success, '
                      f'thresh={args.success_thresh})')
    colored_lines(axes[1], dp_trajs, 'Blues_r', alpha=0.7, lw=1.2)
    axes[1].scatter([], [], color='steelblue', s=20, marker='o', label='start')
    axes[1].scatter([], [], color='steelblue', s=30, marker='*', label='end')
    axes[1].legend(fontsize=7, loc='lower right')

    # ── panel 3: Heat eval ────────────────────────────────────────────────
    setup_ax(axes[2], f'Heat Dissipation — Eval\n'
                      f'({len(heat_trajs)}/{args.n_configs} success, '
                      f'thresh={args.success_thresh})')
    colored_lines(axes[2], heat_trajs, 'Oranges_r', alpha=0.7, lw=1.2)
    axes[2].scatter([], [], color='darkorange', s=20, marker='o', label='start')
    axes[2].scatter([], [], color='darkorange', s=30, marker='*', label='end')
    axes[2].legend(fontsize=7, loc='lower right')

    # shared legend for goal marker
    goal_patch = mpatches.Patch(facecolor='#90EE90', edgecolor='#228B22',
                                 label='goal region')
    for ax in axes:
        ax.legend(handles=ax.get_legend().legend_handles + [goal_patch]
                  if ax.get_legend() else [goal_patch],
                  fontsize=7, loc='lower right')

    plt.suptitle('PushT Agent Trajectories — Training Data vs Learned Policies',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUT, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {OUT}")


if __name__ == '__main__':
    main()
