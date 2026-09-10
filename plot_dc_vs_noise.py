"""
plot_dc_vs_noise.py
===================
Fix one environment configuration (seed) and vary only the stochastic
input to each policy across N trials:
  - Heat: evenly-spaced DC levels from 0.05 to 0.95
  - DP  : different random noise seeds (same env state)

Two side-by-side panels, trajectories coloured by DC level / trial index.

Usage
-----
  python plot_dc_vs_noise.py
  python plot_dc_vs_noise.py --seed 985772 --n_trials 18
"""

import argparse, collections, copy, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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


# ── single-trial eval with fixed stochastic input ────────────────────────────

def run_dp(net, stats, env_seed, noise_seed, max_steps=EVAL_MAX_STEPS):
    """Run DP with a fixed torch random seed for the initial noise."""
    scheduler = DDPMScheduler(num_train_timesteps=NUM_DDPM_STEPS,
                              beta_schedule='squaredcos_cap_v2',
                              clip_sample=True, prediction_type='epsilon')
    env = PushTEnv()
    env.seed(env_seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    traj, rewards = [obs[:2].copy()], []
    done, step_idx, action_buf, buf_idx = False, 0, None, 0
    rng = torch.Generator(device=DEVICE).manual_seed(noise_seed)

    with torch.no_grad():
        while not done:
            if action_buf is None or buf_idx >= ACTION_HORIZON:
                nobs     = _normalize(np.stack(obs_deque), stats['obs'])
                obs_cond = torch.from_numpy(nobs).unsqueeze(0).flatten(start_dim=1).to(DEVICE)
                nact     = torch.randn(1, PRED_HORIZON, ACTION_DIM,
                                       device=DEVICE, generator=rng)
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


def run_heat(net, heat, stats, env_seed, dc_val, max_steps=EVAL_MAX_STEPS,
             prediction_type='x0'):
    """Run Heat with a fixed scalar DC value for both action dimensions."""
    env = PushTEnv()
    env.seed(env_seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    traj, rewards = [obs[:2].copy()], []
    done, step_idx, action_buf, buf_idx = False, 0, None, 0
    episode_dc = torch.full((1, 1, ACTION_DIM), dc_val,
                            dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        while not done:
            if action_buf is None or buf_idx >= ACTION_HORIZON:
                nobs     = _normalize(np.stack(obs_deque), stats['obs'])
                obs_cond = torch.from_numpy(nobs).unsqueeze(0).flatten(start_dim=1).to(DEVICE)
                nact = episode_dc.expand(1, PRED_HORIZON, ACTION_DIM).clone()
                for k in reversed(range(heat.T)):
                    t_b  = torch.full((1,), k, device=DEVICE, dtype=torch.long)
                    pred = net(nact, t_b, global_cond=obs_cond)
                    if prediction_type == 'residual':
                        nact = nact + pred   # accumulate incremental delta, no re-blur
                    else:
                        pred_a0 = pred.clamp(0, 1)
                        nact = heat.blur(pred_a0, k-1) if k > 0 else pred_a0
                nact = nact.clamp(0, 1)
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
    verts_bar  = [(-L*scale/2, 0), (L*scale/2, 0),
                  (L*scale/2, scale), (-L*scale/2, scale)]
    verts_stem = [(-scale/2, scale), (-scale/2, L*scale),
                  (scale/2, L*scale), (scale/2, scale)]
    pts = np.array(verts_bar + verts_stem[::-1], dtype=float)
    pts[:, 1] -= (scale + L*scale) / 2
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T + np.array([cx, cy])
    ax.fill(pts[:, 0], pts[:, 1], color='#90EE90', alpha=0.5, zorder=1)
    ax.plot(np.append(pts[:, 0], pts[0, 0]),
            np.append(pts[:, 1], pts[0, 1]),
            color='#228B22', lw=1.5, zorder=2)


def setup_ax(ax, title):
    ax.set_xlim(0, ENV_SIZE); ax.set_ylim(ENV_SIZE, 0)
    ax.set_aspect('equal')
    ax.set_facecolor('#111111')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8, color='white')
    ax.tick_params(colors='#aaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#555'); spine.set_linewidth(1.2)
    ax.set_xlabel('x (px)', fontsize=9, color='#ccc')
    ax.set_ylabel('y (px)', fontsize=9, color='#ccc')
    draw_goal(ax)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed',       type=int,   default=985772,
                    help='env seed (fixed T configuration)')
    ap.add_argument('--n_trials',   type=int,   default=18,
                    help='number of DC levels / noise seeds')
    ap.add_argument('--heat_steps', type=int,   default=100)
    ap.add_argument('--alpha_max',  type=float, default=10.0)
    ap.add_argument('--ckpt',       default=CKPT,
                    help='checkpoint .pt to load')
    ap.add_argument('--tag',             default='',
                    help='suffix for output filename, e.g. "no_dc_aug"')
    ap.add_argument('--prediction_type', default='x0', choices=['x0', 'residual'],
                    help='heat prediction type used during training')
    args = ap.parse_args()

    tag = f'_{args.tag}' if args.tag else ''
    out = os.path.join(BASE, f'dc_vs_noise_seed{args.seed}{tag}.png')

    # ── load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint …")
    ckpt  = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    stats = ckpt['stats']

    global_cond_dim = OBS_HORIZON * OBS_DIM
    dp_net   = ConditionalUnet1D(ACTION_DIM, global_cond_dim).to(DEVICE)
    heat_net = ConditionalUnet1D(ACTION_DIM, global_cond_dim).to(DEVICE)
    dp_ema   = EMAModel(parameters=dp_net.parameters(),   power=0.75)
    heat_ema = EMAModel(parameters=heat_net.parameters(), power=0.75)
    dp_net.load_state_dict(ckpt['dp_net']); heat_net.load_state_dict(ckpt['heat_net'])
    dp_ema.load_state_dict(ckpt['dp_ema']); heat_ema.load_state_dict(ckpt['heat_ema'])
    dp_eval   = _ema_copy(dp_net,   dp_ema)
    heat_eval = _ema_copy(heat_net, heat_ema)
    heat      = HeatDissipation1D(args.heat_steps, args.alpha_max)

    # ── DC levels and noise seeds ─────────────────────────────────────────
    N       = args.n_trials
    dc_vals = np.linspace(0.05, 0.95, N)   # evenly cover [0,1]
    noise_seeds = list(range(N))

    # ── run trials ────────────────────────────────────────────────────────
    dp_results   = []   # (traj, score, noise_seed)
    heat_results = []   # (traj, score, dc_val)

    print(f"\nEnv seed: {args.seed}  |  {N} trials each\n")
    print(f"{'Trial':>5}  {'DC':>6}  {'Heat score':>10}  {'Noise seed':>10}  {'DP score':>8}")
    print('─' * 50)
    for i in range(N):
        ht, hs = run_heat(heat_eval, heat, stats, args.seed, float(dc_vals[i]),
                          prediction_type=args.prediction_type)
        dt, ds = run_dp(dp_eval, stats, args.seed, noise_seed=noise_seeds[i])
        heat_results.append((ht, hs, dc_vals[i]))
        dp_results.append((dt, ds, noise_seeds[i]))
        print(f"{i+1:>5}  {dc_vals[i]:>6.3f}  {hs:>10.3f}  {noise_seeds[i]:>10d}  {ds:>8.3f}")

    # ── plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             facecolor='#1a1a1a')
    fig.subplots_adjust(wspace=0.12, left=0.06, right=0.94,
                        top=0.88, bottom=0.08)

    # ── left: Diffusion Policy ────────────────────────────────────────────
    ax = axes[0]
    setup_ax(ax, f'Diffusion Policy\n{N} runs, same env (seed={args.seed}), different noise')
    cmap_dp = cm.cool
    for traj, score, ns in dp_results:
        color = cmap_dp(ns / max(N - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=color,
                alpha=0.75, lw=1.2, zorder=3)
        ax.scatter(traj[0, 0],  traj[0, 1],  color=color,
                   s=30, zorder=5, marker='o', edgecolors='white', linewidths=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color,
                   s=50, zorder=5, marker='*', edgecolors='white', linewidths=0.4)
    sm_dp = plt.cm.ScalarMappable(cmap=cmap_dp,
                                  norm=mcolors.Normalize(0, N-1))
    sm_dp.set_array([])
    cb = fig.colorbar(sm_dp, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Trial index', color='#ccc', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='#ccc', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#ccc')

    # ── right: Heat Dissipation ───────────────────────────────────────────
    ax = axes[1]
    setup_ax(ax, f'Heat Dissipation\n{N} runs, same env (seed={args.seed}), DC = 0.05 → 0.95')
    cmap_heat = cm.plasma
    for traj, score, dc in heat_results:
        color = cmap_heat((dc - 0.05) / 0.90)
        ax.plot(traj[:, 0], traj[:, 1], color=color,
                alpha=0.75, lw=1.2, zorder=3)
        ax.scatter(traj[0, 0],  traj[0, 1],  color=color,
                   s=30, zorder=5, marker='o', edgecolors='white', linewidths=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color,
                   s=50, zorder=5, marker='*', edgecolors='white', linewidths=0.4)
        # annotate DC value near start point
        ax.text(traj[0, 0] + 6, traj[0, 1] - 6, f'{dc:.2f}',
                color=color, fontsize=5.5, zorder=6, alpha=0.9)

    sm_heat = plt.cm.ScalarMappable(cmap=cmap_heat,
                                    norm=mcolors.Normalize(0.05, 0.95))
    sm_heat.set_array([])
    cb2 = fig.colorbar(sm_heat, ax=ax, fraction=0.04, pad=0.02)
    cb2.set_label('DC level (starting flat value)', color='#ccc', fontsize=8)
    cb2.ax.yaxis.set_tick_params(color='#ccc', labelsize=7)
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color='#ccc')

    # scores text
    dp_scores   = [r[1] for r in dp_results]
    heat_scores = [r[1] for r in heat_results]
    for ax_i, scores, label in [(axes[0], dp_scores, 'DP'),
                                 (axes[1], heat_scores, 'Heat')]:
        ax_i.text(0.02, 0.97,
                  f'mean score: {np.mean(scores):.3f}\n'
                  f'min: {np.min(scores):.3f}   max: {np.max(scores):.3f}',
                  transform=ax_i.transAxes, fontsize=8, color='white',
                  va='top', ha='left',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#333',
                            alpha=0.8, edgecolor='#555'))

    fig.suptitle(
        f'Same T-block configuration — different stochastic inputs\n'
        f'● = start   ★ = end   (green = goal region)',
        fontsize=11, color='white', y=0.97)

    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
