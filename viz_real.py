"""
viz_real.py — Rollout GIFs + entropy field visualization for EFP-UNet models.

Loads models from models_real/ (produced by efp_pusht_real.py).

Usage
-----
  python viz_real.py              # rollouts + entropy field
  python viz_real.py --entropy    # entropy field plots only
  python viz_real.py --rollout    # rollouts only
  python viz_real.py --tries N    # seeds per condition (default 8)

Output
------
  gifs_real/compare_standard.gif
  gifs_real/compare_perturb.gif
  gifs_real/<cond>_<policy>.gif
  gifs_real/entropy_field.png     2×3 heatmap grid
  gifs_real/entropy_trajectory.png  H(t) along one EFP episode
"""

import sys, os, math, argparse, collections
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as iio
import gymnasium as gym
import gym_pusht  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from efp_pusht import (
    raw_to_state, reset_env,
    GOAL_VEC, IMG, ACT_DIM, STATE_DIM, SUCCESS_THR,
    BLK_ORIGIN_X, BLK_ORIGIN_Y, BLK_INIT_ANG,
    GOAL_X, GOAL_Y, GOAL_ANGLE,
    EVAL_STD, EVAL_PERTURB,
)
from efp_pusht_real import (
    load_all, DPPolicy, EFPUNetPolicy,
    get_stats, load_zarr,
    OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON, ACT_DIM,
    _infer_bc, _infer_mdn,
)

BASE      = os.path.dirname(os.path.abspath(__file__))
GIFS_DIR  = os.path.join(BASE, 'gifs_real')
MAX_STEPS = 400


# ── Model loading ──────────────────────────────────────────────────────────────

def load_real_policies():
    print("Loading models from models_real/ ...")
    enet_full, enet_nofail, dyn_net, efp_vel_unet, bc_net, mdn_net, dp_model = load_all()
    obs_raw, act_raw, ends = load_zarr()
    stats = get_stats(obs_raw, act_raw)

    policies = [
        ("EFP_failure",   "EFP (w/ failure)",
         EFPUNetPolicy(efp_vel_unet, enet_full,   dyn_net, guidance=0.03)),
        ("EFP_nofailure", "EFP (no failure)",
         EFPUNetPolicy(efp_vel_unet, enet_nofail, dyn_net, guidance=0.03)),
        ("EFP_noguide",   "EFP (no guidance)",
         EFPUNetPolicy(efp_vel_unet, None,        dyn_net, guidance=0.0)),
        ("DP",            "Diffusion Policy",
         DPPolicy(dp_model, stats)),
        ("BC",            "BC (Mean)",
         lambda o, b=bc_net:  _infer_bc(raw_to_state(o), b)),
        ("MDN_BC",        "MDN-BC",
         lambda o, m=mdn_net: _infer_mdn(raw_to_state(o), m)),
    ]
    return policies, enet_full, enet_nofail, dyn_net


# ── Rendering helpers ──────────────────────────────────────────────────────────

def make_render_env():
    return gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")


def run_episode(policy_fn, ax, ay, bx_off=0.0, by_off=0.0, ba_off=0.0,
                enet=None, dyn=None):
    """Run one episode; returns (frames, max_cov, H_trace).
    H_trace only populated when enet+dyn supplied.
    """
    env = make_render_env()
    obs, _ = reset_env(env, agent_x=ax, agent_y=ay,
                       bx_off=bx_off, by_off=by_off, ba_off=ba_off)
    if hasattr(policy_fn, 'reset'):
        policy_fn.reset(obs)
    frames  = [env.render()]
    max_cov = 0.0
    H_trace = []
    g_t = torch.tensor(GOAL_VEC, dtype=torch.float32).unsqueeze(0)

    for _ in range(MAX_STEPS):
        action = policy_fn(obs)
        if enet is not None:
            with torch.no_grad():
                s_t = torch.tensor(raw_to_state(obs), dtype=torch.float32).unsqueeze(0)
                H_trace.append(float(enet(torch.cat([s_t, g_t], -1))))
        obs, reward, done, truncated, _ = env.step(action)
        max_cov = max(max_cov, float(reward))
        frames.append(env.render())
        if done or truncated:
            break

    env.close()
    return frames, max_cov, H_trace


def _annotate(frame_np, text, bg=(20, 20, 20), fg=(255, 255, 255)):
    img = Image.fromarray(frame_np)
    d   = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 28], fill=bg)
    d.text((8, 7), text, fill=fg)
    return np.array(img)


def save_gif(frames, path, fps=15, label=None, size=(340, 340)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = []
    for f in frames:
        if label:
            f = _annotate(f, label)
        if size:
            f = np.array(Image.fromarray(f).resize(size, Image.LANCZOS))
        out.append(f.astype(np.uint8))
    iio.mimwrite(path, out, fps=fps)
    print(f"    → {path}  ({len(out)} frames)")


def save_comparison_gif(all_frames, labels, path, fps=15, cell=200):
    n      = len(all_frames)
    max_t  = max(len(f) for f in all_frames)
    padded = [f + [f[-1]] * (max_t - len(f)) for f in all_frames]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = []
    for t in range(max_t):
        cells = []
        for i in range(n):
            img = Image.fromarray(padded[i][t]).resize((cell, cell), Image.LANCZOS)
            d   = ImageDraw.Draw(img)
            d.rectangle([0, 0, cell, 26], fill=(20, 20, 20))
            d.text((5, 6), labels[i][:32], fill=(255, 255, 255))
            cells.append(np.array(img))
        # arrange 3 per row for 6 policies
        rows = []
        for r in range(0, n, 3):
            row_cells = cells[r:r+3]
            if len(row_cells) < 3:
                blank = np.zeros_like(row_cells[0])
                row_cells += [blank] * (3 - len(row_cells))
            rows.append(np.concatenate(row_cells, axis=1))
        out.append(np.concatenate(rows, axis=0).astype(np.uint8))
    iio.mimwrite(path, out, fps=fps)
    print(f"    → {path}  ({max_t} frames, 3×2 grid {3*cell}×{2*cell}px)")


# ── Policy rollouts ────────────────────────────────────────────────────────────

def do_rollouts(policies, n_tries=8):
    os.makedirs(GIFS_DIR, exist_ok=True)

    # Use the first n_tries seeds from eval sets for consistency
    std_seeds    = EVAL_STD[:n_tries]
    perturb_seeds = EVAL_PERTURB[:n_tries]

    conditions = [
        ("standard", std_seeds),
        ("perturb",  perturb_seeds),
    ]

    efp_fn = policies[0][2]   # EFP (w/ failure) — use to pick best seed

    for cond_key, seeds in conditions:
        print(f"\n=== {cond_key.upper()} ===")

        # Pick seed where EFP (w/ failure) gets highest coverage
        print(f"  Searching {len(seeds)} seeds for best visual...", flush=True)
        best_cov, best_seed = -1.0, seeds[0]
        for seed in seeds:
            ax, ay, bx_off, by_off, ba_off = seed
            if hasattr(efp_fn, 'reset'):
                efp_fn.reset(np.array([ax, ay,
                                       BLK_ORIGIN_X + bx_off,
                                       BLK_ORIGIN_Y + by_off,
                                       BLK_INIT_ANG + ba_off]))
            _, cov, _ = run_episode(efp_fn, ax, ay, bx_off, by_off, ba_off)
            if cov > best_cov:
                best_cov, best_seed = cov, seed
        ax, ay, bx_off, by_off, ba_off = best_seed
        ok = "SUCCESS" if best_cov > SUCCESS_THR else f"cov={best_cov:.2f}"
        print(f"  Best: EFP(failure) {ok}  "
              f"agent=({ax:.0f},{ay:.0f})  block_off=({bx_off:.1f},{by_off:.1f})")

        all_frames, all_labels = [], []
        for file_key, display, fn in policies:
            print(f"  Rolling out {display}...", end=' ', flush=True)
            frames, cov, _ = run_episode(fn, ax, ay, bx_off, by_off, ba_off)
            ep_tag = "SUCCESS" if cov > SUCCESS_THR else f"fail  cov={cov:.2f}"
            print(f"[{ep_tag}]  {len(frames)} steps")

            save_gif(frames,
                     os.path.join(GIFS_DIR, f'{cond_key}_{file_key}.gif'),
                     fps=15, label=f"{display}  [{ep_tag}]")

            all_frames.append(frames)
            all_labels.append(f"{display}  [{ep_tag}]")

        print("  Building comparison GIF...")
        save_comparison_gif(all_frames, all_labels,
                            os.path.join(GIFS_DIR, f'compare_{cond_key}.gif'),
                            fps=15, cell=220)


# ── Entropy field visualization ────────────────────────────────────────────────

def _entropy_grid(enet, xs, ys, fixed_state_fn):
    """Evaluate enet on a 2D grid.  fixed_state_fn(x, y) → 6D state array."""
    G = torch.tensor(GOAL_VEC, dtype=torch.float32)
    H = np.zeros((len(ys), len(xs)), dtype=np.float32)
    enet.eval()
    with torch.no_grad():
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                s  = torch.tensor(fixed_state_fn(x, y), dtype=torch.float32)
                sg = torch.cat([s, G]).unsqueeze(0)
                H[i, j] = float(enet(sg))
    return H


def plot_entropy_field(enet_full, enet_nofail, out_path):
    """
    Generate a 2×3 heatmap figure:
      Row 1: H over (block_x, block_y)  — agent fixed at centre
      Row 2: H over (agent_x, agent_y)  — block fixed at origin
    Columns: enet_full | enet_nofail | difference (full − nofail)
    """
    N = 80   # grid resolution

    # ── Grid 1: block position, agent at image centre ──────────────────────────
    bx_range = np.linspace(40,  470, N)
    by_range = np.linspace(40,  470, N)
    # fix agent at (256,256), block angle at goal angle
    def state_blk(bx, by):
        return raw_to_state(np.array(
            [256.0, 256.0, bx, by, GOAL_ANGLE], dtype=np.float32))

    print("  Computing entropy field over block position (enet_full)...")
    H_blk_full   = _entropy_grid(enet_full,   bx_range, by_range, state_blk)
    print("  Computing entropy field over block position (enet_nofail)...")
    H_blk_nofail = _entropy_grid(enet_nofail, bx_range, by_range, state_blk)

    # ── Grid 2: agent position, block at canonical origin ─────────────────────
    ax_range = np.linspace(20,  490, N)
    ay_range = np.linspace(20,  490, N)
    def state_agent(ax, ay):
        return raw_to_state(np.array(
            [ax, ay, BLK_ORIGIN_X, BLK_ORIGIN_Y, GOAL_ANGLE], dtype=np.float32))

    print("  Computing entropy field over agent position (enet_full)...")
    H_agt_full   = _entropy_grid(enet_full,   ax_range, ay_range, state_agent)
    print("  Computing entropy field over agent position (enet_nofail)...")
    H_agt_nofail = _entropy_grid(enet_nofail, ax_range, ay_range, state_agent)

    # ── Plot ───────────────────────────────────────────────────────────────────
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    cmap = 'plasma_r'   # bright = low H = close to goal

    def _heatmap(ax, H, xs, ys, title, xlabel, ylabel,
                 goal_px=None, block_px=None, vmin=None, vmax=None):
        vm = (vmin, vmax) if vmin is not None else (H.min(), H.max())
        im = ax.imshow(H, origin='lower', aspect='auto',
                       extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       cmap=cmap, vmin=vm[0], vmax=vm[1])
        if goal_px is not None:
            ax.scatter(*goal_px, marker='*', s=220, c='cyan',
                       edgecolors='white', linewidths=0.8, zorder=5,
                       label='Goal position')
        if block_px is not None:
            ax.scatter(*block_px, marker='s', s=120, c='lime',
                       edgecolors='white', linewidths=0.8, zorder=5,
                       label='Block origin')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.04)
        plt.colorbar(im, cax=cax, label='H(s,g) = log(steps to goal)')
        if goal_px is not None:
            ax.legend(fontsize=7, loc='upper left')
        return im

    # Shared colour limits per row so the full/nofail columns are comparable
    vlim_blk = (min(H_blk_full.min(), H_blk_nofail.min()),
                 max(H_blk_full.max(), H_blk_nofail.max()))
    vlim_agt = (min(H_agt_full.min(), H_agt_nofail.min()),
                 max(H_agt_full.max(), H_agt_nofail.max()))

    _heatmap(axes[0, 0], H_blk_full, bx_range, by_range,
             'Block position — Hθ (with failure)',
             'block x (px)', 'block y (px)',
             goal_px=(GOAL_X, GOAL_Y), vmin=vlim_blk[0], vmax=vlim_blk[1])
    _heatmap(axes[0, 1], H_blk_nofail, bx_range, by_range,
             'Block position — Hθ (no failure)',
             'block x (px)', 'block y (px)',
             goal_px=(GOAL_X, GOAL_Y), vmin=vlim_blk[0], vmax=vlim_blk[1])
    diff_blk = H_blk_full - H_blk_nofail
    im_d = axes[0, 2].imshow(diff_blk, origin='lower', aspect='auto',
                               extent=[bx_range[0], bx_range[-1],
                                       by_range[0], by_range[-1]],
                               cmap='RdBu_r',
                               vmin=-abs(diff_blk).max(), vmax=abs(diff_blk).max())
    axes[0, 2].scatter(GOAL_X, GOAL_Y, marker='*', s=200, c='black', zorder=5)
    axes[0, 2].set_title('Block position — Δ H (full − nofail)', fontsize=11,
                          fontweight='bold')
    axes[0, 2].set_xlabel('block x (px)'); axes[0, 2].set_ylabel('block y (px)')
    div = make_axes_locatable(axes[0, 2])
    plt.colorbar(im_d, cax=div.append_axes('right', '4%', 0.04),
                 label='ΔH (red = failure supervision raised H)')

    _heatmap(axes[1, 0], H_agt_full, ax_range, ay_range,
             'Agent position — Hθ (with failure)\n[block at canonical origin]',
             'agent x (px)', 'agent y (px)',
             block_px=(BLK_ORIGIN_X, BLK_ORIGIN_Y),
             vmin=vlim_agt[0], vmax=vlim_agt[1])
    _heatmap(axes[1, 1], H_agt_nofail, ax_range, ay_range,
             'Agent position — Hθ (no failure)\n[block at canonical origin]',
             'agent x (px)', 'agent y (px)',
             block_px=(BLK_ORIGIN_X, BLK_ORIGIN_Y),
             vmin=vlim_agt[0], vmax=vlim_agt[1])
    diff_agt = H_agt_full - H_agt_nofail
    im_d2 = axes[1, 2].imshow(diff_agt, origin='lower', aspect='auto',
                                extent=[ax_range[0], ax_range[-1],
                                        ay_range[0], ay_range[-1]],
                                cmap='RdBu_r',
                                vmin=-abs(diff_agt).max(), vmax=abs(diff_agt).max())
    axes[1, 2].scatter(BLK_ORIGIN_X, BLK_ORIGIN_Y, marker='s', s=150,
                        c='black', zorder=5)
    axes[1, 2].set_title('Agent position — Δ H (full − nofail)', fontsize=11,
                          fontweight='bold')
    axes[1, 2].set_xlabel('agent x (px)'); axes[1, 2].set_ylabel('agent y (px)')
    div2 = make_axes_locatable(axes[1, 2])
    plt.colorbar(im_d2, cax=div2.append_axes('right', '4%', 0.04),
                 label='ΔH (red = failure supervision raised H)')

    plt.suptitle(
        'Entropy field H(s,g) — bright (yellow) = low entropy = close to goal\n'
        'Cyan ★ = block goal   Green □ = block canonical origin',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  → {out_path}")


# ── Entropy along trajectory ───────────────────────────────────────────────────

def plot_entropy_trajectory(named_policy_enet_pairs, out_path):
    """
    Plot H(t) and coverage(t) for a list of (label, policy_fn, enet) tuples
    all starting from the first EVAL_STD seed.
    """
    ax0, ay0, bx0, by0, ba0 = EVAL_STD[0]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = ['#e74c3c', '#27ae60', '#95a5a6']
    g_t = torch.tensor(GOAL_VEC, dtype=torch.float32).unsqueeze(0)

    for (label, fn, enet), color in zip(named_policy_enet_pairs, colors):
        env = make_render_env()
        obs, _ = reset_env(env, agent_x=ax0, agent_y=ay0,
                           bx_off=bx0, by_off=by0, ba_off=ba0)
        if hasattr(fn, 'reset'):
            fn.reset(obs)
        H_vals, cov_vals = [], []
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                s_t = torch.tensor(raw_to_state(obs), dtype=torch.float32).unsqueeze(0)
                H_vals.append(float(enet(torch.cat([s_t, g_t], -1))))
            action = fn(obs)
            obs, reward, done, truncated, _ = env.step(action)
            cov_vals.append(float(reward))
            if done or truncated:
                break
        env.close()
        top.plot(H_vals,   color=color, lw=1.8, label=label, alpha=0.85)
        bot.plot(cov_vals, color=color, lw=1.8, label=label, alpha=0.85)

    top.axhline(math.log(200), color='grey', ls=':', lw=1.2, label='H_FAILURE')
    top.set_ylabel('H(s,g) = log(steps to goal)', fontsize=10)
    top.set_title('Entropy H(t) along episode — standard start', fontsize=11)
    top.legend(fontsize=9); top.grid(alpha=0.3)

    bot.axhline(SUCCESS_THR, color='grey', ls='--', lw=1.2,
                label=f'success threshold ({SUCCESS_THR})')
    bot.set_ylabel('Coverage (block-goal overlap)', fontsize=10)
    bot.set_xlabel('Step', fontsize=10)
    bot.set_title('Coverage over time', fontsize=11)
    bot.legend(fontsize=9); bot.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entropy', action='store_true',
                    help='only render entropy field (skip rollouts)')
    ap.add_argument('--rollout', action='store_true',
                    help='only render rollouts (skip entropy field)')
    ap.add_argument('--tries', type=int, default=8,
                    help='seeds to search per condition for best demo')
    args = ap.parse_args()

    do_rolls = not args.entropy
    do_ent   = not args.rollout

    print("=" * 60)
    print("viz_real.py — EFP-UNet rollouts + entropy field")
    print("=" * 60)

    policies, enet_full, enet_nofail, dyn_net = load_real_policies()
    os.makedirs(GIFS_DIR, exist_ok=True)

    if do_rolls:
        print("\n[1] Policy rollouts")
        do_rollouts(policies, n_tries=args.tries)

    if do_ent:
        print("\n[2] Entropy field")
        plot_entropy_field(
            enet_full, enet_nofail,
            os.path.join(GIFS_DIR, 'entropy_field.png'))

        print("\n[3] Entropy along trajectory")
        traj_pairs = [
            ("EFP (w/ failure)",  policies[0][2], enet_full),
            ("EFP (no failure)",  policies[1][2], enet_nofail),
            ("EFP (no guidance)", policies[2][2], enet_full),
        ]
        plot_entropy_trajectory(
            traj_pairs,
            os.path.join(GIFS_DIR, 'entropy_trajectory.png'))

    print(f"\nAll outputs in {GIFS_DIR}/")


if __name__ == '__main__':
    main()
