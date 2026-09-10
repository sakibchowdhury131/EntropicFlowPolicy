"""
viz_pusht.py — Render EFP policy rollouts and training demos on Push-T.

Usage
-----
  python viz_pusht.py              # policy rollouts (loads saved models)
  python viz_pusht.py --demos      # training demo rollouts (no models needed)
  python viz_pusht.py --retrain    # retrain all models, then show rollouts
  python viz_pusht.py --quick      # fast retrain (fewer epochs)
  python viz_pusht.py --tries N    # seeds to try per condition (default 10)

Output (policy mode)
  gifs/compare_standard.gif        all 5 policies from best standard seed
  gifs/compare_perturb.gif         all 5 policies from best perturb seed
  gifs/<cond>_<policy>.gif         individual episodes

Output (--demos mode)
  gifs/demos/expert_left_<N>.gif   LEFT-detour expert demos
  gifs/demos/expert_right_<N>.gif  RIGHT-detour expert demos
  gifs/demos/failure_<N>.gif       failure demos (agent descends onto block)
  gifs/demos/compare_demos.gif     side-by-side: LEFT | RIGHT | failure
"""

import sys, os, math, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio.v2 as iio
import gymnasium as gym
import gym_pusht  # noqa: F401 — registers env

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from efp_pusht import (
    EntropyNet, DynNet, VelocityNet, BCNet, MDNNet,
    _infer_flow, _infer_bc, _infer_mdn,
    raw_to_state, reset_env,
    generate_demos, build_datasets,
    train_entropy, train_dynamics, train_flow, train_bc, train_mdn,
    GUIDANCE_SCALE, SUCCESS_THR, IMG,
    BLK_ORIGIN_X, BLK_ORIGIN_Y, BLK_INIT_ANG,
)

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, 'models')
GIFS_DIR   = os.path.join(BASE, 'gifs')

MAX_STEPS  = 400   # gym-pusht truncates at 300; allow a few more for wrap-around


# ── Model I/O ──────────────────────────────────────────────────────────────────

def save_models(enet_full, enet_nofail, dyn_net, vel_net, bc_net, mdn_net):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, net in [('enet_full', enet_full), ('enet_nofail', enet_nofail),
                      ('dyn_net',   dyn_net),   ('vel_net',     vel_net),
                      ('bc_net',    bc_net),    ('mdn_net',     mdn_net)]:
        torch.save(net.state_dict(), f'{MODELS_DIR}/{name}.pt')
    print(f"Models saved → {MODELS_DIR}/")


def load_models():
    classes = [EntropyNet, EntropyNet, DynNet, VelocityNet, BCNet, MDNNet]
    names   = ['enet_full', 'enet_nofail', 'dyn_net', 'vel_net', 'bc_net', 'mdn_net']
    nets = []
    for cls, name in zip(classes, names):
        path = f'{MODELS_DIR}/{name}.pt'
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run  python viz_pusht.py --retrain  to train first.")
        net = cls()
        net.load_state_dict(torch.load(path))
        nets.append(net)
    print(f"Models loaded ← {MODELS_DIR}/")
    return nets


def do_retrain(quick=False):
    n_s  = 100 if quick else 300
    n_f  = 50  if quick else 120
    f_ep = 150 if quick else 300
    e_ep = 60  if quick else 120
    d_ep = 20  if quick else 40
    b_ep = 40  if quick else 80

    succ, fail = generate_demos(n_success=n_s, n_failure=n_f)
    S, A, SN, H, G, NOISE, SF, HF, GF, SF_S, SF_A, SF_SN = build_datasets(succ, fail)

    enet_full = EntropyNet()
    train_entropy(enet_full,   S, H, G, SF, HF, GF, use_failure=True,  epochs=e_ep)
    enet_nofail = EntropyNet()
    train_entropy(enet_nofail, S, H, G, SF, HF, GF, use_failure=False, epochs=e_ep)
    dyn_net = DynNet()
    train_dynamics(dyn_net,
                   torch.cat([S, SF_S]), torch.cat([A, SF_A]), torch.cat([SN, SF_SN]),
                   epochs=d_ep)
    vel_net = VelocityNet()
    train_flow(vel_net, S, G, A, NOISE, epochs=f_ep)
    bc_net = BCNet()
    train_bc(bc_net, S, G, A, epochs=b_ep)
    mdn_net = MDNNet()
    train_mdn(mdn_net, S, G, A, epochs=b_ep)

    save_models(enet_full, enet_nofail, dyn_net, vel_net, bc_net, mdn_net)
    return enet_full, enet_nofail, dyn_net, vel_net, bc_net, mdn_net


# ── Rendering helpers ──────────────────────────────────────────────────────────

def make_render_env():
    return gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")


def run_episode(policy_fn, agent_x, agent_y,
                bx_off=0.0, by_off=0.0, ba_off=0.0):
    env = make_render_env()
    obs, _ = reset_env(env, agent_x=agent_x, agent_y=agent_y,
                       bx_off=bx_off, by_off=by_off, ba_off=ba_off)
    frames  = [env.render()]
    max_cov = 0.0
    for _ in range(MAX_STEPS):
        action = policy_fn(raw_to_state(obs))
        obs, reward, done, truncated, _ = env.step(action)
        max_cov = max(max_cov, float(reward))
        frames.append(env.render())
        if done or truncated:
            break
    env.close()
    return frames, max_cov


def _annotate(frame_np, text, bg=(20, 20, 20), fg=(255, 255, 255)):
    img = Image.fromarray(frame_np)
    d   = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 28], fill=bg)
    d.text((8, 7), text, fill=fg)
    return np.array(img)


def save_gif(frames, path, fps=15, label=None, size=None):
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
    """Tile N policy episodes side-by-side into one GIF."""
    n     = len(all_frames)
    max_t = max(len(f) for f in all_frames)
    padded = [f + [f[-1]] * (max_t - len(f)) for f in all_frames]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = []
    for t in range(max_t):
        cells = []
        for i in range(n):
            img = Image.fromarray(padded[i][t]).resize((cell, cell), Image.LANCZOS)
            d   = ImageDraw.Draw(img)
            d.rectangle([0, 0, cell, 26], fill=(20, 20, 20))
            d.text((5, 6), labels[i][:26], fill=(255, 255, 255))
            cells.append(np.array(img))
        out.append(np.concatenate(cells, axis=1).astype(np.uint8))
    iio.mimwrite(path, out, fps=fps)
    print(f"    → {path}  ({len(out)} frames, {n * cell}×{cell}px)")


# ── Seed search ────────────────────────────────────────────────────────────────

def find_best_init(policy_fn, base_ax, base_ay, n_tries=10,
                   ax_range=(-8, 8), ay_range=(-10, 10),
                   bx_range=(-8, 8), by_range=(-8, 8), ba_range=(-0.1, 0.1)):
    """
    Try n_tries jittered starting positions; return the one where policy_fn
    achieves the highest coverage (i.e. best single episode).
    """
    rng = np.random.RandomState(77)
    best_cov, best_init = -1.0, None
    for _ in range(n_tries):
        init = dict(
            agent_x = base_ax + rng.uniform(*ax_range),
            agent_y = base_ay + rng.uniform(*ay_range),
            bx_off  = rng.uniform(*bx_range),
            by_off  = rng.uniform(*by_range),
            ba_off  = rng.uniform(*ba_range),
        )
        _, cov = run_episode(policy_fn, **init)
        if cov > best_cov:
            best_cov, best_init = cov, init
    return best_init, best_cov


# ── Training demo rendering ────────────────────────────────────────────────────

def _render_walk_to(env, obs, target, max_steps=200):
    """Step env toward pixel target; return (obs, frames)."""
    frames = []
    target = np.array(target, dtype=np.float64)
    for _ in range(max_steps):
        frames.append(env.render())
        if np.linalg.norm(obs[:2] - target) < 8:
            break
        action = np.clip(target, 0, IMG).astype(np.float32)
        obs, _reward, done, truncated, _ = env.step(action)
        if done or truncated:
            frames.append(env.render())
            break
    return obs, frames


def render_expert_episode(mode='LEFT', seed=0):
    """
    Scripted expert with rgb_array rendering.
    Mirrors run_expert_episode() but captures frames inline.
    """
    rng = np.random.RandomState(seed)
    env = make_render_env()
    obs, _ = reset_env(env,
        agent_x = 256 + rng.uniform(-8, 8),
        agent_y = 50  + rng.uniform(-5, 5),
        bx_off  = rng.uniform(-8, 8),
        by_off  = rng.uniform(-8, 8),
        ba_off  = rng.uniform(-0.1, 0.1),
    )

    block_com = obs[2:4].copy()
    detour_x  = 80.0  if mode == 'LEFT' else 430.0
    x_off     = +20   if mode == 'LEFT' else -10
    ax        = block_com[0] + x_off
    ay        = block_com[1] + 85
    push_y    = 250.0
    bottom_y  = 490.0

    all_frames = [env.render()]
    phases = [
        np.array([detour_x, 50.0]),
        np.array([detour_x, bottom_y]),
        np.array([ax,       bottom_y]),
        np.array([ax,       ay      ]),
        np.array([ax,       push_y  ]),
    ]
    for tgt in phases:
        obs, frames = _render_walk_to(env, obs, tgt)
        all_frames.extend(frames)

    env.close()
    return all_frames


def render_failure_episode(seed=0):
    """
    Failure demo with rgb_array rendering.
    Agent starts above block, descends straight down → pushes block away from goal.
    """
    rng = np.random.RandomState(seed)
    env = make_render_env()
    obs, _ = reset_env(env,
        agent_x = 288 + rng.uniform(-8, 8),
        agent_y = 50  + rng.uniform(-5, 5),
        bx_off  = rng.uniform(-5, 5),
        by_off  = rng.uniform(-5, 5),
        ba_off  = rng.uniform(-0.05, 0.05),
    )
    target    = np.array([obs[0], 490.0])
    all_frames = [env.render()]
    _, frames  = _render_walk_to(env, obs, target, max_steps=300)
    all_frames.extend(frames)
    env.close()
    return all_frames


def show_training_demos(n_each=3):
    """Render n_each episodes of each demo type and save GIFs."""
    demos_dir = os.path.join(GIFS_DIR, 'demos')
    os.makedirs(demos_dir, exist_ok=True)

    demo_specs = (
        [('LEFT',  s) for s in range(n_each)] +
        [('RIGHT', s) for s in range(n_each)] +
        [('FAIL',  s) for s in range(n_each)]
    )

    # Collect one representative of each type for the comparison strip
    rep_frames = {'LEFT': None, 'RIGHT': None, 'FAIL': None}

    for dtype, seed in demo_specs:
        print(f"  Rendering {dtype} demo seed={seed}...", end=' ', flush=True)
        if dtype == 'FAIL':
            frames = render_failure_episode(seed=seed)
            label  = f"FAILURE demo (seed {seed})  [H = log(200) = 5.30]"
            fname  = f"failure_{seed}.gif"
        else:
            frames = render_expert_episode(mode=dtype, seed=seed)
            label  = f"Expert {dtype} (seed {seed})  [H decreases to goal]"
            fname  = f"expert_{dtype.lower()}_{seed}.gif"
        print(f"{len(frames)} frames")

        save_gif(frames,
                 os.path.join(demos_dir, fname),
                 fps=20, label=label, size=(340, 340))

        if seed == 0:
            rep_frames[dtype] = frames

    # Side-by-side comparison: LEFT expert | RIGHT expert | failure
    print("  Building demo comparison GIF...")
    save_comparison_gif(
        [rep_frames['LEFT'], rep_frames['RIGHT'], rep_frames['FAIL']],
        ['Expert LEFT  [success path]',
         'Expert RIGHT  [success path]',
         'FAILURE demo  [H = 5.30]'],
        os.path.join(demos_dir, 'compare_demos.gif'),
        fps=20, cell=240,
    )
    print(f"\nTraining demo GIFs in {demos_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos',   action='store_true',
                    help='render training demonstrations (no models needed)')
    ap.add_argument('--retrain', action='store_true')
    ap.add_argument('--quick',   action='store_true')
    ap.add_argument('--tries',   type=int, default=10,
                    help='seeds to search for best policy demo per condition')
    args = ap.parse_args()

    if args.demos:
        print("=== Training demo rollouts ===")
        show_training_demos(n_each=3)
        return

    # Load or train models
    if args.retrain or not os.path.exists(f'{MODELS_DIR}/vel_net.pt'):
        mode = "quick" if args.quick else "full"
        print(f"Training models ({mode} mode)...")
        enet_full, enet_nofail, dyn_net, vel_net, bc_net, mdn_net = do_retrain(args.quick)
    else:
        nets = load_models()
        enet_full, enet_nofail, dyn_net, vel_net, bc_net, mdn_net = nets

    _G = GUIDANCE_SCALE
    ef, en, vel, dyn, bc, mdn = enet_full, enet_nofail, vel_net, dyn_net, bc_net, mdn_net

    # Policies — default-arg binding for safe closure capture
    policies = [
        ("EFP_failure",      "EFP (w/ failure)",
         lambda s, e=ef, v=vel, d=dyn, g=_G: _infer_flow(s, v, e, d, guidance=g)),
        ("EFP_nofailure",    "EFP (no failure)",
         lambda s, e=en, v=vel, d=dyn, g=_G: _infer_flow(s, v, e, d, guidance=g)),
        ("Diffusion_Policy", "Diffusion Policy",
         lambda s, v=vel: _infer_flow(s, v, guidance=0.0)),
        ("BC",               "BC (Mean)",
         lambda s, b=bc:  _infer_bc(s, b)),
        ("MDN_BC",           "MDN-BC",
         lambda s, m=mdn: _infer_mdn(s, m)),
    ]

    # Conditions: search for the best seed for EFP(failure), then replay all policies
    # from that same starting state so the comparison is fair.
    conditions = [
        ("standard",
         dict(base_ax=256, base_ay=50,
              ax_range=(-5, 5),   ay_range=(-3, 3),
              bx_range=(-8, 8),   by_range=(-8, 8),   ba_range=(-0.1, 0.1))),
        ("perturb",
         dict(base_ax=288, base_ay=280,
              ax_range=(-8, 8),   ay_range=(-10, 10),
              bx_range=(-8, 8),   by_range=(-8,  8),  ba_range=(-0.1, 0.1))),
    ]

    os.makedirs(GIFS_DIR, exist_ok=True)
    efp_failure_fn = policies[0][2]

    for cond_key, search_kwargs in conditions:
        print(f"\n=== {cond_key.upper()} ===")
        print(f"  Searching {args.tries} seeds for best EFP(failure) demo...", flush=True)
        best_init, best_cov = find_best_init(efp_failure_fn, n_tries=args.tries,
                                              **search_kwargs)
        tag = "SUCCESS" if best_cov > SUCCESS_THR else f"cov={best_cov:.2f}"
        print(f"  Best seed: EFP(failure) {tag}"
              f"  agent=({best_init['agent_x']:.0f},{best_init['agent_y']:.0f})")

        all_frames, all_labels = [], []

        for file_key, display, fn in policies:
            print(f"  Rolling out {display}...", end=' ', flush=True)
            frames, cov = run_episode(fn, **best_init)
            ok  = cov > SUCCESS_THR
            ep_tag = "SUCCESS" if ok else f"fail  cov={cov:.2f}"
            print(f"[{ep_tag}]  {len(frames)} steps")

            gif_label = f"{display}  [{ep_tag}]"
            save_gif(frames,
                     os.path.join(GIFS_DIR, f'{cond_key}_{file_key}.gif'),
                     fps=15, label=gif_label, size=(340, 340))

            all_frames.append(frames)
            all_labels.append(f"{display}  [{ep_tag}]")

        print(f"  Building comparison GIF...")
        save_comparison_gif(all_frames, all_labels,
                            os.path.join(GIFS_DIR, f'compare_{cond_key}.gif'),
                            fps=15, cell=200)

    print(f"\nDone!  GIFs in {GIFS_DIR}/")
    for cond_key, _ in conditions:
        print(f"  {GIFS_DIR}/compare_{cond_key}.gif")


if __name__ == '__main__':
    main()
