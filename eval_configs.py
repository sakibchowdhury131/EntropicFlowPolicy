"""
eval_configs.py
===============
Load a trained checkpoint and evaluate both Diffusion Policy and Heat
Dissipation Policy on a set of diverse PushT initial configurations.
Saves one side-by-side video per configuration.

Usage
-----
  python eval_configs.py                          # 16 random seeds
  python eval_configs.py --ckpt checkpoints/epoch_100.pt --n_configs 24
  python eval_configs.py --seeds 1 42 999 12345   # specific seeds
"""

import argparse, collections, copy, math, os
import cv2
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d

# ── re-use everything from the training script ──────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_dp_heat_pusht import (
    PushTEnv, ConditionalUnet1D, HeatDissipation1D,
    _normalize, _unnormalize, _ema_copy,
    OBS_DIM, ACTION_DIM, OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON,
    NUM_DDPM_STEPS, EVAL_MAX_STEPS, DEVICE,
)

BASE      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE, 'videos_eval')


# ── video helpers ────────────────────────────────────────────────────────────

def vwrite(path, imgs):
    h, w = imgs[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    for img in imgs:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()


def side_by_side(frames_dp, frames_heat, score_dp, score_heat, seed):
    """Interleave frames from both policies into a single side-by-side video."""
    n = max(len(frames_dp), len(frames_heat))
    # pad shorter one with its last frame
    while len(frames_dp)   < n: frames_dp.append(frames_dp[-1])
    while len(frames_heat) < n: frames_heat.append(frames_heat[-1])

    out = []
    for fd, fh in zip(frames_dp, frames_heat):
        # add text labels
        fd = fd.copy(); fh = fh.copy()
        cv2.putText(fd, f'DP  score={score_dp:.3f}',  (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 200), 1, cv2.LINE_AA)
        cv2.putText(fh, f'Heat score={score_heat:.3f}', (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 80, 10), 1, cv2.LINE_AA)
        cv2.putText(fd, f'seed={seed}', (4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)
        out.append(np.concatenate([fd, fh], axis=1))
    return out


# ── eval functions (copied from training script, same logic) ─────────────────

def eval_dp(net, stats, seed, max_steps=EVAL_MAX_STEPS):
    scheduler = DDPMScheduler(num_train_timesteps=NUM_DDPM_STEPS,
                              beta_schedule='squaredcos_cap_v2',
                              clip_sample=True, prediction_type='epsilon')
    env = PushTEnv()
    env.seed(seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    imgs, rewards = [env.render()], []
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
            imgs.append(env.render())
            buf_idx += 1; step_idx += 1
            if step_idx >= max_steps: done = True

    return imgs, float(max(rewards)) if rewards else 0.0


def eval_heat(net, heat, stats, seed, max_steps=EVAL_MAX_STEPS):
    env = PushTEnv()
    env.seed(seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    imgs, rewards = [env.render()], []
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
            imgs.append(env.render())
            buf_idx += 1; step_idx += 1
            if step_idx >= max_steps: done = True

    return imgs, float(max(rewards)) if rewards else 0.0


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',      default=os.path.join(BASE, 'checkpoints', 'epoch_100.pt'))
    ap.add_argument('--n_configs', type=int, default=16,
                    help='number of random seeds to evaluate (ignored if --seeds given)')
    ap.add_argument('--seeds',     type=int, nargs='+', default=None,
                    help='explicit list of seeds to evaluate')
    ap.add_argument('--heat_steps', type=int,   default=100)
    ap.add_argument('--alpha_max',  type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── load checkpoint ───────────────────────────────────────────────────
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
    heat      = HeatDissipation1D(num_timesteps=args.heat_steps, alpha_max=args.alpha_max)

    print(f"  epoch={ckpt['epoch']}  device={DEVICE}")

    # ── pick seeds ────────────────────────────────────────────────────────
    if args.seeds is not None:
        seeds = args.seeds
    else:
        rng   = np.random.RandomState(0)
        seeds = rng.randint(0, 1_000_000, size=args.n_configs).tolist()

    print(f"\nEvaluating {len(seeds)} configurations ...\n")

    results = []
    for i, seed in enumerate(seeds):
        dp_imgs,   dp_score   = eval_dp(dp_eval,   stats, seed=seed)
        heat_imgs, heat_score = eval_heat(heat_eval, heat, stats, seed=seed)

        frames = side_by_side(dp_imgs, heat_imgs, dp_score, heat_score, seed)
        fname  = os.path.join(OUT_DIR,
                              f'config_{i+1:03d}_seed_{seed}_dp{dp_score:.3f}_heat{heat_score:.3f}.mp4')
        vwrite(fname, frames)

        results.append((seed, dp_score, heat_score))
        print(f'  [{i+1:3d}/{len(seeds)}]  seed={seed:<8d}  '
              f'dp={dp_score:.3f}  heat={heat_score:.3f}')

    # ── summary ───────────────────────────────────────────────────────────
    dp_arr   = np.array([r[1] for r in results])
    heat_arr = np.array([r[2] for r in results])
    print(f'\n{"─"*50}')
    print(f'  Diffusion Policy  — mean={dp_arr.mean():.3f}  '
          f'min={dp_arr.min():.3f}  max={dp_arr.max():.3f}')
    print(f'  Heat Dissipation  — mean={heat_arr.mean():.3f}  '
          f'min={heat_arr.min():.3f}  max={heat_arr.max():.3f}')
    print(f'\nVideos saved → {OUT_DIR}/')


if __name__ == '__main__':
    main()
