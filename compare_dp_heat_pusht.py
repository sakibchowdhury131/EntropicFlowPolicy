"""
compare_dp_heat_pusht.py
========================
Train Diffusion Policy (DDPM) and Heat Dissipation Policy on PushT
side-by-side from the same dataset and same epoch loop.

After every epoch both policies run one evaluation episode and save a video:
  videos_dp/epoch_001_score_0.000.mp4
  videos_heat/epoch_001_score_0.000.mp4

A score-curve plot is saved as comparison_scores.png when training ends.

Usage
-----
  python compare_dp_heat_pusht.py
  python compare_dp_heat_pusht.py --epochs 50 --heat_steps 50 --alpha_max 6.0
"""

import argparse, collections, copy, math, os
from typing import Tuple, Sequence

import cv2
import gdown
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
import skimage.transform as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
from tqdm.auto import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

BASE            = os.path.dirname(os.path.abspath(__file__))
ZARR_PATH       = os.path.join(BASE, 'pusht_cchi_v7_replay.zarr.zip')
VIDEO_DP_DIR    = os.path.join(BASE, 'videos_dp')
VIDEO_HEAT_DIR  = os.path.join(BASE, 'videos_heat')
CKPT_DIR        = os.path.join(BASE, 'checkpoints')
SCORE_LOG       = os.path.join(BASE, 'scores.csv')

OBS_DIM        = 5
ACTION_DIM     = 2
OBS_HORIZON    = 2
PRED_HORIZON   = 16
ACTION_HORIZON = 8
NUM_DDPM_STEPS = 100
EVAL_SEED      = 1234
EVAL_MAX_STEPS = 300
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Video writer (cv2, no ffmpeg needed) ───────────────────────────────────────

def vwrite(path, imgs):
    h, w = imgs[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    for img in imgs:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()

# ── PushT Environment ──────────────────────────────────────────────────────────

positive_y_is_up = False

def _to_pygame(p, surface):
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    return round(p[0]), round(p[1])

def _light_color(color):
    c = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), 255.)
    return SpaceDebugColor(r=c[0], g=c[1], b=c[2], a=c[3])

class _DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface):
        self.surface = surface
        super().__init__()

    def draw_circle(self, pos, angle, radius, outline_color, fill_color):
        p = _to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, _light_color(fill_color).as_int(), p, round(radius - 4), 0)

    def draw_segment(self, a, b, color):
        pygame.draw.aalines(self.surface, color.as_int(), False,
                            [_to_pygame(a, self.surface), _to_pygame(b, self.surface)])

    def draw_fat_segment(self, a, b, radius, outline_color, fill_color):
        p1, p2 = _to_pygame(a, self.surface), _to_pygame(b, self.surface)
        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orth = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orth[0] == 0 and orth[1] == 0:
                return
            scale = radius / (orth[0]**2 + orth[1]**2) ** 0.5
            orth = [round(orth[0] * scale), round(orth[1] * scale)]
            pts = [(p1[0]-orth[0], p1[1]-orth[1]), (p1[0]+orth[0], p1[1]+orth[1]),
                   (p2[0]+orth[0], p2[1]+orth[1]), (p2[0]-orth[0], p2[1]-orth[1])]
            pygame.draw.polygon(self.surface, fill_color.as_int(), pts)
            for p in (p1, p2):
                pygame.draw.circle(self.surface, fill_color.as_int(),
                                   (round(p[0]), round(p[1])), round(radius))

    def draw_polygon(self, verts, radius, outline_color, fill_color):
        ps = [_to_pygame(v, self.surface) for v in verts] + [_to_pygame(verts[0], self.surface)]
        pygame.draw.polygon(self.surface, _light_color(fill_color).as_int(), ps)
        if radius > 0:
            for i in range(len(verts)):
                self.draw_fat_segment(verts[i], verts[(i+1) % len(verts)],
                                      2, fill_color, fill_color)

    def draw_dot(self, size, pos, color):
        pygame.draw.circle(self.surface, color.as_int(), _to_pygame(pos, self.surface), round(size), 0)


def _pymunk_to_shapely(body, shapes):
    geoms = [sg.Polygon([body.local_to_world(v) for v in s.get_vertices()] + [body.local_to_world(s.get_vertices()[0])])
             for s in shapes if isinstance(s, pymunk.shapes.Poly)]
    return sg.MultiPolygon(geoms)


class PushTEnv:
    metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, render_size=96, reset_to_state=None):
        self._seed = None
        self.seed()
        self.window_size = 512
        self.render_size = render_size
        self.sim_hz      = 100
        self.k_p, self.k_v = 100, 20
        self.control_hz  = 10
        self.reset_to_state = reset_to_state
        self.window = self.clock = self.screen = None
        self.space  = None
        self.render_buffer = None
        self.latest_action = None
        self.success_threshold = 0.95

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed

    def reset(self):
        self._setup()
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=self._seed)
            state = np.array([rs.randint(50, 450), rs.randint(50, 450),
                              rs.randint(100, 400), rs.randint(100, 400),
                              rs.randn() * 2 * np.pi - np.pi])
        self._set_state(state)
        return self._get_obs(), self._get_info()

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for _ in range(n_steps):
                acc = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0,0) - self.agent.velocity)
                self.agent.velocity += acc * dt
                self.space.step(dt)
        goal_body = self._goal_body()
        goal_geom  = _pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = _pymunk_to_shapely(self.block,  self.block.shapes)
        coverage   = goal_geom.intersection(block_geom).area / goal_geom.area
        reward     = np.clip(coverage / self.success_threshold, 0, 1)
        done       = coverage > self.success_threshold
        return self._get_obs(), reward, done, done, self._get_info()

    def render(self, mode='rgb_array'):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas
        opts = _DrawOptions(canvas)
        goal_body = self._goal_body()
        for shape in self.block.shapes:
            gpts = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), opts.surface)
                    for v in shape.get_vertices()]
            pygame.draw.polygon(canvas, self.goal_color, gpts)
        self.space.debug_draw(opts)
        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.latest_action is not None:
            coord = (np.array(self.latest_action) / 512 * 96).astype(np.int32)
            cv2.drawMarker(img, coord, (255, 0, 0), cv2.MARKER_CROSS, 8, 1)
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit(); pygame.quit()

    # ── private ──────────────────────────────────────────────────────────────

    def _get_obs(self):
        return np.array(tuple(self.agent.position) + tuple(self.block.position) +
                        (self.block.angle % (2 * np.pi),))

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        return {'pos_agent': np.array(self.agent.position),
                'vel_agent': np.array(self.agent.velocity),
                'block_pose': np.array(list(self.block.position) + [self.block.angle]),
                'n_contacts': int(np.ceil(self.n_contact_points / n_steps))}

    def _goal_body(self):
        mass = 1
        body = pymunk.Body(mass, pymunk.moment_for_box(mass, (50, 100)))
        body.position = self.goal_pose[:2].tolist()
        body.angle    = self.goal_pose[2]
        return body

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        self.agent.position = state[:2]
        self.block.angle    = state[4]
        self.block.position = state[2:4]
        self.space.step(1.0 / self.sim_hz)

    def _setup(self):
        pygame.init()
        self.space   = pymunk.Space()
        self.space.gravity  = 0, 0
        self.space.damping  = 0
        self.render_buffer  = []
        ws = self.window_size
        walls = [pymunk.Segment(self.space.static_body, a, b, 2)
                 for a, b in [((5,506),(5,5)), ((5,5),(506,5)),
                               ((506,5),(506,506)), ((5,506),(506,506))]]
        for w in walls:
            w.color = pygame.Color('LightGray')
        self.space.add(*walls)
        self.agent      = self._add_circle((256, 400), 15)
        self.block      = self._add_tee((256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose  = np.array([256, 256, np.pi / 4])
        handler = self.space.add_collision_handler(0, 0)
        handler.post_solve = lambda arb, sp, data: setattr(
            self, 'n_contact_points',
            self.n_contact_points + len(arb.contact_point_set.points))
        self.n_contact_points = 0

    def _add_circle(self, position, radius):
        body        = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        shape       = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def _add_tee(self, position, angle, scale=30, color='LightSlateGray'):
        mass = 1
        L    = 4
        v1   = [(-L*scale/2,scale),(L*scale/2,scale),(L*scale/2,0),(-L*scale/2,0)]
        v2   = [(-scale/2,scale),(-scale/2,L*scale),(scale/2,L*scale),(scale/2,scale)]
        body = pymunk.Body(mass, pymunk.moment_for_poly(mass,v1) + pymunk.moment_for_poly(mass,v1))
        s1, s2 = pymunk.Poly(body, v1), pymunk.Poly(body, v2)
        for s in (s1, s2):
            s.color  = pygame.Color(color)
            s.filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS())
        body.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
        body.position = position
        body.angle    = angle
        self.space.add(body, s1, s2)
        return body

# ── Dataset ────────────────────────────────────────────────────────────────────

def _create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        end_idx   = episode_ends[i]
        ep_len    = end_idx - start_idx
        for idx in range(-pad_before, ep_len - sequence_length + pad_after + 1):
            buf_s = max(idx, 0) + start_idx
            buf_e = min(idx + sequence_length, ep_len) + start_idx
            sam_s = buf_s - (idx + start_idx)
            sam_e = sequence_length - ((idx + sequence_length + start_idx) - buf_e)
            indices.append([buf_s, buf_e, sam_s, sam_e])
    return np.array(indices)


def _sample_sequence(train_data, seq_len, buf_s, buf_e, sam_s, sam_e):
    result = {}
    for key, arr in train_data.items():
        sample = arr[buf_s:buf_e]
        if sam_s > 0 or sam_e < seq_len:
            data = np.zeros((seq_len,) + arr.shape[1:], dtype=arr.dtype)
            if sam_s > 0:   data[:sam_s] = sample[0]
            if sam_e < seq_len: data[sam_e:] = sample[-1]
            data[sam_s:sam_e] = sample
        else:
            data = sample
        result[key] = data
    return result


def _get_stats(data):
    d = data.reshape(-1, data.shape[-1])
    return {'min': d.min(0), 'max': d.max(0)}

def _normalize(data, stats):
    return ((data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)).astype(np.float32)

def _unnormalize(ndata, stats):
    return ndata * (stats['max'] - stats['min']) + stats['min']


class PushTDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_path):
        store = zarr.storage.ZipStore(zarr_path, mode='r')
        root  = zarr.open(store=store, mode='r')
        obs_raw = root['data']['state'][:]    # (N, 5)
        act_raw = root['data']['action'][:]   # (N, 2)
        ends    = root['meta']['episode_ends'][:]

        self.stats = {'obs': _get_stats(obs_raw), 'action': _get_stats(act_raw)}
        norm_obs = _normalize(obs_raw, self.stats['obs'])
        norm_act = _normalize(act_raw, self.stats['action'])

        self.indices = _create_sample_indices(
            ends, PRED_HORIZON,
            pad_before=OBS_HORIZON - 1,
            pad_after=ACTION_HORIZON - 1)
        self.data = {'obs': norm_obs, 'action': norm_act}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buf_s, buf_e, sam_s, sam_e = self.indices[idx]
        sample = _sample_sequence(self.data, PRED_HORIZON, buf_s, buf_e, sam_s, sam_e)
        obs    = torch.from_numpy(sample['obs'][:OBS_HORIZON])     # (obs_horizon, 5)
        action = torch.from_numpy(sample['action'])                # (pred_horizon, 2)
        return obs, action

# ── Heat Dissipation 1D ────────────────────────────────────────────────────────

class HeatDissipation1D:
    """Pure 1-D heat blur — no noise.

    Forward:  a_t = blur(a_0, t)          (FFT-based, mass-conserving)
    The DC component (trajectory mean) is always preserved exactly.
    With [0,1]-normalised actions, different trajectories start the
    reverse process from different flat levels in [0,1], giving the
    'many energy levels' property.  alpha_max=10 kills the fundamental
    frequency for seq_len=16 to <0.05%, so the blurred signal at t=T-1
    is a near-perfect flat line at the trajectory mean.
    """

    def __init__(self, num_timesteps=100, alpha_max=10.0):
        self.T      = num_timesteps
        self.sigmas = torch.sqrt(torch.linspace(0.01**2, alpha_max**2, num_timesteps))
        self._cache = {}

    def _filt(self, sigma, seq_len, device):
        key = (round(sigma, 6), seq_len)
        if key not in self._cache:
            freq = torch.fft.fftfreq(seq_len, device=device)
            self._cache[key] = torch.exp(-2 * math.pi**2 * sigma**2 * freq**2)
        return self._cache[key]

    def blur(self, x, t_idx):
        sigma = self.sigmas[t_idx].item()
        filt  = self._filt(sigma, x.shape[1], x.device)
        xp    = x.permute(0, 2, 1)
        return torch.fft.ifft(torch.fft.fft(xp, dim=-1) * filt[None, None, :],
                              dim=-1).real.permute(0, 2, 1)

    def blur_batch(self, x, t):
        out = torch.zeros_like(x)
        for ti in t.unique():
            mask = t == ti
            out[mask] = self.blur(x[mask], ti.item())
        return out

# ── Shared UNet architecture ───────────────────────────────────────────────────

class _SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb  = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class _Conv1dBlock(nn.Module):
    def __init__(self, inp, out, ks, ng=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(inp, out, ks, padding=ks//2),
                                   nn.GroupNorm(ng, out), nn.Mish())
    def forward(self, x): return self.block(x)

class _CondResBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_dim, ks=3, ng=8):
        super().__init__()
        self.b1   = _Conv1dBlock(in_c, out_c, ks, ng)
        self.b2   = _Conv1dBlock(out_c, out_c, ks, ng)
        self.cenc = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_c*2),
                                  nn.Unflatten(-1, (-1, 1)))
        self.res  = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.out_c = out_c
    def forward(self, x, cond):
        out  = self.b1(x)
        emb  = self.cenc(cond).reshape(cond.shape[0], 2, self.out_c, 1)
        out  = emb[:, 0] * out + emb[:, 1]
        return self.b2(out) + self.res(x)

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim,
                 dsed=256, down_dims=(256, 512, 1024), ks=5, ng=8):
        super().__init__()
        dims     = [input_dim] + list(down_dims)
        cond_dim = dsed + global_cond_dim
        in_out   = list(zip(dims[:-1], dims[1:]))

        self.emb = nn.Sequential(_SinEmb(dsed), nn.Linear(dsed, dsed*4),
                                 nn.Mish(), nn.Linear(dsed*4, dsed))
        self.mid = nn.ModuleList([_CondResBlock(dims[-1], dims[-1], cond_dim, ks, ng),
                                  _CondResBlock(dims[-1], dims[-1], cond_dim, ks, ng)])
        self.down = nn.ModuleList([
            nn.ModuleList([_CondResBlock(di, do, cond_dim, ks, ng),
                           _CondResBlock(do, do, cond_dim, ks, ng),
                           nn.Conv1d(do, do, 3, 2, 1) if i < len(in_out)-1 else nn.Identity()])
            for i, (di, do) in enumerate(in_out)])
        self.up = nn.ModuleList([
            nn.ModuleList([_CondResBlock(do*2, di, cond_dim, ks, ng),
                           _CondResBlock(di, di, cond_dim, ks, ng),
                           nn.ConvTranspose1d(di, di, 4, 2, 1) if i < len(in_out)-1 else nn.Identity()])
            for i, (di, do) in enumerate(reversed(in_out[1:]))])
        self.final = nn.Sequential(_Conv1dBlock(down_dims[0], down_dims[0], ks),
                                   nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, sample, timestep, global_cond=None):
        x  = sample.moveaxis(-1, -2)
        ts = torch.as_tensor(timestep, device=x.device).expand(x.shape[0]).long()
        gf = self.emb(ts)
        if global_cond is not None:
            gf = torch.cat([gf, global_cond], dim=-1)
        h = []
        for r1, r2, ds in self.down:
            x = r2(r1(x, gf), gf); h.append(x); x = ds(x)
        for m in self.mid:
            x = m(x, gf)
        for r1, r2, us in self.up:
            x = r2(r1(torch.cat([x, h.pop()], 1), gf), gf); x = us(x)
        return self.final(x).moveaxis(-1, -2)

# ── Evaluation ─────────────────────────────────────────────────────────────────

def _ema_copy(net, ema):
    """Return a deepcopy of net with EMA weights applied, set to eval mode."""
    e = copy.deepcopy(net)
    ema.copy_to(e.parameters())
    e.eval()
    return e


def eval_dp(net, stats, seed=EVAL_SEED, max_steps=EVAL_MAX_STEPS):
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
            if step_idx >= max_steps:
                done = True

    return imgs, float(max(rewards)) if rewards else 0.0


def eval_heat(net, heat, stats, seed=EVAL_SEED, max_steps=EVAL_MAX_STEPS,
              prediction_type='x0'):
    env = PushTEnv()
    env.seed(seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    imgs, rewards = [env.render()], []
    done, step_idx, action_buf, buf_idx = False, 0, None, 0

    # Fix DC for the entire episode so the trajectory mean is consistent
    # across all replanning steps. obs_cond guides the shape; DC sets the level.
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
                    if prediction_type == 'residual':
                        # pred = b_{k-1} - b_k  →  just add, no re-blur
                        nact = nact + pred
                    else:
                        pred_a0 = pred.clamp(0, 1)
                        nact = heat.blur(pred_a0, k-1) if k > 0 else pred_a0
                nact = nact.clamp(0, 1)
                action_buf = _unnormalize(nact[0].cpu().numpy(), stats['action'])
                buf_idx = 0

            obs, reward, done, _, _ = env.step(action_buf[buf_idx])
            obs_deque.append(obs)
            rewards.append(reward)
            imgs.append(env.render())
            buf_idx += 1; step_idx += 1
            if step_idx >= max_steps:
                done = True

    return imgs, float(max(rewards)) if rewards else 0.0

# ── Training ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',     type=int,   default=100)
    ap.add_argument('--heat_steps', type=int,   default=100)
    ap.add_argument('--alpha_max',  type=float, default=10.0)
    ap.add_argument('--dc_shift',        type=float, default=0.5,
                    help='max DC augmentation shift during training (0=off)')
    ap.add_argument('--prediction_type', type=str,   default='x0',
                    choices=['x0', 'residual'],
                    help='heat prediction target: x0=clean action, residual=a0-a_t')
    ap.add_argument('--batch_size', type=int,   default=256)
    ap.add_argument('--lr',         type=float, default=1e-4)
    ap.add_argument('--wd',         type=float, default=1e-6)
    ap.add_argument('--warmup',     type=int,   default=500)
    ap.add_argument('--outdir',     type=str,   default=None,
                    help='output directory (default: script directory)')
    ap.add_argument('--resume',     type=str,   default=None,
                    help='checkpoint .pt to resume from')
    ap.add_argument('--video_every', type=int,  default=10,
                    help='save eval video every N epochs (0=never)')
    args = ap.parse_args()

    run_base      = os.path.abspath(args.outdir) if args.outdir else BASE
    video_dp_dir  = os.path.join(run_base, 'videos_dp')
    video_heat_dir= os.path.join(run_base, 'videos_heat')
    ckpt_dir      = os.path.join(run_base, 'checkpoints')
    score_log     = os.path.join(run_base, 'scores.csv')

    os.makedirs(video_dp_dir,   exist_ok=True)
    os.makedirs(video_heat_dir, exist_ok=True)
    os.makedirs(ckpt_dir,       exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}  |  Heat steps: {args.heat_steps}  |  alpha_max: {args.alpha_max}  |  eval_seed: {EVAL_SEED}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if not os.path.isfile(ZARR_PATH):
        print("Downloading dataset...")
        gdown.download(id="1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t",
                       output=ZARR_PATH, quiet=False)

    print("Loading dataset...")
    dataset = PushTDataset(ZARR_PATH)
    stats   = dataset.stats
    print(f"  {len(dataset)} training windows")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=False)

    total_steps = len(loader) * args.epochs

    # ── Models ────────────────────────────────────────────────────────────────
    global_cond_dim = OBS_HORIZON * OBS_DIM   # 10

    dp_net   = ConditionalUnet1D(ACTION_DIM, global_cond_dim).to(DEVICE)
    heat_net = ConditionalUnet1D(ACTION_DIM, global_cond_dim).to(DEVICE)
    n_dp   = sum(p.numel() for p in dp_net.parameters())
    n_heat = sum(p.numel() for p in heat_net.parameters())
    print(f"  DP params: {n_dp/1e6:.1f}M   Heat params: {n_heat/1e6:.1f}M")

    # ── Optimisers / schedulers / EMA ─────────────────────────────────────────
    ddpm_sched = DDPMScheduler(num_train_timesteps=NUM_DDPM_STEPS,
                               beta_schedule='squaredcos_cap_v2',
                               clip_sample=True, prediction_type='epsilon')
    heat = HeatDissipation1D(num_timesteps=args.heat_steps, alpha_max=args.alpha_max)

    dp_opt   = torch.optim.AdamW(dp_net.parameters(),   lr=args.lr, weight_decay=args.wd)
    heat_opt = torch.optim.AdamW(heat_net.parameters(), lr=args.lr, weight_decay=args.wd)

    dp_lr   = get_scheduler('cosine', optimizer=dp_opt,
                            num_warmup_steps=args.warmup,
                            num_training_steps=total_steps)
    heat_lr = get_scheduler('cosine', optimizer=heat_opt,
                            num_warmup_steps=args.warmup,
                            num_training_steps=total_steps)

    dp_ema   = EMAModel(parameters=dp_net.parameters(),   power=0.75)
    heat_ema = EMAModel(parameters=heat_net.parameters(), power=0.75)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    dp_scores, heat_scores = [], []
    if args.resume:
        print(f"Resuming from {args.resume} …")
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        dp_net.load_state_dict(ckpt['dp_net'])
        heat_net.load_state_dict(ckpt['heat_net'])
        dp_ema.load_state_dict(ckpt['dp_ema'])
        heat_ema.load_state_dict(ckpt['heat_ema'])
        dp_opt.load_state_dict(ckpt['dp_opt'])
        heat_opt.load_state_dict(ckpt['heat_opt'])
        start_epoch = ckpt['epoch'] + 1
        # restore score history from CSV if it exists
        if os.path.isfile(score_log):
            import csv
            with open(score_log) as f:
                for row in csv.DictReader(f):
                    dp_scores.append(float(row['score_dp']))
                    heat_scores.append(float(row['score_heat']))
        print(f"  Resumed at epoch {start_epoch}")
    else:
        # fresh run: initialise score log
        with open(score_log, 'w') as f:
            f.write('epoch,loss_dp,loss_heat,score_dp,score_heat\n')

    # ── Epoch loop ────────────────────────────────────────────────────────────
    print("\nTraining...")
    with tqdm(range(start_epoch, args.epochs + 1), desc='Epoch') as tglobal:
        for epoch in tglobal:
            dp_net.train(); heat_net.train()
            dp_losses, heat_losses = [], []

            for nobs, naction in loader:
                nobs    = nobs.to(DEVICE)       # (B, obs_horizon, 5)
                naction = naction.to(DEVICE)    # (B, pred_horizon, 2)
                B       = nobs.shape[0]
                obs_cond = nobs.flatten(start_dim=1)   # (B, 10)

                # ── Diffusion Policy step (predict noise) ──────────────────
                noise  = torch.randn_like(naction)
                t_dp   = torch.randint(0, NUM_DDPM_STEPS, (B,), device=DEVICE).long()
                noisy  = ddpm_sched.add_noise(naction, noise, t_dp)
                dp_opt.zero_grad()
                loss_dp = F.mse_loss(dp_net(noisy, t_dp, global_cond=obs_cond), noise)
                loss_dp.backward()
                dp_opt.step(); dp_lr.step(); dp_ema.step(dp_net.parameters())
                dp_losses.append(loss_dp.item())

                # ── Heat Dissipation step ──────────────────────────────────
                t_heat      = torch.randint(0, args.heat_steps, (B,), device=DEVICE).long()
                blurred_t   = heat.blur_batch(naction, t_heat)         # b_t = blur(a0, t)
                dc_shift    = (torch.rand(B, 1, ACTION_DIM, device=DEVICE) - 0.5) * 2 * args.dc_shift
                blurred_aug = (blurred_t + dc_shift).clamp(0, 1)       # input to network

                if args.prediction_type == 'residual':
                    # target = b_{t-1} - b_t  (incremental structure recovered per step)
                    # at t=0, "previous" is the clean action a_0 itself
                    t_prev      = (t_heat - 1).clamp(min=0)
                    blurred_tm1 = heat.blur_batch(naction, t_prev)
                    blurred_tm1[t_heat == 0] = naction[t_heat == 0]    # t=0: prev = a_0
                    target = blurred_tm1 - blurred_aug
                else:
                    target = naction

                heat_opt.zero_grad()
                loss_heat = F.mse_loss(heat_net(blurred_aug, t_heat, global_cond=obs_cond), target)
                loss_heat.backward()
                heat_opt.step(); heat_lr.step(); heat_ema.step(heat_net.parameters())
                heat_losses.append(loss_heat.item())

            mean_dp   = np.mean(dp_losses)
            mean_heat = np.mean(heat_losses)
            tglobal.set_postfix(dp=f'{mean_dp:.4f}', heat=f'{mean_heat:.4f}')

            # ── Evaluate both with EMA weights (single seed) ─────────────
            dp_eval   = _ema_copy(dp_net,   dp_ema)
            heat_eval = _ema_copy(heat_net, heat_ema)

            dp_imgs,   dp_score   = eval_dp(dp_eval,   stats, seed=EVAL_SEED)
            heat_imgs, heat_score = eval_heat(heat_eval, heat, stats, seed=EVAL_SEED,
                                              prediction_type=args.prediction_type)
            del dp_eval, heat_eval

            # save video only every video_every epochs to conserve disk
            if args.video_every > 0 and epoch % args.video_every == 0:
                vwrite(os.path.join(video_dp_dir,
                       f'epoch_{epoch:03d}_score_{dp_score:.3f}.mp4'), dp_imgs)
                vwrite(os.path.join(video_heat_dir,
                       f'epoch_{epoch:03d}_score_{heat_score:.3f}.mp4'), heat_imgs)

            dp_scores.append(dp_score)
            heat_scores.append(heat_score)

            # ── Checkpoint every 10 epochs (keep only latest) ────────────
            if epoch % 10 == 0:
                new_ckpt = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt')
                torch.save({
                    'epoch': epoch,
                    'dp_net': dp_net.state_dict(),
                    'heat_net': heat_net.state_dict(),
                    'dp_ema': dp_ema.state_dict(),
                    'heat_ema': heat_ema.state_dict(),
                    'dp_opt': dp_opt.state_dict(),
                    'heat_opt': heat_opt.state_dict(),
                    'stats': stats,
                }, new_ckpt)
                # delete previous checkpoint to save disk space
                prev_ckpt = os.path.join(ckpt_dir, f'epoch_{epoch-10:03d}.pt')
                if os.path.isfile(prev_ckpt):
                    os.remove(prev_ckpt)

            # ── Log scores to CSV ─────────────────────────────────────────
            with open(score_log, 'a') as f:
                f.write(f'{epoch},{mean_dp:.6f},{mean_heat:.6f},{dp_score:.4f},{heat_score:.4f}\n')

            tqdm.write(f'  [epoch {epoch:3d}]  '
                       f'loss dp={mean_dp:.4f}  heat={mean_heat:.4f}  |  '
                       f'score dp={dp_score:.3f}  heat={heat_score:.3f}')

    # ── Score curve ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs  = np.arange(1, len(dp_scores) + 1)
    ax.plot(epochs, dp_scores,   label='Diffusion Policy', color='steelblue',  linewidth=2)
    ax.plot(epochs, heat_scores, label='Heat Dissipation', color='darkorange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Max Coverage Score')
    ax.set_title('Diffusion Policy vs Heat Dissipation Policy — PushT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(run_base, 'comparison_scores.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nScore curve saved → {out}")

    best_dp_ep   = int(np.argmax(dp_scores))   + 1
    best_heat_ep = int(np.argmax(heat_scores)) + 1
    print(f"\nFinal scores (seed={EVAL_SEED}):")
    print(f"  Diffusion Policy : ep{args.epochs}={dp_scores[-1]:.3f}  best={max(dp_scores):.3f} @ ep{best_dp_ep}")
    print(f"  Heat Dissipation : ep{args.epochs}={heat_scores[-1]:.3f}  best={max(heat_scores):.3f} @ ep{best_heat_ep}")
    print(f"\nScore log → {score_log}")


if __name__ == '__main__':
    main()
