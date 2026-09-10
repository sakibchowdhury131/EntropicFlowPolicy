"""
Entropic Flow Policy — Hard Experiment v6
==========================================
Three major changes from v5:

1. FLOW MATCHING replaces DDPM score network
   Old: ScoreNet predicts added noise ε via DDPM objective (10 discrete steps,
        nn.Embedding time conditioning). Loss plateaued at 0.40 across all runs.
   New: VelocityNet predicts the straight-line velocity v = a_clean - noise via
        Conditional Flow Matching (CFM). Training: sample t ~ U[0,1], form
        a_t = (1-t)*noise + t*a_clean, minimise MSE(v_θ(a_t,s,g,t), a_clean-noise).
        Inference: Euler-integrate the velocity ODE from t=0→1 in 10 steps.
        Time conditioning uses sinusoidal embeddings (continuous, not discrete).
        Why better: straight flow paths → 10 steps is genuinely sufficient;
        sinusoidal embedding interpolates across time; no aggressive β schedule.

2. DIFFUSION POLICY added as proper baseline
   EFP and DP share the SAME VelocityNet weights trained on the same data.
   At inference, DP runs the flow ODE without entropy guidance; EFP adds the
   normalised entropy gradient correction at each step. This isolates exactly
   what entropy guidance contributes over pure flow matching.

3. SINGLE CENTER BLOCK + FIXED START ZONE (y ∈ [-0.2, 0.2])
   Obstacle is now a single solid block at x ∈ [-0.4, 0.4], y ∈ [-1.3, 1.3].
   Both modes must arc symmetrically above (apex y > 1.9) or below (apex y < -1.9).
   BC's mean action averages equal-and-opposite y-velocities → points straight
   into the center block regardless of small y-offset at start.
   Previous two-block design with a gap let BC exploit tiny y-offsets to route.
   Guidance scale fixed: correction now multiplied by dt so total perturbation
   over 10 steps is 0.3 units (was 3.0 — dominated the flow ODE).

Baseline lineup: EFP (Ours) | Diffusion Policy | BC (Mean) | MDN-BC
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H_FAILURE = np.log(200.0)   # ~5.3, well above max success label log(49)~3.9


# ============================================================
# ENVIRONMENT
# ============================================================

class ObstacleNavEnv:
    """
    2D point navigation. Single solid block at x ∈ [-0.4, 0.4], y ∈ [-1.3, 1.3].
    Two viable paths: arc above (apex y > 1.9) or arc below (apex y < -1.9).
    BC's mean action averages symmetric upper/lower y-velocities → hits block.
    Goal: (2.0, 0.0).  Start: x ∈ [-2.5, -1.5], y ∈ [-0.2, 0.2].
    """
    def __init__(self):
        self.dt = 0.1
        self.max_speed = 1.5
        self.goal = np.array([2.0, 0.0])
        self.obstacles = [
            (-0.4, 0.4, -1.3, 1.3),  # single center block
        ]

    def in_obstacle(self, pos):
        x, y = pos
        for xmin, xmax, ymin, ymax in self.obstacles:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False

    def step(self, pos, action):
        action = np.clip(action, -self.max_speed, self.max_speed)
        new_pos = np.clip(pos + action * self.dt, -3, 3)
        if self.in_obstacle(new_pos):
            tx = np.array([new_pos[0], pos[1]])
            ty = np.array([pos[0], new_pos[1]])
            if not self.in_obstacle(tx): return tx
            if not self.in_obstacle(ty): return ty
            return pos.copy()
        return new_pos

    def reached_goal(self, pos, threshold=0.2):
        return np.linalg.norm(pos - self.goal) < threshold


# ============================================================
# DEMONSTRATIONS
# ============================================================

def generate_demos(env, n_demos=300, n_points=48):
    """
    Upper/lower paths only (no middle — gap is closed).
    Start y restricted to [-0.2, 0.2] so both modes depart from the same
    region, creating genuine multi-modal conflict for BC.
    """
    demos = []
    modes = ['upper'] * (n_demos // 2) + ['lower'] * (n_demos - n_demos // 2)
    np.random.shuffle(modes)
    for idx in range(n_demos):
        start = np.array([
            np.random.uniform(-2.5, -1.5),
            0.0,   # fixed y=0: exact symmetry forces BC mean action to y=0 → hits block
        ])
        goal = env.goal.copy()
        mode = modes[idx]

        if mode == 'upper':
            apex_y = np.random.uniform(1.9, 2.5)
            waypoints = [
                start.copy(),
                np.array([-0.8,  apex_y * 0.55]),
                np.array([-0.45, apex_y]),
                np.array([ 0.45, apex_y]),
                np.array([ 1.1,  apex_y * 0.4]),
                np.array([ 1.6,  apex_y * 0.05]),
                goal.copy(),
            ]
        else:
            apex_y = np.random.uniform(-2.5, -1.9)
            waypoints = [
                start.copy(),
                np.array([-0.8,  apex_y * 0.55]),
                np.array([-0.45, apex_y]),
                np.array([ 0.45, apex_y]),
                np.array([ 1.1,  apex_y * 0.4]),
                np.array([ 1.6,  apex_y * 0.05]),
                goal.copy(),
            ]

        path = []
        segs = len(waypoints) - 1
        pts_per = n_points // segs
        for seg in range(segs):
            p1, p2 = waypoints[seg], waypoints[seg+1]
            for t in range(pts_per):
                alpha = 0.5*(1 - np.cos(np.pi*t/pts_per))
                path.append(p1*(1-alpha) + p2*alpha)
        path.append(goal.copy())

        path = np.array(path)
        noise = np.random.randn(*path.shape) * 0.04
        noise[0] = 0; noise[-1] = 0
        path += noise

        if any(env.in_obstacle(p) for p in path):
            continue

        traj = []
        for t in range(len(path)-1):
            traj.append({
                'state':      path[t].copy(),
                'action':     (path[t+1] - path[t]) / env.dt,
                'next_state': path[t+1].copy(),
                'goal':       goal.copy(),
                'timestep':   t,
                'total_steps': len(path)-1,
                'mode':       mode,
            })
        demos.append(traj)

    modes = [d[0]['mode'] for d in demos]
    print(f"Generated {len(demos)} demos  "
          f"(upper={modes.count('upper')}, lower={modes.count('lower')})")
    return demos


def generate_failure_demos(env, n_per_obstacle=60, boundary_steps=6):
    """
    Left-approach-only, boundary-labeled failure trajectories (from v4).
    Only the last boundary_steps states before collision are kept and labeled
    H_FAILURE. The goal-approach corridor (right of obstacles) stays clean.
    """
    demos = []
    for xmin, xmax, ymin, ymax in env.obstacles:
        generated = 0
        attempts = 0
        while generated < n_per_obstacle and attempts < n_per_obstacle * 15:
            attempts += 1
            start = np.array([
                np.random.uniform(-2.5, xmin - 0.1),
                np.random.uniform(ymin - 0.5, ymax + 0.5),
            ])
            if env.in_obstacle(start):
                continue
            target = np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax),
            ])
            pos = start.copy()
            all_steps, hit = [], False
            for _ in range(50):
                d = target - pos
                dist = np.linalg.norm(d)
                if dist < 1e-3: break
                action = (d / dist) * env.max_speed
                next_pos = np.clip(pos + action*env.dt, -3, 3)
                all_steps.append({'state': pos.copy(), 'action': action.copy(),
                                   'next_state': next_pos.copy(), 'goal': env.goal.copy()})
                pos = next_pos.copy()
                if env.in_obstacle(pos):
                    for _ in range(2):
                        all_steps.append({'state': pos.copy(), 'action': np.zeros(2),
                                          'next_state': pos.copy(), 'goal': env.goal.copy()})
                    hit = True; break
            if not hit or len(all_steps) < 2: continue
            boundary = all_steps[-boundary_steps:]
            for s in boundary: s['entropy_label'] = H_FAILURE
            demos.append(boundary)
            generated += 1

    total = sum(len(d) for d in demos)
    print(f"Generated {len(demos)} failure trajectories ({total} boundary states)")
    return demos


# ============================================================
# DATASETS
# ============================================================

class NavDataset(Dataset):
    def __init__(self, demonstrations):
        S, A, NS, G, EL = [], [], [], [], []
        for traj in demonstrations:
            for step in traj:
                S.append(step['state']); A.append(step['action'])
                NS.append(step['next_state']); G.append(step['goal'])
                rem = step['total_steps'] - step['timestep']
                EL.append(np.log(rem + 1))
        self.states      = torch.FloatTensor(np.array(S))
        self.actions     = torch.FloatTensor(np.array(A))
        self.next_states = torch.FloatTensor(np.array(NS))
        self.goals       = torch.FloatTensor(np.array(G))
        self.entropy_labels = torch.FloatTensor(np.array(EL))
    def __len__(self): return len(self.states)
    def __getitem__(self, idx):
        return {'state': self.states[idx], 'action': self.actions[idx],
                'next_state': self.next_states[idx], 'goal': self.goals[idx],
                'entropy_label': self.entropy_labels[idx]}

class PairDataset(Dataset):
    def __init__(self, demos, pairs_per_traj=20):
        se, sl, gs = [], [], []
        for traj in demos:
            n = len(traj)
            for _ in range(pairs_per_traj):
                i = np.random.randint(0, n-2); j = np.random.randint(i+1, n)
                se.append(traj[i]['state']); sl.append(traj[j]['state'])
                gs.append(traj[0]['goal'])
        self.se = torch.FloatTensor(np.array(se))
        self.sl = torch.FloatTensor(np.array(sl))
        self.gs = torch.FloatTensor(np.array(gs))
    def __len__(self): return len(self.se)
    def __getitem__(self, idx): return self.se[idx], self.sl[idx], self.gs[idx]

class FailureDataset(Dataset):
    def __init__(self, failure_demos):
        S, G, EL = [], [], []
        for traj in failure_demos:
            for step in traj:
                S.append(step['state']); G.append(step['goal'])
                EL.append(step['entropy_label'])
        self.states = torch.FloatTensor(np.array(S))
        self.goals  = torch.FloatTensor(np.array(G))
        self.entropy_labels = torch.FloatTensor(np.array(EL))
    def __len__(self): return len(self.states)
    def __getitem__(self, idx):
        return {'state': self.states[idx], 'goal': self.goals[idx],
                'entropy_label': self.entropy_labels[idx]}


# ============================================================
# NETWORKS
# ============================================================

class SinusoidalEmbedding(nn.Module):
    """Continuous time embedding for flow matching. Better than nn.Embedding
    because it interpolates smoothly between time values."""
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
    def forward(self, t):
        # t: (batch,) float in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        x = t[:, None] * freqs[None]
        return torch.cat([x.sin(), x.cos()], dim=-1)


class VelocityNet(nn.Module):
    """
    Conditional flow matching network. Predicts the velocity field
    v(a_t, s, g, t) = a_clean - noise along the straight interpolation path.

    Replaces ScoreNet (DDPM noise predictor). Key differences:
    - Continuous time t ∈ [0,1] with sinusoidal embedding (vs discrete k with nn.Embedding)
    - Predicts velocity (direction + magnitude toward clean action) vs noise
    - 10 Euler steps at inference is genuinely sufficient for straight-path ODE
    """
    def __init__(self, h=128, t_dim=16):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(t_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + 2 + 2 + t_dim, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, 2))
    def forward(self, a, s, g, t):
        return self.net(torch.cat([a, s, g, self.time_emb(t)], dim=-1))


class EntropyNet(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h//2), nn.GELU(),
            nn.Linear(h//2, 1), nn.Softplus())
    def forward(self, s, g):
        return self.net(torch.cat([s, g], -1)).squeeze(-1)

class DynNet(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2))
    def forward(self, s, a):
        return s + self.net(torch.cat([s, a], -1))

class BCNet(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2))
    def forward(self, s, g):
        return self.net(torch.cat([s, g], -1))

class MDN(nn.Module):
    def __init__(self, h=128, K=3):
        super().__init__()
        self.K = K
        self.backbone = nn.Sequential(
            nn.Linear(4, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU())
        self.pi_head  = nn.Linear(h, K)
        self.mu_head  = nn.Linear(h, K*2)
        self.sig_head = nn.Linear(h, K*2)
    def forward(self, s, g):
        f  = self.backbone(torch.cat([s, g], -1))
        pi = torch.softmax(self.pi_head(f), -1)
        mu = self.mu_head(f).reshape(-1, self.K, 2)
        sig = torch.exp(self.sig_head(f).reshape(-1, self.K, 2).clamp(-5, 2))
        return pi, mu, sig
    def sample(self, s, g):
        pi, mu, sig = self.forward(s, g)
        c = torch.multinomial(pi, 1).squeeze(-1)
        idx = torch.arange(len(c))
        return mu[idx, c] + sig[idx, c] * torch.randn_like(sig[idx, c])


# ============================================================
# TRAINING
# ============================================================

def train_flow(net, ds, ep=150):
    """
    Conditional Flow Matching objective.
    Sample t ~ U[0,1], form a_t = (1-t)*noise + t*a_clean, train to predict
    the constant velocity v = a_clean - noise along the straight-line path.
    """
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ld  = DataLoader(ds, batch_size=256, shuffle=True)
    losses = []
    for e in range(ep):
        tot, n = 0, 0
        for b in ld:
            a_clean = b['action']
            bsz     = a_clean.shape[0]
            t       = torch.rand(bsz)
            noise   = torch.randn_like(a_clean)
            a_t     = (1 - t[:, None]) * noise + t[:, None] * a_clean
            target_v = a_clean - noise
            loss = nn.functional.mse_loss(net(a_t, b['state'], b['goal'], t), target_v)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        losses.append(tot / max(n, 1))
        if (e+1) % 30 == 0: print(f"  Flow {e+1}/{ep}: {losses[-1]:.4f}")
    return losses

def train_entropy(net, pairs, ds, failure_ds=None, ep=60):
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pl  = DataLoader(pairs, batch_size=256, shuffle=True)
    rl  = DataLoader(ds, batch_size=256, shuffle=True)
    fl  = DataLoader(failure_ds, batch_size=128, shuffle=True) if failure_ds else None
    losses = []
    for e in range(ep):
        tot, n = 0, 0
        for se, sl, g in pl:
            he, hl = net(se, g), net(sl, g)
            loss = torch.relu(hl - he + 0.3).mean() + 0.5*(net(g, g)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        for b in rl:
            reg = nn.functional.mse_loss(net(b['state'], b['goal']), b['entropy_label'])
            opt.zero_grad(); (0.3*reg).backward(); opt.step()
            tot += reg.item(); n += 1
        if fl is not None:
            for bf in fl:
                loss = nn.functional.mse_loss(net(bf['state'], bf['goal']), bf['entropy_label'])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); n += 1
        losses.append(tot / max(n, 1))
        if (e+1) % 20 == 0: print(f"  Entropy {e+1}/{ep}: {losses[-1]:.4f}")
    return losses

def train_dyn(net, ds, ep=40):
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ld  = DataLoader(ds, batch_size=256, shuffle=True)
    losses = []
    for e in range(ep):
        tot, n = 0, 0
        for b in ld:
            loss = nn.functional.mse_loss(net(b['state'], b['action']), b['next_state'])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        losses.append(tot / max(n, 1))
        if (e+1) % 20 == 0: print(f"  Dynamics {e+1}/{ep}: {losses[-1]:.4f}")
    return losses

def train_bc(net, ds, ep=60):
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ld  = DataLoader(ds, batch_size=256, shuffle=True)
    losses = []
    for e in range(ep):
        tot, n = 0, 0
        for b in ld:
            loss = nn.functional.mse_loss(net(b['state'], b['goal']), b['action'])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        losses.append(tot / max(n, 1))
        if (e+1) % 20 == 0: print(f"  BC {e+1}/{ep}: {losses[-1]:.4f}")
    return losses

def train_mdn(net, ds, ep=60):
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ld  = DataLoader(ds, batch_size=256, shuffle=True)
    losses = []
    for e in range(ep):
        tot, n = 0, 0
        for b in ld:
            pi, mu, sig = net(b['state'], b['goal'])
            a = b['action'].unsqueeze(1)
            lp = -0.5*((a-mu)/sig)**2 - torch.log(sig) - 0.5*np.log(2*np.pi)
            lp = lp.sum(-1) + torch.log(pi + 1e-8)
            loss = -torch.logsumexp(lp, -1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        losses.append(tot / max(n, 1))
        if (e+1) % 20 == 0: print(f"  MDN {e+1}/{ep}: {losses[-1]:.4f}")
    return losses


# ============================================================
# INFERENCE
# ============================================================

def flow_sample(vel_net, entropy_net, dyn_net, state, goal, steps=10, guidance=0.3):
    """
    EFP inference via flow matching ODE + entropy gradient guidance.

    Euler-integrates the velocity field from t=0 (noise) to t=1 (action).
    At each step, applies normalised entropy gradient as a correction term:
      a ← a - guidance * (∇_a H_θ(f(s,a), g) / ||∇||) * escale
    Normalising the gradient separates steering direction from magnitude,
    preventing steep failure walls from over-correcting (fix from v4).
    """
    s = torch.FloatTensor(state).unsqueeze(0)
    g = torch.FloatTensor(goal).unsqueeze(0)
    with torch.no_grad():
        h = entropy_net(s, g).item()
    escale = min(h / 3.0, 1.0)

    a  = torch.randn(1, 2)
    dt = 1.0 / steps
    with torch.no_grad():
        for i in range(steps):
            t  = torch.FloatTensor([i * dt])
            v  = vel_net(a, s, g, t)
            a  = a + dt * v

            if guidance > 0:
                sf  = dyn_net(s, a)
                hc  = entropy_net(sf, g)
                grad = torch.zeros_like(a)
                eps  = 0.02
                for d in range(2):
                    ap = a.clone(); ap[0, d] += eps
                    grad[0, d] = (entropy_net(dyn_net(s, ap), g) - hc) / eps
                grad = grad / (torch.norm(grad) + 1e-8)
                a = a - guidance * grad * escale * dt

    return a.squeeze(0).numpy()


def dp_sample(vel_net, state, goal, steps=10):
    """
    Diffusion Policy inference — same VelocityNet as EFP, no entropy guidance.
    Isolates the contribution of entropy guidance over plain flow matching.
    """
    s = torch.FloatTensor(state).unsqueeze(0)
    g = torch.FloatTensor(goal).unsqueeze(0)
    a  = torch.randn(1, 2)
    dt = 1.0 / steps
    with torch.no_grad():
        for i in range(steps):
            t = torch.FloatTensor([i * dt])
            a = a + dt * vel_net(a, s, g, t)
    return a.squeeze(0).numpy()


# ============================================================
# EVALUATION
# ============================================================

def rollout(env, pol, start, max_t=100, perturb_t=None):
    pos  = start.copy()
    traj = [pos.copy()]
    for t in range(max_t):
        if env.reached_goal(pos): break
        if perturb_t and t == perturb_t:
            pos = np.clip(pos + np.random.randn(2)*0.4, -3, 3)
        a = pol(pos, env.goal)
        pos = env.step(pos, np.clip(a, -env.max_speed, env.max_speed))
        traj.append(pos.copy())
    return {'trajectory': np.array(traj),
            'final_dist': np.linalg.norm(pos - env.goal),
            'success':    env.reached_goal(pos),
            'steps':      len(traj)}


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("ENTROPIC FLOW POLICY — Multi-Modal Navigation v6")
    print("  Flow Matching + DP baseline + Fixed start zone")
    print("="*60)

    np.random.seed(42); torch.manual_seed(42)
    env = ObstacleNavEnv()

    print("\n[1] Generating success demonstrations...")
    demos = generate_demos(env, n_demos=300, n_points=48)

    print("\n[2] Generating failure demonstrations...")
    failure_demos = generate_failure_demos(env, n_per_obstacle=60, boundary_steps=6)

    print("\n[3] Datasets...")
    ds         = NavDataset(demos)
    pairs      = PairDataset(demos, 20)
    failure_ds = FailureDataset(failure_demos)
    print(f"  Success: {len(ds)} transitions, {len(pairs)} pairs")
    print(f"  Failure: {len(failure_ds)} transitions  (H_FAILURE={H_FAILURE:.2f})")

    H = 128
    enet  = EntropyNet(H)
    dnet  = DynNet(H)
    vnet  = VelocityNet(H)          # shared by EFP and Diffusion Policy
    bcnet = BCNet(H)
    mdn   = MDN(H, 3)

    print("\n[4] Training entropy estimator...")
    el = train_entropy(enet, pairs, ds, failure_ds, ep=60)
    print("\n[5] Training dynamics...")
    dl = train_dyn(dnet, ds, ep=40)
    print("\n[6] Training flow matching (VelocityNet — shared by EFP + DP)...")
    fl = train_flow(vnet, ds, ep=150)
    print("\n[7] Training BC...")
    bl = train_bc(bcnet, ds, ep=60)
    print("\n[8] Training MDN-BC...")
    ml = train_mdn(mdn, ds, ep=60)

    enet.eval(); dnet.eval(); vnet.eval(); bcnet.eval(); mdn.eval()

    def efp_pol(p, g):
        return flow_sample(vnet, enet, dnet, p, g, steps=10, guidance=0.3)
    def dp_pol(p, g):
        return dp_sample(vnet, p, g, steps=10)
    def bc_pol(p, g):
        with torch.no_grad():
            return bcnet(torch.FloatTensor(p).unsqueeze(0),
                         torch.FloatTensor(g).unsqueeze(0)).squeeze(0).numpy()
    def mdn_pol(p, g):
        with torch.no_grad():
            return mdn.sample(torch.FloatTensor(p).unsqueeze(0),
                              torch.FloatTensor(g).unsqueeze(0)).squeeze(0).numpy()

    pols = {
        'EFP (Ours)':        efp_pol,
        'Diffusion Policy':  dp_pol,
        'BC (Mean)':         bc_pol,
        'MDN-BC':            mdn_pol,
    }
    mc = {
        'EFP (Ours)':       '#1565C0',
        'Diffusion Policy': '#7B1FA2',
        'BC (Mean)':        '#E53935',
        'MDN-BC':           '#43A047',
    }

    print("\n[9] Evaluation...")
    np.random.seed(777)

    # In-dist: y fixed at 0.0 matching training — BC sees identical state for both modes
    id_starts = [np.array([np.random.uniform(-2.5, -1.5), 0.0]) for _ in range(25)]
    # OOD: close to wall, slightly wider y
    ood_starts = [np.array([np.random.uniform(-0.7, -0.35),
                             np.random.uniform(-0.3,  0.3)]) for _ in range(25)]

    results = {}
    for cond, starts, pert in [('In-Dist', id_starts, None),
                                ('OOD',     ood_starts, None),
                                ('Perturb', id_starts[:15], 20)]:
        print(f"\n  === {cond} ===")
        results[cond] = {}
        for name, pol in pols.items():
            runs = [rollout(env, pol, s, 100, pert) for s in starts]
            sr = np.mean([r['success'] for r in runs])
            ad = np.mean([r['final_dist'] for r in runs])
            print(f"    {name:<20} success={sr:.0%}  dist={ad:.3f}")
            results[cond][name] = runs

    # ============ VISUALIZATION ============
    print("\n[10] Plotting...")
    conds = ['In-Dist', 'OOD', 'Perturb']
    names = list(pols.keys())

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(
        'EFP v6 — Flow Matching + DP Baseline + Fixed Start Zone',
        fontsize=14, fontweight='bold', y=0.99)

    def draw_obs(ax):
        for xmin, xmax, ymin, ymax in env.obstacles:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                        color='#424242', alpha=0.85, zorder=3))

    # Row 1: Trajectories
    for ci, cond in enumerate(conds):
        ax = fig.add_subplot(3, 4, ci+1)
        draw_obs(ax)
        for name in names:
            for r in results[cond][name][:8]:
                t = r['trajectory']
                ax.plot(t[:,0], t[:,1], color=mc[name], alpha=0.35, lw=0.8)
        ax.plot(*env.goal, 'r*', ms=14, zorder=5)
        ax.set_xlim(-3.1, 3.1); ax.set_ylim(-3.1, 3.1); ax.set_aspect('equal')
        ax.set_title(f'{cond} Trajectories', fontsize=11); ax.grid(True, alpha=0.2)
        if ci == 0:
            for n, c in mc.items(): ax.plot([], [], color=c, lw=2, label=n)
            ax.legend(fontsize=6, loc='lower left')

    # Expert + failure demos
    ax = fig.add_subplot(3, 4, 4)
    draw_obs(ax)
    mode_colors = {'upper': '#1565C0', 'lower': '#E53935'}
    for d in demos[:60]:
        t = np.array([s['state'] for s in d])
        ax.plot(t[:,0], t[:,1], color=mode_colors[d[0]['mode']], alpha=0.3, lw=0.7)
    for d in failure_demos[:40]:
        t = np.array([s['state'] for s in d])
        ax.plot(t[:,0], t[:,1], color='#FF6F00', alpha=0.5, lw=0.7)
    ax.plot(*env.goal, 'r*', ms=14, zorder=5)
    ax.set_xlim(-3.1, 3.1); ax.set_ylim(-3.1, 3.1); ax.set_aspect('equal')
    ax.set_title('Expert + Failure Demos', fontsize=11)
    for m, c in mode_colors.items(): ax.plot([], [], color=c, label=m.capitalize())
    ax.plot([], [], color='#FF6F00', label='Failure')
    ax.legend(fontsize=7, loc='lower left'); ax.grid(True, alpha=0.2)

    # Row 2: Success bars
    for ci, cond in enumerate(conds):
        ax = fig.add_subplot(3, 4, 5+ci)
        rates = [np.mean([r['success'] for r in results[cond][n]]) for n in names]
        bars = ax.bar(range(len(names)), rates, color=[mc[n] for n in names],
                      width=0.6, edgecolor='black', lw=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
        ax.set_ylabel('Success Rate'); ax.set_title(f'{cond} Success', fontsize=11)
        ax.set_ylim(0, 1.15)
        for b, r in zip(bars, rates):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                    f'{r:.0%}', ha='center', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    # Avg distance
    ax = fig.add_subplot(3, 4, 8)
    x = np.arange(len(conds)); w = 0.2
    for i, name in enumerate(names):
        means = [np.mean([r['final_dist'] for r in results[c][name]]) for c in conds]
        ax.bar(x + (i - 1.5)*w, means, w, label=name, color=mc[name],
               edgecolor='black', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel('Avg Final Distance'); ax.set_title('Final Distance', fontsize=11)
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3, axis='y')

    # Row 3: Entropy landscape
    ax = fig.add_subplot(3, 4, 9)
    xs = np.linspace(-3, 3, 80); ys = np.linspace(-3, 3, 80)
    Hmap = np.zeros((80, 80))
    gt = torch.FloatTensor(env.goal).unsqueeze(0)
    with torch.no_grad():
        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                Hmap[j, i] = enet(torch.FloatTensor([[xv, yv]]), gt).item()
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            if env.in_obstacle(np.array([xv, yv])): Hmap[j, i] = np.nan
    im = ax.contourf(xs, ys, np.ma.masked_invalid(Hmap), levels=30, cmap='viridis_r')
    draw_obs(ax)
    ax.plot(*env.goal, 'r*', ms=14, zorder=5)
    ax.set_xlim(-3.1, 3.1); ax.set_ylim(-3.1, 3.1); ax.set_aspect('equal')
    ax.set_title('Learned Entropy Landscape', fontsize=11)
    plt.colorbar(im, ax=ax, label='H(s,g)', shrink=0.8)

    # Training losses
    ax = fig.add_subplot(3, 4, 10)
    ax.plot(el, label='Entropy',  color='#1565C0', lw=1.5)
    ax.plot(fl, label='Flow (v6)', color='#7B1FA2', lw=1.5)
    ax.plot(bl, label='BC',        color='#E53935', lw=1.5)
    ax.plot(ml, label='MDN',       color='#43A047', lw=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Losses (v6)', fontsize=11)
    ax.legend(fontsize=8); ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    # Entropy profiles
    ax = fig.add_subplot(3, 4, 11)
    with torch.no_grad():
        for r in results['In-Dist']['EFP (Ours)'][:15]:
            hs = [enet(torch.FloatTensor(p).unsqueeze(0), gt).item()
                  for p in r['trajectory']]
            ax.plot(hs, color='#1565C0' if r['success'] else '#FF9800',
                    alpha=0.6, lw=1.2)
    ax.set_xlabel('Timestep'); ax.set_ylabel('H(s,g)')
    ax.set_title('Entropy Along EFP Rollouts', fontsize=11)
    ax.plot([], [], color='#1565C0', label='Success')
    ax.plot([], [], color='#FF9800', label='Failure')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Summary
    ax = fig.add_subplot(3, 4, 12); ax.axis('off')
    txt = "RESULTS SUMMARY (v6)\n" + "="*44 + "\n\n"
    for cond in conds:
        txt += f"{cond}:\n"
        for name in names:
            sr = np.mean([r['success'] for r in results[cond][name]])
            ad = np.mean([r['final_dist'] for r in results[cond][name]])
            txt += f"  {name:<20} {sr:4.0%}  d={ad:.3f}\n"
        txt += "\n"
    txt += f"Flow matching: 150 epochs\n"
    txt += f"H_FAILURE = {H_FAILURE:.2f}  |  starts y∈[-0.2,0.2]"
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=8,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = '/home/sakib/working_dir/entropyFlowPolicy/efp_v6_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
