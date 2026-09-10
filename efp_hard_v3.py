"""
Entropic Flow Policy — Hard Experiment v3
==========================================
New in v3: Failure demonstration supervision.

Core idea: the entropy estimator in v2 was trained only on successful
(collision-free) trajectories. It learned entropy as a smooth function
of temporal proximity to the goal, with no signal about obstacle regions.
The obstacle areas were simply out-of-distribution — the model interpolated
through them with no repulsion.

Fix: generate intentional failure demonstrations (trajectories that walk
directly into obstacles) and label every state on those trajectories with
H_FAILURE >> max success entropy. The ordering principle holds: from a
state that leads to collision, the count of feasible trajectories to the
goal is near zero, so true entropy is very high.

This gives the entropy landscape "walls" around obstacle boundaries, and
the entropy gradient during EFP inference naturally repels the policy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# H_FAILURE >> max success entropy (log(48+1) ~ 3.87 for n_points=48)
H_FAILURE = np.log(200.0)   # ~5.3


# ============================================================
# ENVIRONMENT: 2D Navigation with Two Obstacles
# ============================================================

class ObstacleNavEnv:
    """
    2D point navigation with two rectangular obstacles.

    Layout:
        Workspace: [-3, 3] x [-3, 3]
        Obstacle 1: rectangle at x ∈ [-0.3, 0.3], y ∈ [0.4, 2.5]
        Obstacle 2: rectangle at x ∈ [-0.3, 0.3], y ∈ [-2.5, -0.4]
        Gap between them: y ∈ [-0.4, 0.4]

    This creates THREE viable paths:
        - UPPER: go above obstacle 1
        - LOWER: go below obstacle 2
        - MIDDLE: go through the gap between them

    Start region: x ∈ [-2.5, -1.5]
    Goal: (2.0, 0.0)
    """
    def __init__(self):
        self.dt = 0.1
        self.max_speed = 1.5
        self.goal = np.array([2.0, 0.0])

        # Obstacles (x_min, x_max, y_min, y_max)
        self.obstacles = [
            (-0.3, 0.3, 0.4, 2.5),   # upper block
            (-0.3, 0.3, -2.5, -0.4),  # lower block
        ]

    def in_obstacle(self, pos):
        x, y = pos
        for xmin, xmax, ymin, ymax in self.obstacles:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False

    def step(self, pos, action):
        action = np.clip(action, -self.max_speed, self.max_speed)
        new_pos = pos + action * self.dt
        new_pos[0] = np.clip(new_pos[0], -3, 3)
        new_pos[1] = np.clip(new_pos[1], -3, 3)

        # Hard collision: stay put if in obstacle
        if self.in_obstacle(new_pos):
            test_x = np.array([new_pos[0], pos[1]])
            test_y = np.array([pos[0], new_pos[1]])
            if not self.in_obstacle(test_x):
                return test_x
            elif not self.in_obstacle(test_y):
                return test_y
            return pos.copy()
        return new_pos

    def reached_goal(self, pos, threshold=0.2):
        return np.linalg.norm(pos - self.goal) < threshold


# ============================================================
# EXPERT DEMONSTRATIONS via waypoint paths (unchanged from v2)
# ============================================================

def generate_demos(env, n_demos=300, n_points=50):
    demos = []

    for i in range(n_demos):
        start = np.array([
            np.random.uniform(-2.5, -1.5),
            np.random.uniform(-2.0, 2.0),
        ])

        goal = env.goal.copy()

        # Choose mode
        r = np.random.rand()
        if r < 0.35:
            mode = 'upper'
        elif r < 0.70:
            mode = 'lower'
        else:
            mode = 'middle'

        # Generate waypoints that are GUARANTEED collision-free
        if mode == 'upper':
            apex_y = np.random.uniform(2.6, 2.9)
            waypoints = [
                start.copy(),
                np.array([start[0] * 0.5 - 0.3, apex_y * 0.7 + start[1] * 0.3]),
                np.array([-0.3, apex_y]),
                np.array([0.3, apex_y]),
                np.array([0.8, apex_y * 0.5 + goal[1] * 0.5]),
                np.array([1.3, goal[1] + (apex_y - goal[1]) * 0.2]),
                goal.copy(),
            ]
        elif mode == 'lower':
            apex_y = np.random.uniform(-2.9, -2.6)
            waypoints = [
                start.copy(),
                np.array([start[0] * 0.5 - 0.3, apex_y * 0.7 + start[1] * 0.3]),
                np.array([-0.3, apex_y]),
                np.array([0.3, apex_y]),
                np.array([0.8, apex_y * 0.5 + goal[1] * 0.5]),
                np.array([1.3, goal[1] + (apex_y - goal[1]) * 0.2]),
                goal.copy(),
            ]
        else:  # middle - through the gap
            gap_y = np.random.uniform(-0.2, 0.2)
            waypoints = [
                start.copy(),
                np.array([-0.8, gap_y * 0.5 + start[1] * 0.5]),
                np.array([-0.4, gap_y]),
                np.array([0.0, gap_y]),
                np.array([0.4, gap_y]),
                np.array([1.0, gap_y * 0.3 + goal[1] * 0.7]),
                goal.copy(),
            ]

        # Smooth interpolation
        path = []
        segs = len(waypoints) - 1
        pts_per = n_points // segs
        for seg in range(segs):
            p1, p2 = waypoints[seg], waypoints[seg + 1]
            for t in range(pts_per):
                alpha = 0.5 * (1 - np.cos(np.pi * t / pts_per))
                path.append(p1 * (1 - alpha) + p2 * alpha)
        path.append(goal.copy())

        # Add small noise
        path = np.array(path)
        noise = np.random.randn(*path.shape) * 0.04
        noise[0] = 0; noise[-1] = 0
        path += noise

        # Verify collision-free
        has_collision = any(env.in_obstacle(p) for p in path)
        if has_collision:
            continue

        # Build trajectory
        trajectory = []
        for t in range(len(path) - 1):
            action = (path[t+1] - path[t]) / env.dt
            trajectory.append({
                'state': path[t].copy(),
                'action': action.copy(),
                'next_state': path[t+1].copy(),
                'goal': goal.copy(),
                'timestep': t,
                'total_steps': len(path) - 1,
                'mode': mode,
            })
        demos.append(trajectory)

    modes = [d[0]['mode'] for d in demos]
    print(f"Generated {len(demos)} collision-free demonstrations")
    print(f"  Upper: {modes.count('upper')}, Lower: {modes.count('lower')}, Middle: {modes.count('middle')}")
    return demos


def generate_failure_demos(env, n_total=120):
    """
    Generate trajectories that intentionally enter obstacles.

    For each obstacle, we sample random approach angles and walk from
    outside toward the obstacle center, recording all states. Every state
    on these trajectories — pre-collision approaches and interior states —
    gets labeled H_FAILURE.

    Theoretical justification: from a state inside or heading into a wall,
    the count of feasible trajectories to goal within horizon H is near zero.
    The true entropy H(s, s_g) is therefore very high (→ log(0) limit).
    H_FAILURE = log(200) is a finite proxy well above any success label.
    """
    demos = []
    n_per_obstacle = n_total // len(env.obstacles)

    for obs_idx, (xmin, xmax, ymin, ymax) in enumerate(env.obstacles):
        obs_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        generated = 0
        attempts = 0

        while generated < n_per_obstacle and attempts < n_per_obstacle * 20:
            attempts += 1

            # Sample a random approach angle and distance
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.4, 2.0)
            start = obs_center + dist * np.array([np.cos(angle), np.sin(angle)])

            # Clip to workspace and skip if start is inside any obstacle
            start[0] = np.clip(start[0], -2.9, 2.9)
            start[1] = np.clip(start[1], -2.9, 2.9)
            if env.in_obstacle(start):
                continue

            # Walk directly toward obstacle center
            pos = start.copy()
            direction = obs_center - pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            speed = env.max_speed

            trajectory = []
            hit = False
            for t in range(30):
                action = direction * speed
                next_pos = pos + action * env.dt
                next_pos[0] = np.clip(next_pos[0], -3, 3)
                next_pos[1] = np.clip(next_pos[1], -3, 3)

                trajectory.append({
                    'state': pos.copy(),
                    'action': action.copy(),
                    'next_state': next_pos.copy(),
                    'goal': env.goal.copy(),
                    'entropy_label': H_FAILURE,
                })
                pos = next_pos.copy()

                if env.in_obstacle(pos):
                    # Record a few steps inside the obstacle too
                    for _ in range(4):
                        trajectory.append({
                            'state': pos.copy(),
                            'action': np.zeros(2),
                            'next_state': pos.copy(),
                            'goal': env.goal.copy(),
                            'entropy_label': H_FAILURE,
                        })
                    hit = True
                    break

            # Only keep trajectories that actually reached the obstacle
            if hit and len(trajectory) >= 3:
                demos.append(trajectory)
                generated += 1

    total_steps = sum(len(d) for d in demos)
    print(f"Generated {len(demos)} failure trajectories ({total_steps} failure states)")
    return demos


# ============================================================
# DATASETS
# ============================================================

class NavDataset(Dataset):
    def __init__(self, demonstrations):
        states, actions, next_states, goals, entropy_labels = [], [], [], [], []
        for traj in demonstrations:
            for step in traj:
                states.append(step['state'])
                actions.append(step['action'])
                next_states.append(step['next_state'])
                goals.append(step['goal'])
                remaining = step['total_steps'] - step['timestep']
                entropy_labels.append(np.log(remaining + 1))
        self.states = torch.FloatTensor(np.array(states))
        self.actions = torch.FloatTensor(np.array(actions))
        self.next_states = torch.FloatTensor(np.array(next_states))
        self.goals = torch.FloatTensor(np.array(goals))
        self.entropy_labels = torch.FloatTensor(np.array(entropy_labels))

    def __len__(self): return len(self.states)
    def __getitem__(self, idx):
        return {'state': self.states[idx], 'action': self.actions[idx],
                'next_state': self.next_states[idx], 'goal': self.goals[idx],
                'entropy_label': self.entropy_labels[idx]}


class PairDataset(Dataset):
    def __init__(self, demos, pairs_per_traj=20):
        s_e, s_l, gs = [], [], []
        for traj in demos:
            n = len(traj)
            for _ in range(pairs_per_traj):
                i = np.random.randint(0, n-2)
                j = np.random.randint(i+1, n)
                s_e.append(traj[i]['state']); s_l.append(traj[j]['state'])
                gs.append(traj[0]['goal'])
        self.s_e = torch.FloatTensor(np.array(s_e))
        self.s_l = torch.FloatTensor(np.array(s_l))
        self.gs = torch.FloatTensor(np.array(gs))
    def __len__(self): return len(self.s_e)
    def __getitem__(self, idx): return self.s_e[idx], self.s_l[idx], self.gs[idx]


class FailureDataset(Dataset):
    """States from failure trajectories, all labeled H_FAILURE."""
    def __init__(self, failure_demos):
        states, goals, entropy_labels = [], [], []
        for traj in failure_demos:
            for step in traj:
                states.append(step['state'])
                goals.append(step['goal'])
                entropy_labels.append(step['entropy_label'])
        self.states = torch.FloatTensor(np.array(states))
        self.goals = torch.FloatTensor(np.array(goals))
        self.entropy_labels = torch.FloatTensor(np.array(entropy_labels))

    def __len__(self): return len(self.states)
    def __getitem__(self, idx):
        return {'state': self.states[idx], 'goal': self.goals[idx],
                'entropy_label': self.entropy_labels[idx]}


# ============================================================
# NETWORKS (unchanged from v2)
# ============================================================

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

class ScoreNet(nn.Module):
    def __init__(self, h=128, max_k=20):
        super().__init__()
        self.emb = nn.Embedding(max_k, 16)
        self.net = nn.Sequential(
            nn.Linear(2+2+2+16, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2))
    def forward(self, a, s, g, k):
        return self.net(torch.cat([a, s, g, self.emb(k)], -1))

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
        self.pi_head = nn.Linear(h, K)
        self.mu_head = nn.Linear(h, K*2)
        self.sig_head = nn.Linear(h, K*2)
    def forward(self, s, g):
        f = self.backbone(torch.cat([s, g], -1))
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

def train_entropy(net, pairs, ds, failure_ds=None, ep=60):
    """
    Three loss terms:
      1. Ordering loss on success pairs (earlier state < later state in entropy)
      2. Soft regression on success states (entropy ≈ log(remaining + 1))
      3. [NEW] Regression on failure states (entropy ≈ H_FAILURE)

    The failure loss term creates high-entropy walls around obstacles.
    No special pairwise margin between failure and success states is needed:
    H_FAILURE is set high enough that the regression signal alone dominates.
    """
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pl = DataLoader(pairs, batch_size=256, shuffle=True)
    rl = DataLoader(ds, batch_size=256, shuffle=True)
    fl = DataLoader(failure_ds, batch_size=128, shuffle=True) if failure_ds is not None else None
    losses = []
    for e in range(ep):
        tot, n = 0, 0
        # Ordering + anchor loss on success pairs
        for se, sl, g in pl:
            he, hl = net(se, g), net(sl, g)
            loss = torch.relu(hl - he + 0.3).mean() + 0.5*(net(g, g)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        # Soft regression on success states
        for b in rl:
            reg = nn.functional.mse_loss(net(b['state'], b['goal']), b['entropy_label'])
            opt.zero_grad(); (0.3*reg).backward(); opt.step()
            tot += reg.item(); n += 1
        # Regression on failure states — full weight (no 0.3 downscaling)
        if fl is not None:
            for bf in fl:
                fail_reg = nn.functional.mse_loss(
                    net(bf['state'], bf['goal']), bf['entropy_label'])
                opt.zero_grad(); fail_reg.backward(); opt.step()
                tot += fail_reg.item(); n += 1
        losses.append(tot/max(n,1))
        if (e+1) % 20 == 0: print(f"  Entropy {e+1}/{ep}: {losses[-1]:.4f}")
    return losses

def train_dyn(net, ds, ep=40):
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ld = DataLoader(ds, batch_size=256, shuffle=True)
    losses = []
    for e in range(ep):
        tot, n = 0, 0
        for b in ld:
            loss = nn.functional.mse_loss(net(b['state'], b['action']), b['next_state'])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        losses.append(tot/max(n,1))
        if (e+1) % 20 == 0: print(f"  Dynamics {e+1}/{ep}: {losses[-1]:.4f}")
    return losses

def train_score(net, ds, K=10, ep=60):
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ld = DataLoader(ds, batch_size=256, shuffle=True)
    betas = torch.linspace(0.01, 0.5, K)
    alphas = 1.0 - betas
    abars = torch.cumprod(alphas, 0)
    losses = []
    for e in range(ep):
        tot, n = 0, 0
        for b in ld:
            a = b['action']; bsz = a.shape[0]
            k = torch.randint(0, K, (bsz,))
            ab = abars[k].unsqueeze(-1)
            noise = torch.randn_like(a)
            an = torch.sqrt(ab)*a + torch.sqrt(1-ab)*noise
            loss = nn.functional.mse_loss(net(an, b['state'], b['goal'], k), noise)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        losses.append(tot/max(n,1))
        if (e+1) % 20 == 0: print(f"  Score {e+1}/{ep}: {losses[-1]:.4f}")
    return losses, betas, abars

def train_bc(net, ds, ep=60):
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ld = DataLoader(ds, batch_size=256, shuffle=True)
    losses = []
    for e in range(ep):
        tot, n = 0, 0
        for b in ld:
            loss = nn.functional.mse_loss(net(b['state'], b['goal']), b['action'])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        losses.append(tot/max(n,1))
        if (e+1) % 20 == 0: print(f"  BC {e+1}/{ep}: {losses[-1]:.4f}")
    return losses

def train_mdn(net, ds, ep=60):
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ld = DataLoader(ds, batch_size=256, shuffle=True)
    losses = []
    for e in range(ep):
        tot, n = 0, 0
        for b in ld:
            pi, mu, sig = net(b['state'], b['goal'])
            a = b['action'].unsqueeze(1)
            lp = -0.5*((a-mu)/sig)**2 - torch.log(sig) - 0.5*np.log(2*np.pi)
            lp = lp.sum(-1) + torch.log(pi+1e-8)
            loss = -torch.logsumexp(lp, -1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        losses.append(tot/max(n,1))
        if (e+1) % 20 == 0: print(f"  MDN {e+1}/{ep}: {losses[-1]:.4f}")
    return losses


# ============================================================
# INFERENCE
# ============================================================

def efp_sample(score_net, entropy_net, dyn_net, state, goal, betas, abars, K=5, guidance=0.3):
    s = torch.FloatTensor(state).unsqueeze(0)
    g = torch.FloatTensor(goal).unsqueeze(0)
    with torch.no_grad():
        h = entropy_net(s, g).item()
    escale = min(h / 3.0, 1.0)
    a = torch.randn(1, 2) * escale
    al = 1.0 - betas
    with torch.no_grad():
        for k in reversed(range(K)):
            pn = score_net(a, s, g, torch.LongTensor([k]))
            am = (1/torch.sqrt(al[k]))*(a - (betas[k]/torch.sqrt(1-abars[k]))*pn)
            if guidance > 0 and k > 0 and k % 2 == 0:
                eps = 0.02
                sf = dyn_net(s, am)
                hc = entropy_net(sf, g)
                grad = torch.zeros_like(am)
                for d in range(2):
                    ap = am.clone(); ap[0,d] += eps
                    grad[0,d] = (entropy_net(dyn_net(s, ap), g) - hc) / eps
                am = am - guidance * grad * escale
            if k > 0:
                a = am + torch.randn_like(a)*torch.sqrt(betas[k])*escale
            else:
                a = am
    return a.squeeze(0).numpy()


# ============================================================
# EVALUATION
# ============================================================

def rollout(env, pol, start, max_t=80, perturb_t=None):
    pos = start.copy()
    traj = [pos.copy()]
    for t in range(max_t):
        if env.reached_goal(pos): break
        if perturb_t and t == perturb_t:
            pos = pos + np.random.randn(2)*0.4
            pos[0] = np.clip(pos[0], -3, 3)
            pos[1] = np.clip(pos[1], -3, 3)
        a = pol(pos, env.goal)
        a = np.clip(a, -env.max_speed, env.max_speed)
        pos = env.step(pos, a)
        traj.append(pos.copy())
    return {'trajectory': np.array(traj),
            'final_dist': np.linalg.norm(pos - env.goal),
            'success': env.reached_goal(pos), 'steps': len(traj)}


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("ENTROPIC FLOW POLICY — Multi-Modal Obstacle Navigation v3")
    print("  + Failure Demonstration Supervision")
    print("="*60)

    np.random.seed(42); torch.manual_seed(42)
    env = ObstacleNavEnv()

    print("\n[1] Generating success demonstrations...")
    demos = generate_demos(env, n_demos=300, n_points=48)

    print("\n[2] Generating failure demonstrations...")
    failure_demos = generate_failure_demos(env, n_total=120)

    print("\n[3] Datasets...")
    ds = NavDataset(demos)
    pairs = PairDataset(demos, 20)
    failure_ds = FailureDataset(failure_demos)
    print(f"  Success: {len(ds)} transitions, {len(pairs)} pairs")
    print(f"  Failure: {len(failure_ds)} transitions (H_FAILURE={H_FAILURE:.2f})")

    H = 128
    enet = EntropyNet(H); dnet = DynNet(H); snet = ScoreNet(H)
    bcnet = BCNet(H); mdn = MDN(H, 3)

    print("\n[4] Training entropy estimator (with failure supervision)...")
    hl = train_entropy(enet, pairs, ds, failure_ds, ep=60)
    print("\n[5] Training dynamics...")
    dl = train_dyn(dnet, ds, 40)
    print("\n[6] Training score network...")
    sl, betas, abars = train_score(snet, ds, 10, 60)
    print("\n[7] Training BC...")
    bl = train_bc(bcnet, ds, 60)
    print("\n[8] Training MDN-BC...")
    ml = train_mdn(mdn, ds, 60)

    enet.eval(); dnet.eval(); snet.eval(); bcnet.eval(); mdn.eval()

    def efp_pol(p, g):
        return efp_sample(snet, enet, dnet, p, g, betas, abars, 5, 0.3)
    def bc_pol(p, g):
        with torch.no_grad():
            return bcnet(torch.FloatTensor(p).unsqueeze(0),
                        torch.FloatTensor(g).unsqueeze(0)).squeeze(0).numpy()
    def mdn_pol(p, g):
        with torch.no_grad():
            return mdn.sample(torch.FloatTensor(p).unsqueeze(0),
                             torch.FloatTensor(g).unsqueeze(0)).squeeze(0).numpy()

    pols = {'EFP (Ours)': efp_pol, 'BC (Mean)': bc_pol, 'MDN-BC': mdn_pol}

    print("\n[9] Evaluation...")
    np.random.seed(777)

    id_starts = [np.array([np.random.uniform(-2.5,-1.5), np.random.uniform(-2,2)]) for _ in range(25)]
    ood_starts = [np.array([np.random.uniform(-0.8,-0.4), np.random.uniform(-2,2)]) for _ in range(15)]
    ood_starts += [np.array([np.random.uniform(-2.8,-2.5), np.random.uniform(-2.8,2.8)]) for _ in range(10)]

    results = {}
    for cond, starts, pert in [('In-Dist', id_starts, None),
                                ('OOD', ood_starts, None),
                                ('Perturb', id_starts[:15], 15)]:
        print(f"\n  === {cond} ===")
        results[cond] = {}
        for name, pol in pols.items():
            runs = [rollout(env, pol, s, 80, pert) for s in starts]
            sr = np.mean([r['success'] for r in runs])
            ad = np.mean([r['final_dist'] for r in runs])
            print(f"    {name:<14} success={sr:.0%}  dist={ad:.3f}")
            results[cond][name] = runs

    # ============ VISUALIZATION ============
    print("\n[10] Plotting...")

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('Entropic Flow Policy v3 — Failure Supervision + Multi-Modal Navigation',
                 fontsize=14, fontweight='bold', y=0.99)

    mc = {'EFP (Ours)': '#1565C0', 'BC (Mean)': '#E53935', 'MDN-BC': '#43A047'}

    def draw_obs(ax):
        for xmin, xmax, ymin, ymax in env.obstacles:
            ax.add_patch(plt.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin,
                         color='#424242', alpha=0.85, zorder=3))

    # Row 1: Trajectories per condition
    for ci, cond in enumerate(['In-Dist', 'OOD', 'Perturb']):
        ax = fig.add_subplot(3, 4, ci+1)
        draw_obs(ax)
        for name in pols:
            for r in results[cond][name][:10]:
                t = r['trajectory']
                ax.plot(t[:,0], t[:,1], color=mc[name], alpha=0.4, lw=0.9)
        ax.plot(*env.goal, 'r*', ms=14, zorder=5)
        ax.set_xlim(-3.1,3.1); ax.set_ylim(-3.1,3.1); ax.set_aspect('equal')
        ax.set_title(f'{cond} Trajectories', fontsize=11)
        ax.grid(True, alpha=0.2)
        if ci == 0:
            for n, c in mc.items(): ax.plot([],[],color=c,lw=2,label=n)
            ax.legend(fontsize=7, loc='lower left')

    # Expert demos + failure demos overlay
    ax = fig.add_subplot(3, 4, 4)
    draw_obs(ax)
    mode_colors = {'upper':'#1565C0','lower':'#E53935','middle':'#43A047'}
    for d in demos[:60]:
        t = np.array([s['state'] for s in d])
        ax.plot(t[:,0], t[:,1], color=mode_colors[d[0]['mode']], alpha=0.3, lw=0.7)
    for d in failure_demos[:40]:
        t = np.array([s['state'] for s in d])
        ax.plot(t[:,0], t[:,1], color='#FF6F00', alpha=0.5, lw=0.7)
    ax.plot(*env.goal, 'r*', ms=14, zorder=5)
    ax.set_xlim(-3.1,3.1); ax.set_ylim(-3.1,3.1); ax.set_aspect('equal')
    ax.set_title('Expert + Failure Demos', fontsize=11)
    for m,c in mode_colors.items(): ax.plot([],[],color=c,label=m.capitalize())
    ax.plot([],[],color='#FF6F00',label='Failure')
    ax.legend(fontsize=7, loc='lower left'); ax.grid(True, alpha=0.2)

    # Row 2: Success rates
    for ci, cond in enumerate(['In-Dist', 'OOD', 'Perturb']):
        ax = fig.add_subplot(3, 4, 5+ci)
        names = list(pols.keys())
        rates = [np.mean([r['success'] for r in results[cond][n]]) for n in names]
        bars = ax.bar(range(len(names)), rates, color=[mc[n] for n in names],
                     width=0.6, edgecolor='black', lw=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(' ','\n') for n in names], fontsize=8)
        ax.set_ylabel('Success Rate'); ax.set_title(f'{cond} Success', fontsize=11)
        ax.set_ylim(0, 1.15)
        for b, r in zip(bars, rates):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                    f'{r:.0%}', ha='center', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    # Avg distance grouped
    ax = fig.add_subplot(3, 4, 8)
    conds = ['In-Dist', 'OOD', 'Perturb']
    x = np.arange(len(conds)); w = 0.25
    for i, name in enumerate(pols):
        means = [np.mean([r['final_dist'] for r in results[c][name]]) for c in conds]
        ax.bar(x + i*w - w, means, w, label=name, color=mc[name], edgecolor='black', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel('Avg Final Distance'); ax.set_title('Final Distance Comparison', fontsize=11)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    # Row 3: Entropy landscape (key diagnostic — should now show walls near obstacles)
    ax = fig.add_subplot(3, 4, 9)
    enet.eval()
    xs = np.linspace(-3, 3, 80); ys = np.linspace(-3, 3, 80)
    Hmap = np.zeros((80,80))
    gt = torch.FloatTensor(env.goal).unsqueeze(0)
    with torch.no_grad():
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                Hmap[j,i] = enet(torch.FloatTensor([[x,y]]), gt).item()
    # Mask interior of obstacles; boundary regions should show elevated entropy
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if env.in_obstacle(np.array([x,y])):
                Hmap[j,i] = np.nan
    im = ax.contourf(xs, ys, np.ma.masked_invalid(Hmap), levels=30, cmap='viridis_r')
    draw_obs(ax)
    ax.plot(*env.goal, 'r*', ms=14, zorder=5)
    ax.set_xlim(-3.1,3.1); ax.set_ylim(-3.1,3.1); ax.set_aspect('equal')
    ax.set_title('Learned Entropy Landscape (v3)', fontsize=11)
    plt.colorbar(im, ax=ax, label='H(s,g)', shrink=0.8)

    # Losses
    ax = fig.add_subplot(3, 4, 10)
    ax.plot(hl, label='Entropy', color='#1565C0', lw=1.5)
    ax.plot(sl, label='Score', color='#9C27B0', lw=1.5)
    ax.plot(bl, label='BC', color='#E53935', lw=1.5)
    ax.plot(ml, label='MDN', color='#43A047', lw=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Losses', fontsize=11)
    ax.legend(fontsize=8); ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    # Entropy profiles along EFP rollouts
    ax = fig.add_subplot(3, 4, 11)
    for r in results['In-Dist']['EFP (Ours)'][:15]:
        t = r['trajectory']
        hs = []
        with torch.no_grad():
            for p in t:
                hs.append(enet(torch.FloatTensor(p).unsqueeze(0), gt).item())
        c = '#1565C0' if r['success'] else '#FF9800'
        ax.plot(hs, color=c, alpha=0.6, lw=1.2)
    ax.set_xlabel('Timestep'); ax.set_ylabel('H(s,g)')
    ax.set_title('Entropy Along EFP Rollouts', fontsize=11)
    ax.plot([],[], color='#1565C0', label='Success')
    ax.plot([],[], color='#FF9800', label='Failure')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Summary
    ax = fig.add_subplot(3, 4, 12); ax.axis('off')
    txt = "RESULTS SUMMARY (v3)\n" + "="*42 + "\n\n"
    for cond in conds:
        txt += f"{cond}:\n"
        for name in pols:
            sr = np.mean([r['success'] for r in results[cond][name]])
            ad = np.mean([r['final_dist'] for r in results[cond][name]])
            txt += f"  {name:<14} {sr:5.0%}  d={ad:.3f}\n"
        txt += "\n"
    txt += f"\nH_FAILURE = log(200) = {H_FAILURE:.2f}\n"
    txt += f"Failure demos: {len(failure_demos)} trajs\n"
    txt += f"Failure states: {len(failure_ds)}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = '/home/sakib/working_dir/entropyFlowPolicy/efp_v3_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
