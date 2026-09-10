"""
Entropic Flow Policy (EFP) — Proof of Concept
===============================================

Task: 2D Planar Arm Reaching (3-link, redundant)
    - 3 revolute joints → 3D joint space
    - Target: 2D end-effector position
    - Redundancy means MULTIPLE joint configurations reach the same target
      → natural funnel / entropy contraction structure

Why this task?
    - Expert demonstrations generated analytically via IK + trajectory interpolation
    - Multiple valid solutions exist (high entropy far from goal)
    - Near the goal, feasible joint configs converge (low entropy)
    - Simple enough to validate, complex enough to be meaningful

Pipeline:
    1. Generate expert demonstrations using inverse kinematics
    2. Train entropy estimator H_θ(s, s_g) from demonstrations
    3. Train dynamics model f_φ(s, a) from demonstrations
    4. Train score network ε_ψ (entropic denoising direction)
    5. At inference: run entropic diffusion to select actions
    6. Compare against behavioral cloning baseline
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from collections import defaultdict

# ============================================================
# SECTION 1: Environment — 3-Link Planar Arm
# ============================================================

class PlanarArm:
    """
    3-link planar robot arm.
    State: joint angles [θ1, θ2, θ3]
    Action: joint velocities [dθ1, dθ2, dθ3]
    """
    def __init__(self, link_lengths=(1.0, 0.8, 0.6)):
        self.link_lengths = np.array(link_lengths)
        self.n_joints = len(link_lengths)
        self.dt = 0.05  # timestep

    def forward_kinematics(self, joints):
        """Compute end-effector position from joint angles."""
        x, y = 0.0, 0.0
        angle_sum = 0.0
        positions = [(x, y)]
        for i in range(self.n_joints):
            angle_sum += joints[i]
            x += self.link_lengths[i] * np.cos(angle_sum)
            y += self.link_lengths[i] * np.sin(angle_sum)
            positions.append((x, y))
        return np.array([x, y]), positions

    def jacobian(self, joints):
        """Compute the 2x3 Jacobian."""
        J = np.zeros((2, self.n_joints))
        for i in range(self.n_joints):
            angle_sum = np.sum(joints[:i+1])
            for j in range(i+1):
                # Contribution of joint j to link i
                pass
        # Correct Jacobian computation
        J = np.zeros((2, self.n_joints))
        for j in range(self.n_joints):
            angle_sum = np.sum(joints[:j+1])
            # Joint j affects all links from j onward
            for k in range(j, self.n_joints):
                cumulative_angle = np.sum(joints[:k+1])
                J[0, j] += -self.link_lengths[k] * np.sin(cumulative_angle)
                J[1, j] +=  self.link_lengths[k] * np.cos(cumulative_angle)
        return J

    def step(self, joints, action):
        """Apply joint velocity action, return new joints."""
        new_joints = joints + action * self.dt
        # Wrap to [-pi, pi]
        new_joints = (new_joints + np.pi) % (2 * np.pi) - np.pi
        return new_joints

    def inverse_kinematics(self, target, q_init=None, max_iter=200, tol=1e-3):
        """Damped least squares IK from a given initial config."""
        if q_init is None:
            q_init = np.random.uniform(-np.pi, np.pi, self.n_joints)
        q = q_init.copy()
        damping = 0.1

        for _ in range(max_iter):
            ee, _ = self.forward_kinematics(q)
            error = target - ee
            if np.linalg.norm(error) < tol:
                return q, True
            J = self.jacobian(q)
            # Damped least squares: dq = J^T (J J^T + λ²I)^{-1} e
            JJt = J @ J.T + damping**2 * np.eye(2)
            dq = J.T @ np.linalg.solve(JJt, error)
            q = q + dq * 0.5
            q = (q + np.pi) % (2 * np.pi) - np.pi

        return q, False


# ============================================================
# SECTION 2: Expert Demonstration Generation
# ============================================================

def generate_demonstrations(arm, n_demos=500, n_targets=20, steps_per_demo=40):
    """
    Generate expert demonstrations.

    For each target, solve IK from MULTIPLE random initial configs
    → multiple trajectories to same goal → empirical entropy structure.

    Each demonstration: sequence of (state, action, next_state, goal, timestep)
    """
    demonstrations = []
    targets = []

    # Generate reachable targets
    max_reach = sum(arm.link_lengths) * 0.7  # stay within comfortable range
    min_reach = arm.link_lengths[0] * 0.3

    for _ in range(n_targets):
        angle = np.random.uniform(-np.pi, np.pi)
        radius = np.random.uniform(min_reach, max_reach)
        target = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        targets.append(target)

    demos_per_target = n_demos // n_targets

    for target in targets:
        for _ in range(demos_per_target):
            # Random starting configuration
            q_start = np.random.uniform(-np.pi/2, np.pi/2, arm.n_joints)

            # Solve IK for goal configuration
            q_goal, success = arm.inverse_kinematics(target, q_init=None)
            if not success:
                continue

            # Generate smooth trajectory via linear interpolation in joint space
            # (simple but effective for this proof of concept)
            trajectory = []
            for t in range(steps_per_demo):
                alpha = t / (steps_per_demo - 1)
                # Smooth interpolation with cosine schedule
                alpha_smooth = 0.5 * (1 - np.cos(np.pi * alpha))
                q_t = q_start * (1 - alpha_smooth) + q_goal * alpha_smooth
                q_t = (q_t + np.pi) % (2 * np.pi) - np.pi

                if t < steps_per_demo - 1:
                    alpha_next = (t + 1) / (steps_per_demo - 1)
                    alpha_next_smooth = 0.5 * (1 - np.cos(np.pi * alpha_next))
                    q_next = q_start * (1 - alpha_next_smooth) + q_goal * alpha_next_smooth
                    q_next = (q_next + np.pi) % (2 * np.pi) - np.pi
                    action = (q_next - q_t) / arm.dt
                else:
                    action = np.zeros(arm.n_joints)

                ee_pos, _ = arm.forward_kinematics(q_t)

                trajectory.append({
                    'state': q_t.copy(),           # joint angles
                    'ee_pos': ee_pos.copy(),        # end-effector position
                    'action': action.copy(),        # joint velocities
                    'goal': target.copy(),          # target EE position
                    'q_goal': q_goal.copy(),        # goal joint config
                    'timestep': t,
                    'total_steps': steps_per_demo,
                    'normalized_time': alpha,        # 0→1, proxy for entropy
                })

            demonstrations.append(trajectory)

    print(f"Generated {len(demonstrations)} demonstrations "
          f"to {n_targets} targets")
    return demonstrations, targets


# ============================================================
# SECTION 3: Dataset Preparation
# ============================================================

class EFPDataset(Dataset):
    """
    Dataset for training all three EFP components.
    Each sample: (state, action, next_state, goal, entropy_label)
    """
    def __init__(self, demonstrations):
        self.states = []
        self.actions = []
        self.next_states = []
        self.goals = []
        self.entropy_labels = []
        self.ee_positions = []
        self.timesteps = []

        for traj in demonstrations:
            for i, step in enumerate(traj[:-1]):
                self.states.append(step['state'])
                self.actions.append(step['action'])
                self.next_states.append(traj[i+1]['state'])
                self.goals.append(step['q_goal'])  # goal in joint space
                self.ee_positions.append(step['ee_pos'])

                # Entropy label: proportional to remaining steps
                # More remaining steps → higher entropy
                remaining = step['total_steps'] - step['timestep']
                self.entropy_labels.append(np.log(remaining + 1))

                self.timesteps.append(step['timestep'])

        self.states = torch.FloatTensor(np.array(self.states))
        self.actions = torch.FloatTensor(np.array(self.actions))
        self.next_states = torch.FloatTensor(np.array(self.next_states))
        self.goals = torch.FloatTensor(np.array(self.goals))
        self.entropy_labels = torch.FloatTensor(np.array(self.entropy_labels))
        self.ee_positions = torch.FloatTensor(np.array(self.ee_positions))
        self.timesteps = torch.LongTensor(np.array(self.timesteps))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'next_state': self.next_states[idx],
            'goal': self.goals[idx],
            'entropy_label': self.entropy_labels[idx],
            'ee_pos': self.ee_positions[idx],
            'timestep': self.timesteps[idx],
        }


class PairDataset(Dataset):
    """
    Dataset of state PAIRS from same trajectory for ordering loss.
    (s_early, s_late, goal) → H(s_early) > H(s_late)
    """
    def __init__(self, demonstrations, pairs_per_traj=20):
        self.s_early = []
        self.s_late = []
        self.goals = []

        for traj in demonstrations:
            n = len(traj)
            for _ in range(pairs_per_traj):
                i = np.random.randint(0, n - 2)
                j = np.random.randint(i + 1, n)
                self.s_early.append(traj[i]['state'])
                self.s_late.append(traj[j]['state'])
                self.goals.append(traj[0]['q_goal'])

        self.s_early = torch.FloatTensor(np.array(self.s_early))
        self.s_late = torch.FloatTensor(np.array(self.s_late))
        self.goals = torch.FloatTensor(np.array(self.goals))

    def __len__(self):
        return len(self.s_early)

    def __getitem__(self, idx):
        return self.s_early[idx], self.s_late[idx], self.goals[idx]


# ============================================================
# SECTION 4: Neural Network Components
# ============================================================

class EntropyEstimator(nn.Module):
    """
    H_θ(s, s_g) → scalar entropy estimate.
    Input: [state; goal] concatenated
    Output: non-negative scalar (entropy)

    Trained with ordering loss: H(s_early) > H(s_late) for same trajectory.
    """
    def __init__(self, state_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # ensure non-negative entropy
        )

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return self.net(x).squeeze(-1)


class DynamicsModel(nn.Module):
    """
    f_φ(s, a) → s'
    Simple forward dynamics predictor.
    """
    def __init__(self, state_dim=3, action_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Residual: predict delta
        self.residual = True

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        if self.residual:
            return state + delta
        return delta


class EntropicScoreNetwork(nn.Module):
    """
    ε_ψ(a_noisy, s, s_g, k) → action correction direction

    This replaces the noise prediction network in DDPM.
    Instead of predicting added noise, it predicts the direction
    in action space that most reduces state entropy.

    k = diffusion step (integer, embedded)
    """
    def __init__(self, state_dim=3, action_dim=3, hidden_dim=128, max_steps=20):
        super().__init__()
        self.step_embed = nn.Embedding(max_steps, 32)
        self.net = nn.Sequential(
            nn.Linear(action_dim + state_dim * 2 + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, noisy_action, state, goal, step):
        step_emb = self.step_embed(step)
        x = torch.cat([noisy_action, state, goal, step_emb], dim=-1)
        return self.net(x)


class BehavioralCloning(nn.Module):
    """
    Baseline: standard behavioral cloning π_BC(s, s_g) → a
    """
    def __init__(self, state_dim=3, action_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)


# ============================================================
# SECTION 5: Training
# ============================================================

def train_entropy_estimator(model, pair_dataset, dataset, epochs=100, lr=1e-3):
    """
    Train H_θ with:
      1. Ordering loss: H(s_early, g) > H(s_late, g) + margin
      2. Anchor loss: H(s_goal, s_goal) ≈ 0
      3. Regression loss: H ≈ log(remaining_steps) (soft supervision)
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pair_loader = DataLoader(pair_dataset, batch_size=256, shuffle=True)
    reg_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    margin = 0.3

    losses_history = []

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        # Ordering loss
        for s_early, s_late, goals in pair_loader:
            h_early = model(s_early, goals)
            h_late = model(s_late, goals)

            # H(early) should be > H(late) by at least margin
            order_loss = torch.relu(h_late - h_early + margin).mean()

            # Anchor: H(goal, goal) = 0
            h_goal = model(goals, goals)
            anchor_loss = (h_goal ** 2).mean()

            loss = order_loss + 0.5 * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Regression loss (soft target)
        for batch in reg_loader:
            h_pred = model(batch['state'], batch['goal'])
            h_target = batch['entropy_label']
            reg_loss = nn.functional.mse_loss(h_pred, h_target)

            optimizer.zero_grad()
            (0.3 * reg_loss).backward()
            optimizer.step()
            total_loss += reg_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses_history.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  Entropy Estimator Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses_history


def train_dynamics_model(model, dataset, epochs=80, lr=1e-3):
    """Train f_φ(s, a) → s' with MSE loss."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    losses_history = []

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch in loader:
            s_pred = model(batch['state'], batch['action'])
            loss = nn.functional.mse_loss(s_pred, batch['next_state'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses_history.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  Dynamics Model Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses_history


def train_score_network(score_net, entropy_net, dynamics_net, dataset,
                        n_diffusion_steps=10, epochs=100, lr=1e-3):
    """
    Train ε_ψ to predict the entropic gradient direction.

    For each (s, a_expert, g):
      1. Add noise to a_expert at random diffusion step k
      2. Compute target = direction toward entropy reduction
         (approximated by: a_expert - a_noisy, scaled)
      3. Train ε_ψ to predict this direction
    """
    optimizer = optim.AdamW(score_net.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Noise schedule (linear, like DDPM)
    betas = torch.linspace(0.01, 0.5, n_diffusion_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    losses_history = []

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for batch in loader:
            a_expert = batch['action']
            state = batch['state']
            goal = batch['goal']
            bsz = a_expert.shape[0]

            # Random diffusion step
            k = torch.randint(0, n_diffusion_steps, (bsz,))
            alpha_bar_k = alpha_bars[k].unsqueeze(-1)

            # Add noise
            noise = torch.randn_like(a_expert)
            a_noisy = torch.sqrt(alpha_bar_k) * a_expert + \
                      torch.sqrt(1 - alpha_bar_k) * noise

            # Target: predict the noise (standard DDPM objective)
            # But conceptually, this noise points AWAY from the
            # entropy-minimizing action, so predicting it lets us
            # reverse toward entropy reduction
            predicted_noise = score_net(a_noisy, state, goal, k)

            loss = nn.functional.mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses_history.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  Score Network Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses_history, betas, alpha_bars


def train_bc_baseline(model, dataset, epochs=100, lr=1e-3):
    """Train behavioral cloning baseline."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    losses_history = []

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch in loader:
            a_pred = model(batch['state'], batch['goal'])
            loss = nn.functional.mse_loss(a_pred, batch['action'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses_history.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  BC Baseline Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses_history


# ============================================================
# SECTION 6: Inference — Entropic Diffusion Sampling
# ============================================================

def entropic_diffusion_sample(score_net, entropy_net, dynamics_net, state, goal,
                              betas, alpha_bars, n_steps=10, guidance_scale=1.0):
    """
    Generate an action via entropic diffusion.

    Key difference from standard diffusion:
      - Noise level is modulated by current state entropy
      - Denoising direction incorporates entropy gradient

    Args:
        state: current joint angles [3]
        goal: target joint angles [3]
        n_steps: diffusion denoising steps
        guidance_scale: how much to weight entropy gradient vs learned score
    """
    state_t = torch.FloatTensor(state).unsqueeze(0)
    goal_t = torch.FloatTensor(goal).unsqueeze(0)

    with torch.no_grad():
        # Estimate current state entropy
        h_current = entropy_net(state_t, goal_t).item()

    # Adaptive noise: scale initial noise by entropy
    # High entropy → more noise → broader search
    # Low entropy → less noise → precise action
    entropy_scale = min(h_current / 3.0, 1.0)  # normalize roughly

    # Start from noise, scaled by entropy
    a_t = torch.randn(1, 3) * entropy_scale

    alphas = 1.0 - betas
    
    with torch.no_grad():
        for k in reversed(range(n_steps)):
            k_tensor = torch.LongTensor([k])
            alpha_k = alphas[k]
            alpha_bar_k = alpha_bars[k]

            # Predict noise
            pred_noise = score_net(a_t, state_t, goal_t, k_tensor)

            # Standard DDPM reverse step
            a_mean = (1 / torch.sqrt(alpha_k)) * (
                a_t - (betas[k] / torch.sqrt(1 - alpha_bar_k)) * pred_noise
            )

            # Add entropy-guided correction
            if guidance_scale > 0 and k > 0:
                # Estimate entropy gradient numerically
                eps = 0.01
                grad = torch.zeros_like(a_mean)
                for d in range(3):
                    a_plus = a_mean.clone()
                    a_plus[0, d] += eps
                    a_minus = a_mean.clone()
                    a_minus[0, d] -= eps

                    s_plus = dynamics_net(state_t, a_plus)
                    s_minus = dynamics_net(state_t, a_minus)

                    h_plus = entropy_net(s_plus, goal_t)
                    h_minus = entropy_net(s_minus, goal_t)

                    grad[0, d] = (h_plus - h_minus) / (2 * eps)

                # Push toward lower entropy
                a_mean = a_mean - guidance_scale * grad * entropy_scale

            # Add noise (except at last step)
            if k > 0:
                noise = torch.randn_like(a_t) * torch.sqrt(betas[k]) * entropy_scale
                a_t = a_mean + noise
            else:
                a_t = a_mean

    return a_t.squeeze(0).numpy()


# ============================================================
# SECTION 7: Evaluation
# ============================================================

def evaluate_policy(arm, policy_fn, targets, q_starts, max_steps=60):
    """
    Evaluate a policy by rolling out trajectories.
    Returns: success rate, average final distance, trajectory data
    """
    results = []

    for target, q_start in zip(targets, q_starts):
        q = q_start.copy()
        trajectory = [q.copy()]
        ee_traj = []
        entropy_traj = []

        # Get a goal joint config via IK
        q_goal, success = arm.inverse_kinematics(target)
        if not success:
            continue

        for t in range(max_steps):
            ee, _ = arm.forward_kinematics(q)
            ee_traj.append(ee.copy())

            dist = np.linalg.norm(ee - target)
            if dist < 0.05:
                break

            action = policy_fn(q, q_goal)
            action = np.clip(action, -5.0, 5.0)
            q = arm.step(q, action)
            trajectory.append(q.copy())

        final_ee, _ = arm.forward_kinematics(q)
        final_dist = np.linalg.norm(final_ee - target)

        results.append({
            'target': target,
            'final_dist': final_dist,
            'success': final_dist < 0.1,
            'steps': len(trajectory),
            'trajectory': trajectory,
            'ee_trajectory': ee_traj,
        })

    return results


def run_experiment():
    """Main experiment: train EFP and BC, compare."""

    print("=" * 60)
    print("ENTROPIC FLOW POLICY — Proof of Concept Experiment")
    print("=" * 60)

    # --- Setup ---
    arm = PlanarArm(link_lengths=(1.0, 0.8, 0.6))
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Generate demonstrations ---
    print("\n[1] Generating expert demonstrations...")
    demos, train_targets = generate_demonstrations(
        arm, n_demos=600, n_targets=30, steps_per_demo=40
    )

    # --- Prepare datasets ---
    print("\n[2] Preparing datasets...")
    dataset = EFPDataset(demos)
    pair_dataset = PairDataset(demos, pairs_per_traj=30)
    print(f"  Transition samples: {len(dataset)}")
    print(f"  Ordering pairs: {len(pair_dataset)}")

    # --- Train components ---
    entropy_net = EntropyEstimator(state_dim=3, hidden_dim=128)
    dynamics_net = DynamicsModel(state_dim=3, action_dim=3, hidden_dim=128)
    score_net = EntropicScoreNetwork(state_dim=3, action_dim=3, hidden_dim=128)
    bc_net = BehavioralCloning(state_dim=3, action_dim=3, hidden_dim=128)

    print("\n[3] Training Entropy Estimator...")
    h_losses = train_entropy_estimator(entropy_net, pair_dataset, dataset, epochs=100)

    print("\n[4] Training Dynamics Model...")
    d_losses = train_dynamics_model(dynamics_net, dataset, epochs=80)

    print("\n[5] Training Entropic Score Network...")
    s_losses, betas, alpha_bars = train_score_network(
        score_net, entropy_net, dynamics_net, dataset, epochs=100
    )

    print("\n[6] Training Behavioral Cloning Baseline...")
    bc_losses = train_bc_baseline(bc_net, dataset, epochs=100)

    # --- Evaluation ---
    print("\n[7] Evaluating policies...")

    # Generate test scenarios
    n_test = 50
    test_targets = []
    test_q_starts = []
    max_reach = sum(arm.link_lengths) * 0.6
    min_reach = arm.link_lengths[0] * 0.4
    np.random.seed(999)
    for _ in range(n_test):
        angle = np.random.uniform(-np.pi, np.pi)
        radius = np.random.uniform(min_reach, max_reach)
        test_targets.append(np.array([radius * np.cos(angle), radius * np.sin(angle)]))
        test_q_starts.append(np.random.uniform(-np.pi/2, np.pi/2, 3))

    # EFP policy
    entropy_net.eval()
    dynamics_net.eval()
    score_net.eval()

    def efp_policy(q, q_goal):
        return entropic_diffusion_sample(
            score_net, entropy_net, dynamics_net, q, q_goal,
            betas, alpha_bars, n_steps=10, guidance_scale=0.5
        )

    # BC policy
    bc_net.eval()
    def bc_policy(q, q_goal):
        with torch.no_grad():
            s = torch.FloatTensor(q).unsqueeze(0)
            g = torch.FloatTensor(q_goal).unsqueeze(0)
            return bc_net(s, g).squeeze(0).numpy()

    print("\n  Evaluating EFP...")
    efp_results = evaluate_policy(arm, efp_policy, test_targets, test_q_starts)

    print("  Evaluating BC...")
    bc_results = evaluate_policy(arm, bc_policy, test_targets, test_q_starts)

    # --- Metrics ---
    efp_success = np.mean([r['success'] for r in efp_results])
    bc_success = np.mean([r['success'] for r in bc_results])
    efp_dist = np.mean([r['final_dist'] for r in efp_results])
    bc_dist = np.mean([r['final_dist'] for r in bc_results])
    efp_steps = np.mean([r['steps'] for r in efp_results])
    bc_steps = np.mean([r['steps'] for r in bc_results])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'EFP':>10} {'BC':>10}")
    print("-" * 45)
    print(f"{'Success Rate':<25} {efp_success:>10.1%} {bc_success:>10.1%}")
    print(f"{'Avg Final Distance':<25} {efp_dist:>10.4f} {bc_dist:>10.4f}")
    print(f"{'Avg Steps':<25} {efp_steps:>10.1f} {bc_steps:>10.1f}")

    # --- Save results for visualization ---
    results_data = {
        'efp_results': efp_results,
        'bc_results': bc_results,
        'h_losses': h_losses,
        'd_losses': d_losses,
        's_losses': s_losses,
        'bc_losses': bc_losses,
        'efp_success': efp_success,
        'bc_success': bc_success,
        'efp_dist': efp_dist,
        'bc_dist': bc_dist,
        'arm': arm,
        'entropy_net': entropy_net,
        'dynamics_net': dynamics_net,
        'test_targets': test_targets,
    }

    return results_data


# ============================================================
# SECTION 8: Visualization
# ============================================================

def create_visualizations(results_data):
    """Generate comprehensive visualization plots."""

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Entropic Flow Policy — Experimental Results',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Training losses
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(results_data['h_losses'], label='Entropy Est.', color='#2196F3', linewidth=1.5)
    ax1.plot(results_data['bc_losses'], label='BC Baseline', color='#FF5722', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # 2. Score network loss
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(results_data['s_losses'], color='#9C27B0', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Entropic Score Network Loss')
    ax2.grid(True, alpha=0.3)

    # 3. Success rate comparison
    ax3 = fig.add_subplot(2, 3, 3)
    methods = ['EFP\n(Ours)', 'Behavioral\nCloning']
    rates = [results_data['efp_success'], results_data['bc_success']]
    colors = ['#2196F3', '#FF5722']
    bars = ax3.bar(methods, rates, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Success Rate Comparison')
    ax3.set_ylim(0, 1.1)
    for bar, rate in zip(bars, rates):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Final distance distribution
    ax4 = fig.add_subplot(2, 3, 4)
    efp_dists = [r['final_dist'] for r in results_data['efp_results']]
    bc_dists = [r['final_dist'] for r in results_data['bc_results']]
    ax4.hist(efp_dists, bins=20, alpha=0.6, label='EFP', color='#2196F3', edgecolor='black', linewidth=0.5)
    ax4.hist(bc_dists, bins=20, alpha=0.6, label='BC', color='#FF5722', edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Final EE Distance to Target')
    ax4.set_ylabel('Count')
    ax4.set_title('Final Distance Distribution')
    ax4.axvline(x=0.1, color='green', linestyle='--', label='Success threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Example EE trajectories
    ax5 = fig.add_subplot(2, 3, 5)
    n_show = min(8, len(results_data['efp_results']))
    for i in range(n_show):
        if results_data['efp_results'][i]['ee_trajectory']:
            ee_traj = np.array(results_data['efp_results'][i]['ee_trajectory'])
            ax5.plot(ee_traj[:, 0], ee_traj[:, 1], 'b-', alpha=0.4, linewidth=1)
        if results_data['bc_results'][i]['ee_trajectory']:
            ee_traj = np.array(results_data['bc_results'][i]['ee_trajectory'])
            ax5.plot(ee_traj[:, 0], ee_traj[:, 1], 'r-', alpha=0.4, linewidth=1)
    # Plot targets
    for i in range(n_show):
        t = results_data['efp_results'][i]['target']
        ax5.plot(t[0], t[1], 'g*', markersize=10)
    ax5.plot([], [], 'b-', label='EFP')
    ax5.plot([], [], 'r-', label='BC')
    ax5.plot([], [], 'g*', label='Targets')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title('End-Effector Trajectories')
    ax5.legend()
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    # 6. Entropy landscape visualization
    ax6 = fig.add_subplot(2, 3, 6)
    entropy_net = results_data['entropy_net']
    entropy_net.eval()

    # Fix θ3=0, vary θ1 and θ2, compute entropy to a fixed goal
    arm = results_data['arm']
    target = results_data['test_targets'][0]
    q_goal_viz, _ = arm.inverse_kinematics(target)

    theta1_range = np.linspace(-np.pi, np.pi, 50)
    theta2_range = np.linspace(-np.pi, np.pi, 50)
    H_map = np.zeros((50, 50))

    with torch.no_grad():
        for i, t1 in enumerate(theta1_range):
            for j, t2 in enumerate(theta2_range):
                s = torch.FloatTensor([[t1, t2, 0.0]])
                g = torch.FloatTensor([q_goal_viz])
                H_map[j, i] = entropy_net(s, g).item()

    im = ax6.contourf(theta1_range, theta2_range, H_map, levels=30, cmap='viridis_r')
    ax6.plot(q_goal_viz[0], q_goal_viz[1], 'r*', markersize=15, label='Goal')
    ax6.set_xlabel('θ₁')
    ax6.set_ylabel('θ₂')
    ax6.set_title('Learned Entropy Landscape (θ₃=0)')
    ax6.legend()
    plt.colorbar(im, ax=ax6, label='H(s, s_g)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/home/claude/efp_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nVisualization saved to efp_results.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = run_experiment()
    create_visualizations(results)
    print("\nExperiment complete!")
