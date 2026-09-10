"""
efp_guided_bc.py — Entropy-guided BC: apply EFP entropy gradient on top of
the plain BC (Mean) prediction and sweep over guidance scales.

BC outputs a single mean action (averaged over the multimodal demo
distribution).  The entropy gradient correction nudges that action toward
the direction that leads to a lower-entropy (more feasible) next state,
potentially recovering the missing navigation around the block.

Usage
-----
  python efp_guided_bc.py
"""

import os, sys, math
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from efp_pusht import (
    raw_to_state, reset_env, make_env,
    GOAL_VEC, IMG, SUCCESS_THR,
    EVAL_STD, EVAL_PERTURB,
)
from efp_pusht_real import load_all, get_stats, load_zarr, _infer_bc

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Guided-BC inference ────────────────────────────────────────────────────────

def infer_guided_bc(obs_raw, bc_net, enet, dyn_net, guidance):
    """BC prediction + single entropy-gradient correction step."""
    s = torch.tensor(raw_to_state(obs_raw), dtype=torch.float32).unsqueeze(0)
    g = torch.tensor(GOAL_VEC, dtype=torch.float32).unsqueeze(0)

    # Plain BC action (no grad needed)
    with torch.no_grad():
        a_bc = bc_net(torch.cat([s, g], dim=-1))   # (1, 2)

    if guidance == 0.0:
        return np.clip(a_bc.squeeze(0).numpy() * IMG, 0, IMG).astype(np.float32)

    # Entropy gradient: ∂H/∂a  through  a → DynNet → s_next → EntropyNet → H
    a_d    = a_bc.detach().requires_grad_(True)
    s_next = dyn_net(s, a_d)
    h_val  = enet(torch.cat([s_next, g], dim=-1))
    h_val.backward()

    grad  = a_d.grad.data                              # (1, 2)
    gnorm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    a_guided = (a_bc.detach() - guidance * grad / gnorm).squeeze(0).numpy()
    return np.clip(a_guided * IMG, 0, IMG).astype(np.float32)


# ── Multi-step variant ─────────────────────────────────────────────────────────

def infer_guided_bc_multistep(obs_raw, bc_net, enet, dyn_net,
                               guidance, n_steps=5):
    """
    Start from BC action, then take n_steps gradient-descent steps on H.
    Stronger correction than single-step but still anchored to BC's starting point.
    """
    s = torch.tensor(raw_to_state(obs_raw), dtype=torch.float32).unsqueeze(0)
    g = torch.tensor(GOAL_VEC, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        a = bc_net(torch.cat([s, g], dim=-1))   # (1, 2)

    for _ in range(n_steps):
        a_d    = a.detach().requires_grad_(True)
        s_next = dyn_net(s, a_d)
        h_val  = enet(torch.cat([s_next, g], dim=-1))
        h_val.backward()
        grad   = a_d.grad.data
        gnorm  = grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        a      = a_d.detach() - guidance * grad / gnorm

    return np.clip(a.squeeze(0).numpy() * IMG, 0, IMG).astype(np.float32)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(policy_fn, label, eval_set, max_steps=350):
    env = make_env()
    coverages, successes = [], []
    for ax, ay, bx_off, by_off, ba_off in eval_set:
        obs, _ = reset_env(env, agent_x=ax, agent_y=ay,
                           bx_off=bx_off, by_off=by_off, ba_off=ba_off)
        max_cov = 0.0
        for _ in range(max_steps):
            action = policy_fn(obs)
            obs, reward, done, truncated, _ = env.step(action)
            max_cov = max(max_cov, float(reward))
            if done or truncated:
                break
        coverages.append(max_cov)
        successes.append(float(max_cov > SUCCESS_THR))
    env.close()
    mc   = float(np.mean(coverages))
    succ = float(np.mean(successes))
    print(f"  {label:<36} cov={mc:.3f}  success={int(succ*100):3d}%")
    return mc, succ


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Entropy-guided BC — guidance scale sweep")
    print("=" * 60)

    print("\nLoading models...")
    enet_full, enet_nofail, dyn_net, efp_vel_unet, bc_net, mdn_net, dp_model = load_all()

    # Guidance values to sweep (single-step)
    guidances = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    results_std     = {}
    results_perturb = {}

    print("\n--- Standard condition (agent top-centre) ---")
    # Plain BC baseline
    mc, su = evaluate(lambda o: _infer_bc(raw_to_state(o), bc_net),
                      "BC (no guidance)", EVAL_STD)
    results_std["BC (plain)"] = (mc, su)

    for g in guidances:
        label = f"BC + enet_full  g={g}"
        fn = lambda o, _g=g: infer_guided_bc(o, bc_net, enet_full, dyn_net, _g)
        mc, su = evaluate(fn, label, EVAL_STD)
        results_std[label] = (mc, su)

    # Multi-step with best single-step scale
    for n in [3, 5, 10]:
        label = f"BC + multistep({n})  g=1.0"
        fn = lambda o, _n=n: infer_guided_bc_multistep(
            o, bc_net, enet_full, dyn_net, guidance=1.0, n_steps=_n)
        mc, su = evaluate(fn, label, EVAL_STD)
        results_std[label] = (mc, su)

    print("\n--- Perturb condition (agent above block) ---")
    mc, su = evaluate(lambda o: _infer_bc(raw_to_state(o), bc_net),
                      "BC (no guidance)", EVAL_PERTURB)
    results_perturb["BC (plain)"] = (mc, su)

    for g in guidances:
        label = f"BC + enet_full  g={g}"
        fn = lambda o, _g=g: infer_guided_bc(o, bc_net, enet_full, dyn_net, _g)
        mc, su = evaluate(fn, label, EVAL_PERTURB)
        results_perturb[label] = (mc, su)

    for n in [3, 5, 10]:
        label = f"BC + multistep({n})  g=1.0"
        fn = lambda o, _n=n: infer_guided_bc_multistep(
            o, bc_net, enet_full, dyn_net, guidance=1.0, n_steps=_n)
        mc, su = evaluate(fn, label, EVAL_PERTURB)
        results_perturb[label] = (mc, su)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Policy':<36} {'STD cov':>8} {'STD%':>6} {'PERTURB cov':>12} {'PERTURB%':>9}")
    print("-" * 75)
    all_labels = list(results_std.keys())
    for l in all_labels:
        if l in results_perturb:
            sc, ss = results_std[l]
            pc, ps = results_perturb[l]
            print(f"{l:<36} {sc:>8.3f} {int(ss*100):>5}%  {pc:>12.3f} {int(ps*100):>8}%")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sweep_labels = ["BC (plain)"] + [f"BC + enet_full  g={g}" for g in guidances]
    g_vals = [-0.05] + list(guidances)

    for ax, (results, title) in zip(axes, [
        (results_std,     "Standard"),
        (results_perturb, "Perturb (agent above block)"),
    ]):
        covs  = [results[l][0] for l in sweep_labels]
        succs = [results[l][1] * 100 for l in sweep_labels]
        ax2   = ax.twinx()
        ax.plot(g_vals, covs,  'o-', color='#3498db', lw=2, label='Coverage')
        ax2.plot(g_vals, succs, 's--', color='#e74c3c', lw=2, label='Success %')
        ax.axhline(SUCCESS_THR, color='grey', ls=':', lw=1,
                   label=f'cov threshold ({SUCCESS_THR})')
        ax.set_xlabel('Guidance scale', fontsize=11)
        ax.set_ylabel('Mean max coverage', color='#3498db', fontsize=10)
        ax2.set_ylabel('Success %', color='#e74c3c', fontsize=10)
        ax.set_title(f'BC + entropy guidance — {title}', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 0.6); ax2.set_ylim(0, 110)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xticks(g_vals)
        ax.set_xticklabels([str(g) for g in g_vals], rotation=30, ha='right')

    plt.suptitle('Entropy-guided BC: guidance scale sweep\n'
                 '(enet_full = trained with failure supervision)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(BASE, 'efp_guided_bc_results.png')
    plt.savefig(out, dpi=150)
    print(f"\n[plot saved → {out}]")


if __name__ == '__main__':
    main()
