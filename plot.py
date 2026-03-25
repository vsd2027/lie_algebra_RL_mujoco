#!/usr/bin/env python3
"""
plot.py — Plot evaluation results from SB3's evaluations.npz

Usage:
    python plot.py runs/td3_her_a-rotvec*/evaluations.npz
    python plot.py runs/*/evaluations.npz  # overlay all runs
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(npz_paths):
    colors = ['#534AB7', '#D85A30', '#1D9E75', '#378ADD', '#D4537E', '#639922']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for i, path in enumerate(npz_paths):
        data = np.load(path)
        ts = data['timesteps']
        results = data['results']  # (n_evals, n_episodes)
        c = colors[i % len(colors)]

        # Label from directory
        label = os.path.basename(os.path.dirname(os.path.abspath(path)))
        label = label.split('__')[0].replace('td3_her_', '').replace('ddpg_her_', '')

        # Return
        mean_r = results.mean(axis=1)
        std_r = results.std(axis=1)
        axes[0].plot(ts, mean_r, color=c, linewidth=2, label=label)
        axes[0].fill_between(ts, mean_r - std_r, mean_r + std_r, color=c, alpha=0.12)

        # Success rate (from SB3's is_success tracking)
        if 'successes' in data:
            sr = data['successes'].mean(axis=1)
        else:
            # Fallback: episode succeeded if return > -max_steps
            max_steps = int(-results.min()) if results.min() < 0 else 100
            sr = (results > -max_steps).mean(axis=1)
        axes[1].plot(ts, sr, color=c, linewidth=2, label=label)

        # Steps to goal (= -mean_return)
        axes[2].plot(ts, -mean_r, color=c, linewidth=2, label=label)

    axes[0].set(title='Mean return', xlabel='Timesteps', ylabel='Return')
    axes[1].set(title='Success rate', xlabel='Timesteps', ylabel='Rate', ylim=(-0.05, 1.05))
    axes[2].set(title='Mean steps to goal', xlabel='Timesteps', ylabel='Steps')

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(npz_paths[0])), 'comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot.py evaluations.npz [more.npz ...]")
        sys.exit(1)
    plot(sys.argv[1:])
