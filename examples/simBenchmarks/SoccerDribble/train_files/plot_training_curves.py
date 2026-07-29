"""Plot PPO training curves from TensorBoard event files.

TensorBoard's event log is the one artifact that stays continuous across
`--resume` runs (monitor CSVs and eval/evaluations.npz get overwritten each
time the script restarts — see README/train.log for the resume history:
0 -> 1M -> 3M steps across three separate `train_ppo.py` invocations, whose
events all land in the same `PPO_1/` dir).

Usage:
    python plot_training_curves.py --logdir runs/ppo_v1 --out runs/ppo_v1/training_curves.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

_TAGS = [
    ("rollout/success_rate", "Rollout success rate"),
    ("eval/success_rate", "Held-out eval success rate (5 ep)"),
    ("rollout/ep_rew_mean", "Rollout mean episode reward"),
    ("eval/mean_reward", "Held-out eval mean reward (5 ep)"),
    ("train/entropy_loss", "Policy entropy loss"),
    ("train/std", "Action std (exploration)"),
    ("train/explained_variance", "Value function explained variance"),
    ("rollout/ep_len_mean", "Mean episode length"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/ppo_v1")
    parser.add_argument("--tb-subdir", type=str, default="PPO_1")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    tb_dir = f"{args.logdir}/{args.tb_subdir}"
    out_path = args.out or f"{args.logdir}/training_curves.png"

    ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags()["scalars"])

    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    axes = axes.flatten()

    for ax, (tag, title) in zip(axes, _TAGS):
        if tag not in available:
            ax.set_visible(False)
            continue
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        is_eval = tag.startswith("eval/")
        ax.plot(steps, values, marker="o" if is_eval else None, markersize=3,
                linewidth=1 if is_eval else 0.8, alpha=0.9 if is_eval else 0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("timesteps")
        ax.grid(alpha=0.3)
        ax.axvline(1_003_808, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(2_000_000, color="gray", linestyle="--", linewidth=0.8, alpha=0.3)

    fig.suptitle(
        "SoccerDribble PPO training (0 -> 3M steps, 3 resumed runs; "
        "dashed lines mark resume boundaries at 1.0M and ~2.0M)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
