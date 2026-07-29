"""Evaluate a trained SoccerDribble PPO model: run N random episodes and
report success rate / reward / episode length, optionally saving an mp4.

Usage:
    python eval_soccer.py --model ../train_files/runs/ppo_v1/best_model/best_model.zip --episodes 20
    python eval_soccer.py --model ... --episodes 3 --save-video rollout.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train_files"))
from config import SoccerDribbleConfig  # noqa: E402
from soccer_env import SoccerDribbleEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to a saved SB3 .zip model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--save-video", type=str, default=None, help="mp4 path; only records the first episode")
    args = parser.parse_args()

    cfg = SoccerDribbleConfig()
    render_mode = "rgb_array" if args.save_video else None
    env = SoccerDribbleEnv(cfg, render_mode=render_mode)
    model = PPO.load(args.model)

    frames = []
    rewards, successes, lengths = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward, steps = 0.0, 0
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if args.save_video and ep == 0:
                frames.append(env.render())
        rewards.append(total_reward)
        successes.append(bool(info.get("is_success", False)))
        lengths.append(steps)
        print(
            f"episode {ep:3d}: reward={total_reward:8.2f}  steps={steps:4d}  "
            f"termination={info.get('termination', 'timeout')}"
        )

    print("---")
    print(f"mean reward:  {np.mean(rewards):.2f} (+/- {np.std(rewards):.2f})")
    print(f"success rate: {np.mean(successes) * 100:.1f}%")
    print(f"mean length:  {np.mean(lengths):.1f} steps")

    if args.save_video and frames:
        import imageio

        fps = round(1.0 / (cfg.low_level.simulation_dt * cfg.low_level.control_decimation * cfg.high_level.decimation))
        imageio.mimsave(args.save_video, frames, fps=fps)
        print(f"saved video ({len(frames)} frames) to {args.save_video}")

    env.close()


if __name__ == "__main__":
    main()
