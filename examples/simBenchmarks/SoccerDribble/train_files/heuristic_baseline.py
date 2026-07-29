"""Sanity-check the SoccerDribbleEnv reward/observation design with a simple
scripted P-controller, before spending any compute on PPO. It never learns
anything; it just proves the environment's dynamics and reward shaping are
directionally sane (heading to the ball, then to the target, produces
increasing reward and a non-zero success rate).

Usage:
    python heuristic_baseline.py --episodes 30
"""

from __future__ import annotations

import argparse

import numpy as np

from config import SoccerDribbleConfig
from soccer_env import SoccerDribbleEnv


def heuristic_action(obs, cfg: SoccerDribbleConfig):
    """Scripted dribbling controller (proportional speed, distance-gated
    aim point). Heading straight at the ball only shoves it sideways/out of
    bounds, so while far away the robot first aims for a point slightly
    *behind* the ball on the ball-target line (i.e. lines up its approach),
    then once close enough switches to pushing straight through the ball
    along that same line, which is what actually drives the ball toward the
    target instead of just circling it."""
    to_ball = obs[0:2]
    to_target = obs[2:4]
    dist_robot_ball = obs[4]
    prev_cmd = obs[10:13]

    ball_to_target = to_target - to_ball
    bt_dist = np.linalg.norm(ball_to_target)
    ball_to_target_dir = ball_to_target / bt_dist if bt_dist > 1e-6 else np.array([1.0, 0.0])

    dribbling = dist_robot_ball <= cfg.reward.dribble_range
    if dribbling:
        target_dir = ball_to_target_dir
        dist = dist_robot_ball  # keep closing speed proportional to ball distance, not target distance
    else:
        approach_offset = cfg.field_cfg.ball_radius + 0.25
        aim_point = to_ball - ball_to_target_dir * approach_offset
        dist = np.linalg.norm(aim_point)
        target_dir = aim_point / dist if dist > 1e-6 else ball_to_target_dir

    max_speed = cfg.high_level.dribble_speed_soft_limit if dribbling else 0.5
    desired_speed = float(np.clip(0.6 * dist, 0.1, max_speed))
    desired_vx, desired_vy = desired_speed * target_dir
    desired_cmd = np.array([desired_vx, desired_vy, 0.0], dtype=np.float32)

    delta = desired_cmd - prev_cmd
    delta_limit = np.asarray(cfg.high_level.delta_cmd_limit, dtype=np.float32)
    action = np.clip(delta / delta_limit, -1.0, 1.0)
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = SoccerDribbleConfig()
    env = SoccerDribbleEnv(cfg)

    rewards, successes, lengths = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action = heuristic_action(obs, cfg)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
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


if __name__ == "__main__":
    main()
