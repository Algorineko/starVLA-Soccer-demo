"""Train the SoccerDribble high-level "brain" with PPO (stable-baselines3).

The low-level G1 walking controller is frozen (see low_level_policy.py); only
this small MLP policy is trained, to output velocity-command increments.
Run `heuristic_baseline.py` first to sanity-check the env/reward before
spending compute here.

Usage:
    python train_ppo.py --timesteps 2000000 --n-envs 8 --logdir runs/ppo_v1

    # Resume an interrupted run from its latest checkpoint (CheckpointCallback
    # writes runs/<logdir>/checkpoints/rl_model_<N>_steps.zip periodically):
    python train_ppo.py --timesteps 2000000 --n-envs 8 --logdir runs/ppo_v1 \\
        --resume runs/ppo_v1/checkpoints/rl_model_50000_steps.zip
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from config import SoccerDribbleConfig
from soccer_env import SoccerDribbleEnv


def _format_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


class WallClockCallback(BaseCallback):
    """Injects a human-readable clock time + elapsed duration into every
    metrics table SB3 prints (and every CSV/tensorboard dump), so a glance
    at the log tells you when a block was printed and how long the run has
    been going — SB3's own time/time_elapsed is raw seconds, easy to
    glaze over when eyeballing a long log."""

    def _on_training_start(self) -> None:
        self._wall_start = time.time()

    def _on_step(self) -> bool:
        self.logger.record("time/wall_clock", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), exclude="tensorboard")
        self.logger.record("time/elapsed_hms", _format_hms(time.time() - self._wall_start), exclude="tensorboard")
        return True


class TimestampedEvalCallback(EvalCallback):
    """EvalCallback prints "Eval num_timesteps=..." / "Success rate: ..."
    directly via `print()`, bypassing the logger table WallClockCallback
    annotates — so those lines need their own timestamp prefix to stay
    correlatable with the rest of the log."""

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] running eval...")
        return super()._on_step()


def make_env_fn(cfg: SoccerDribbleConfig, seed: int, monitor_path: str | None = None):
    """`monitor_path`, if given, makes SB3 log per-episode reward/length/
    success to `<monitor_path>.monitor.csv` (one file per env) and is what
    populates rollout/ep_rew_mean, ep_len_mean, success_rate in the training
    printout — without Monitor, SB3 only sees raw per-step rewards, not
    episode-level stats."""

    def _init():
        env = SoccerDribbleEnv(cfg)
        if monitor_path is not None:
            env = Monitor(env, filename=monitor_path, info_keywords=("is_success",))
        env.reset(seed=seed)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None, help="overrides cfg.ppo.total_timesteps")
    parser.add_argument("--n-envs", type=int, default=None, help="overrides cfg.ppo.n_envs")
    parser.add_argument("--logdir", type=str, default="runs/ppo_soccer_dribble")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--single-process", action="store_true", help="use DummyVecEnv (debugging)")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint .zip to resume from")
    args = parser.parse_args()

    cfg = SoccerDribbleConfig()
    n_envs = args.n_envs or cfg.ppo.n_envs
    total_timesteps = args.timesteps or cfg.ppo.total_timesteps

    monitor_dir = os.path.join(args.logdir, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)
    env_fns = [
        make_env_fn(cfg, args.seed + i, monitor_path=os.path.join(monitor_dir, f"train_{i}"))
        for i in range(n_envs)
    ]
    vec_env_cls = DummyVecEnv if args.single_process else SubprocVecEnv
    vec_env = vec_env_cls(env_fns)

    if args.resume:
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=args.logdir)
        print(f"Resumed from {args.resume} at {model.num_timesteps} timesteps")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=cfg.ppo.learning_rate,
            n_steps=cfg.ppo.n_steps,
            batch_size=cfg.ppo.batch_size,
            n_epochs=cfg.ppo.n_epochs,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            clip_range=cfg.ppo.clip_range,
            ent_coef=cfg.ppo.ent_coef,
            policy_kwargs={"net_arch": list(cfg.ppo.net_arch)},
            seed=cfg.ppo.seed,
            verbose=1,
            tensorboard_log=args.logdir,
        )

    eval_env = DummyVecEnv(
        [make_env_fn(cfg, args.seed + 10_000, monitor_path=os.path.join(monitor_dir, "eval"))]
    )
    callbacks = [
        WallClockCallback(),
        CheckpointCallback(save_freq=max(1, 50_000 // n_envs), save_path=f"{args.logdir}/checkpoints"),
        TimestampedEvalCallback(
            eval_env,
            best_model_save_path=f"{args.logdir}/best_model",
            log_path=f"{args.logdir}/eval",
            eval_freq=max(1, 20_000 // n_envs),
        ),
    ]

    # SB3 treats `total_timesteps` as *additional* steps when
    # reset_num_timesteps=False, so on resume we pass the remainder to reach
    # `total_timesteps` overall rather than training total_timesteps more.
    remaining = total_timesteps - model.num_timesteps if args.resume else total_timesteps
    remaining = max(0, remaining)

    start = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not args.resume,
    )
    elapsed = time.time() - start
    model.save(f"{args.logdir}/final_model")
    print(f"Saved final model to {args.logdir}/final_model.zip")
    print(f"Trained {remaining} steps in {elapsed / 60:.1f} min ({remaining / max(elapsed, 1e-9):.0f} steps/s)")


if __name__ == "__main__":
    main()
