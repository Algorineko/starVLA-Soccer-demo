# SoccerDribble: A Hierarchical Humanoid Soccer Dribbling Demo

[中文](README.zh.md)

A minimal demo of hierarchical control for humanoid robotics: a **frozen,
pretrained low-level walking policy** drives a Unitree G1 humanoid, and a
**small MLP "decision brain" trained with RL (PPO)** sits on top of it,
outputting velocity commands that make the robot walk to a soccer ball and
dribble it toward a randomly placed target circle, within a small (8m x 6m)
field.

This follows the two-level split common in hierarchical humanoid soccer
control (perception → high-level "coach" policy → low-level "motion" policy):
only the high-level policy is trained here; the low level is treated as an
external black box.

## What this is / isn't

- **Is**: a standalone, from-scratch MuJoCo + Gymnasium + stable-baselines3
  demo. It follows starVLA's `examples/simBenchmarks/<Bench>/{train_files,eval_files}`
  layout convention, but is otherwise **independent of starVLA's model
  registry, training loop, and deployment server** — it doesn't touch
  `starVLA/model/framework/` or `starVLA/training/`. RL (reward, rollouts,
  PPO updates) has no equivalent in starVLA's supervised imitation-learning
  training loop, so this demo runs its own scripts end to end.
- **Isn't**: production-grade, robustness-tested, or sim-to-real ready.
  Trained long enough (3M steps) it does reach a high (84%) success rate —
  see Results — but there's no domain randomization (fixed physics
  parameters, noiseless observations, no action delay), so this is still a
  "does the hierarchy work" demo, not one that's been shown to survive
  physics/sensor mismatches. Only the single-robot / single-ball /
  random-target task is implemented; natural follow-ups would be 2-robot
  competitive dribbling, then a multi-robot tactics layer, but those are out
  of scope for this demo.

## Results

The reference run (`train_files/runs/ppo_v1`) is the same PPO policy
(`MlpPolicy`, `net_arch=[64,64]`, hyperparameters unchanged from
`train_files/config.py`) trained progressively longer across three launches
using `train_ppo.py --resume`, all logging to the same `runs/ppo_v1/`
directory:

| Launch | Steps (cumulative) | New steps | Wall-clock | Notes |
|---|---|---|---|---|
| 1 (fresh) | 0 → 500,000 | 500,000 | 53.0 min | `checkpoints/rl_model_500000_steps.zip` |
| 2 (`--resume`) | 500,000 → 1,003,808 | ~503,808 | 51.0 min | `final_model.zip` (1M snapshot) |
| 3 (`--resume`) | 1,003,808 → 3,002,656 | ~1,998,848 | 181.1 min | current `final_model.zip` / `best_model.zip` |

All runs used CPU-only training with `n_envs=8` (~160-220 steps/s); total
wall-clock across all three launches is **~4.75 hours** for 3M steps.

At **3M steps**, `train_files/runs/ppo_v1/best_model/best_model.zip` reaches
an **84% success rate** over 100 held-out episodes (ball reaches the target
circle), mean reward 18.46, mean episode length 71.8 steps — up from 8% at
1M steps (the original reference number, still visible in
`train_files/runs/ppo_v1/train.log`'s early history). The jump confirms the
1M-step policy was undertrained, not capacity-limited: `heuristic_baseline.py`
still lands around 5%, so the trained policy now clears it by a wide margin
rather than a narrow one.

Convergence signals from the full 0→3M curve (`train_files/runs/ppo_v1/PPO_1`
TensorBoard logs, plotted below) support "trained to convergence, not just
longer": `train/entropy_loss` rises from -4.25 to ≈0 and `train/std` (action
std) falls from 0.99 to ≈0.24, meaning the policy has converged to a
near-deterministic behavior; `train/explained_variance` stabilizes around
0.75-0.85; `rollout/ep_len_mean` drops from ~90 to ~60 steps, meaning
successful episodes are also completing faster, not just more often.

![training curves](train_files/runs/ppo_v1/training_curves.png)

*(Regenerate with `python train_files/plot_training_curves.py --logdir
train_files/runs/ppo_v1` after any further training — it reads the
TensorBoard event files, which stay continuous across `--resume` launches,
unlike the per-run `monitor/*.csv` and `eval/evaluations.npz`, which restart
from empty on every new `train_ppo.py` invocation.)*

Two reward-hacking regressions surfaced and were fixed during training, both
worth knowing about if you extend the reward function: (1) an absolute-value
alignment term let the policy farm reward just by *spawning* in a favorable
position, rather than moving into one — fixed by rewarding the change in
alignment per step instead of its raw value; (2) gating the ball-progress
term behind a `dribble_range` threshold meant a stray touch while still
approaching earned zero credit, making "approach, then discover the push"
too rare an event for PPO to reinforce — fixed by making that term
unconditional (it only widens *when* progress gets credited; it can't be
farmed by not touching the ball, since Δdist_ball_target is ~0 while the
ball is untouched).

A recorded successful rollout from the 3M-step model, captured the same way
as the `--save-video` example below:

<video src="eval_files/dribble_success_demo.mp4" controls width="480"></video>

## Architecture

```
Coach Policy (trained)          Motion Policy (frozen, pretrained)
5 Hz, MLP [64, 64]               50 Hz, LSTM (Unitree's motion.pt)
obs: ball/target geometry   -->  obs: joint state + velocity cmd + gait phase
     in robot body frame         action: 12-DOF joint position targets -> PD
action: Δ(vx, vy, yaw_rate)          |
     (integrated into cmd)          v
                                MuJoCo physics (500 Hz)
```

- **Low level** (`train_files/low_level_policy.py`): wraps Unitree's official
  pretrained G1 walking policy (`assets/policy/motion.pt`, TorchScript LSTM,
  trained with RL in Isaac Gym — see `assets/UNITREE_RL_GYM_LICENSE`,
  BSD-3). Never retrained; the wrapper reproduces `unitree_rl_gym`'s
  standard sim2sim PD-control loop (angular velocity / gravity projection /
  joint state / gait phase observation, minus interactive viewer/keyboard
  handling).
- **High level** (`train_files/soccer_env.py`, trained by `train_files/train_ppo.py`):
  a `gymnasium.Env` that steps the low-level policy internally and exposes a
  13-dim observation (ball/target position in the robot's body frame,
  distances, robot velocity/heading, previous command) and a 3-dim bounded
  action (`Δcmd`), with reward shaped in two phases — close the distance to
  the ball, then push the ball toward the target while capping ball speed so
  it doesn't get kicked away.
- **Field**: `assets/soccer_field.xml`, 8m x 6m (undersized relative to a
  real pitch, on purpose, so episodes stay short); ball physics match
  FIFA-ish scale (radius 0.11m, mass 0.43kg). Target circle is a mocap body
  relocated randomly every `reset()`.

## Setup (venv, not conda)

Requires Python 3.11 (mujoco/torch prebuilt wheels aren't reliably available
for 3.12+ yet on all platforms).

```bash
cd train_files
./setup_env.sh          # creates train_files/.venv
source .venv/bin/activate
```

## Usage

```bash
cd train_files
source .venv/bin/activate

# 1. Sanity-check the env/reward with a scripted (non-learned) controller.
python heuristic_baseline.py --episodes 20

# 2. Train the high-level PPO policy.
python train_ppo.py --timesteps 2000000 --n-envs 8 --logdir runs/ppo_v1

# 2b. Optionally keep training past --timesteps by resuming from the latest
# checkpoint/final model -- this is how the checked-in runs/ppo_v1 reference
# run went from 1M to 3M steps (see Results):
python train_ppo.py --timesteps 3000000 --n-envs 8 --logdir runs/ppo_v1 \
    --resume runs/ppo_v1/final_model.zip

# 3. Evaluate a checkpoint (use more episodes than the default 20 for a
# lower-variance success-rate estimate -- the Results section above uses 100).
cd ../eval_files
python eval_soccer.py --model ../train_files/runs/ppo_v1/best_model/best_model.zip --episodes 100
python eval_soccer.py --model ../train_files/runs/ppo_v1/best_model/best_model.zip --episodes 1 --save-video rollout.mp4

# equivalent shortcut for the first eval command above (run from eval_files/;
# model path is relative to train_files/):
./run_eval.sh runs/ppo_v1/best_model/best_model.zip 20
```

Monitor training with TensorBoard: `tensorboard --logdir train_files/runs/ppo_v1`.
For a static image instead (e.g. to embed in docs), see
`train_files/plot_training_curves.py`, used to generate the plot in Results.

## Provenance

`assets/g1_description/` (MJCF + meshes), `assets/policy/motion.pt`, and
`assets/UNITREE_RL_GYM_LICENSE` are copied from Unitree Robotics' open-source
[`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym)
(BSD-3-Clause). Only the files needed for the 12-DOF G1 sim2sim walking
policy are included (not the full arm/hand model or the Isaac Gym training
code).
