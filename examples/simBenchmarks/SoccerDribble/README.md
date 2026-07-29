# SoccerDribble: A Hierarchical Humanoid Soccer Dribbling Demo

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
- **Isn't**: production-grade or highly tuned. It's a "does the hierarchy
  work at all" demo — expect a modest, not high, success rate. Only the
  single-robot / single-ball / random-target task is implemented; natural
  follow-ups would be 2-robot competitive dribbling, then a multi-robot
  tactics layer, but those are out of scope for this demo.

## Results

The reference run (`train_files/runs/ppo_v1`, 1M PPO steps, `net_arch=[64,64]`)
reaches an **8% success rate** over 50 held-out episodes (ball reaches the
target circle), vs. 5% for the scripted `heuristic_baseline.py`. This
confirms the hierarchy works end-to-end — the trained policy does learn to
approach the ball and, in a minority of episodes, push it toward the target
— but "approach the ball" is learned far more reliably than "dribble it
accurately," which remains the bottleneck skill.

The reference run trained for 1,000,000 timesteps in two 500k-step passes on
CPU (`n_envs=8`, ~160 steps/s, ~52 min per pass — ~104 min total; see
`train_files/runs/ppo_v1/train.log`). The in-training `EvalCallback` (5
episodes every 20k steps) is noisy at this sample size — success rate swings
between 0% and 20% checkpoint-to-checkpoint over the last 100k steps — so
don't read any single in-training eval as the headline number. The 8% vs.
5% figures above come from a separate, larger 50-episode held-out evaluation
via `eval_soccer.py`, which is the number to trust.

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

A recorded successful rollout from this run is at
`eval_files/dribble_success_demo.mp4`, captured the same way as the
`--save-video` example below.

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

# 3. Evaluate a checkpoint.
cd ../eval_files
python eval_soccer.py --model ../train_files/runs/ppo_v1/best_model/best_model.zip --episodes 20
python eval_soccer.py --model ../train_files/runs/ppo_v1/best_model/best_model.zip --episodes 1 --save-video rollout.mp4

# equivalent shortcut for the first eval command above (run from eval_files/;
# model path is relative to train_files/):
./run_eval.sh runs/ppo_v1/best_model/best_model.zip 20
```

Monitor training with TensorBoard: `tensorboard --logdir train_files/runs/ppo_v1`.

## Provenance

`assets/g1_description/` (MJCF + meshes), `assets/policy/motion.pt`, and
`assets/UNITREE_RL_GYM_LICENSE` are copied from Unitree Robotics' open-source
[`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym)
(BSD-3-Clause). Only the files needed for the 12-DOF G1 sim2sim walking
policy are included (not the full arm/hand model or the Isaac Gym training
code).
