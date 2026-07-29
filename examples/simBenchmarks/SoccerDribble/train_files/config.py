"""Central config for the SoccerDribble demo: field/task layout, low-level
policy I/O (mirrors vendor/unitree_rl_gym's g1.yaml), high-level action/reward
shaping, and PPO hyperparameters. Keeping everything here avoids scattering
magic numbers across soccer_env.py / low_level_policy.py / train_ppo.py.
"""

from dataclasses import dataclass, field
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SCENE_XML = ASSETS_DIR / "soccer_field.xml"
LOW_LEVEL_POLICY_PATH = ASSETS_DIR / "policy" / "motion.pt"


@dataclass
class FieldConfig:
    half_x: float = 4.0  # matches <geom name="field" size="4 3 0.1"> in soccer_field.xml
    half_y: float = 3.0
    target_radius: float = 0.5
    ball_radius: float = 0.11
    # Reset randomization ranges (uniform sampling within the field, minus a margin).
    margin: float = 0.6
    min_robot_ball_dist: float = 0.8
    min_ball_target_dist: float = 1.5


@dataclass
class LowLevelConfig:
    """Mirrors vendor/unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml — the
    frozen G1 walking policy is not retrained, so these constants must match
    what motion.pt was trained with."""

    num_actions: int = 12
    num_obs: int = 47
    simulation_dt: float = 0.002
    control_decimation: int = 10  # low-level policy runs at 1/(dt*decim) = 50Hz
    kps: tuple = (100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40)
    kds: tuple = (2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2)
    default_angles: tuple = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0)
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    action_scale: float = 0.25
    cmd_scale: tuple = (2.0, 2.0, 0.25)
    gait_period: float = 0.8


@dataclass
class HighLevelConfig:
    """The 'decision brain': every `decimation` low-level steps (10 * 50Hz-step
    = 5Hz), it observes robot/ball/target geometry and outputs a velocity
    command *increment*, integrated into cmd = clip(cmd + delta_cmd)."""

    decimation: int = 10  # high-level acts every 10 low-level steps -> 5Hz
    delta_cmd_limit: tuple = (0.2, 0.1, 0.1)  # max |Δvx|, |Δvy|, |Δyaw_rate| per decision
    cmd_limit: tuple = (1.0, 0.5, 0.5)  # clip range for the integrated cmd
    max_episode_seconds: float = 20.0
    fall_height_threshold: float = 0.5  # pelvis z below this ends episode as a fall
    dribble_speed_soft_limit: float = 0.6  # penalize ball speed above this while dribbling


@dataclass
class RewardConfig:
    approach_weight: float = 1.0  # -Δ(dist_robot_ball) while far from the ball
    # Rewards approaching from the far side of the ball (i.e. lining up the
    # ball-target line) instead of straight-line to the ball's center, which
    # just knocks the ball sideways/away rather than dribbling it anywhere.
    # cos(robot->ball, ball->target): 1 = robot is directly behind the ball
    # on the line to the target, -1 = robot is beyond the ball already.
    alignment_weight: float = 0.05
    dribble_weight: float = 1.0  # -Δ(dist_ball_target) once close to the ball
    dribble_speed_penalty: float = 0.5
    cmd_smooth_penalty: float = 0.05  # -||delta_cmd||^2
    # Kept deliberately tiny: at 0.05/step x ~100 steps/episode, this alone
    # was worth ~5 reward — comparable to or larger than what the policy
    # could earn by actually approaching/dribbling the ball, which taught it
    # to just stand still near a favorably-aligned spawn instead of moving
    # (observed: eval episodes with high reward and dist_ball_target frozen
    # for the entire episode). At 0.005/step the max full-episode payout is
    # ~0.5, well below approach/alignment/dribble reward accumulated over a
    # real attempt, so standing still is no longer competitive.
    alive_bonus: float = 0.005  # per high-level step
    fall_penalty: float = 10.0
    success_bonus: float = 20.0
    # Distance (m) from robot to ball below which reward switches Approach -> Dribble.
    dribble_range: float = 0.5


@dataclass
class PPOConfig:
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    n_steps: int = 512  # per env, per rollout
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.001
    net_arch: tuple = (64, 64)
    seed: int = 0


@dataclass
class SoccerDribbleConfig:
    field_cfg: FieldConfig = field(default_factory=FieldConfig)
    low_level: LowLevelConfig = field(default_factory=LowLevelConfig)
    high_level: HighLevelConfig = field(default_factory=HighLevelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
