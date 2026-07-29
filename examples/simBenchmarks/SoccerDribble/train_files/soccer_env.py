"""Gymnasium environment for the SoccerDribble demo.

Wraps the frozen low-level G1 walking policy (low_level_policy.py) inside a
MuJoCo simulation and exposes a *high-level* decision interface: the "brain"
this demo trains observes ball/target geometry (in the robot's own body
frame, matching the convention the low-level policy uses for velocity
commands) and outputs a velocity-command increment. See train_files/config.py
for the exact observation/action layout and reward shaping.
"""

from __future__ import annotations

import numpy as np
import mujoco
from gymnasium import Env, spaces

from config import SoccerDribbleConfig, SCENE_XML, LOW_LEVEL_POLICY_PATH
from low_level_policy import LowLevelPolicy

_OBS_DIM = 13
_ACTION_DIM = 3


def _quat_to_yaw(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _yaw_to_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _world_to_body(vec_xy, yaw):
    """Rotate a world-frame xy vector into the robot's body frame (yaw-only)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * vec_xy[0] + s * vec_xy[1], -s * vec_xy[0] + c * vec_xy[1]])


class SoccerDribbleEnv(Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: SoccerDribbleConfig | None = None, render_mode: str | None = None):
        self.cfg = cfg or SoccerDribbleConfig()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.low_level = LowLevelPolicy(self.cfg.low_level, LOW_LEVEL_POLICY_PATH)

        self._base_qpos_adr = self.model.joint("floating_base_joint").qposadr[0]
        self._base_qvel_adr = self.model.joint("floating_base_joint").dofadr[0]
        self._ball_qpos_adr = self.model.joint("ball_joint").qposadr[0]
        self._ball_qvel_adr = self.model.joint("ball_joint").dofadr[0]
        self._target_mocap_id = self.model.body("target_marker").mocapid[0]

        self._physics_steps_per_hl_step = (
            self.cfg.low_level.control_decimation * self.cfg.high_level.decimation
        )
        self._max_hl_steps = int(
            self.cfg.high_level.max_episode_seconds
            / (self.cfg.low_level.simulation_dt * self._physics_steps_per_hl_step)
        )

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(_ACTION_DIM,), dtype=np.float32)

        self._renderer = None
        self._np_random_local = np.random.default_rng()

        self._cmd = np.zeros(3, dtype=np.float32)
        self._hl_step_count = 0
        self._prev_dist_robot_ball = 0.0
        self._prev_dist_ball_target = 0.0

    # -- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random_local = np.random.default_rng(seed)
        rng = self._np_random_local

        mujoco.mj_resetData(self.model, self.data)
        self.low_level.reset()
        self._cmd[:] = 0.0
        self._hl_step_count = 0

        robot_xy, ball_xy, target_xy = self._sample_layout(rng)
        robot_yaw = rng.uniform(-np.pi, np.pi)

        self.data.qpos[self._base_qpos_adr : self._base_qpos_adr + 2] = robot_xy
        self.data.qpos[self._base_qpos_adr + 2] = 0.793  # standing height, from g1.yaml qpos0
        self.data.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7] = _yaw_to_quat(robot_yaw)

        self.data.qpos[self._ball_qpos_adr : self._ball_qpos_adr + 2] = ball_xy
        self.data.qpos[self._ball_qpos_adr + 2] = self.cfg.field_cfg.ball_radius
        self.data.qpos[self._ball_qpos_adr + 3 : self._ball_qpos_adr + 7] = [1, 0, 0, 0]

        self.data.mocap_pos[self._target_mocap_id] = [target_xy[0], target_xy[1], 0.002]

        mujoco.mj_forward(self.model, self.data)

        self._prev_dist_robot_ball = np.linalg.norm(ball_xy - robot_xy)
        self._prev_dist_ball_target = np.linalg.norm(target_xy - ball_xy)
        self._prev_alignment = self._compute_alignment(robot_xy, ball_xy, target_xy)

        obs = self._compute_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        delta_cmd = action * np.asarray(self.cfg.high_level.delta_cmd_limit, dtype=np.float32)
        self._cmd = np.clip(
            self._cmd + delta_cmd,
            -np.asarray(self.cfg.high_level.cmd_limit, dtype=np.float32),
            np.asarray(self.cfg.high_level.cmd_limit, dtype=np.float32),
        )

        for _ in range(self._physics_steps_per_hl_step):
            self.low_level.physics_step(self.model, self.data, self._cmd)

        self._hl_step_count += 1

        robot_xy = self.data.qpos[self._base_qpos_adr : self._base_qpos_adr + 2].copy()
        robot_z = self.data.qpos[self._base_qpos_adr + 2]
        ball_xy = self.data.qpos[self._ball_qpos_adr : self._ball_qpos_adr + 2].copy()
        target_xy = self.data.mocap_pos[self._target_mocap_id][:2].copy()
        ball_vel_xy = self.data.qvel[self._ball_qvel_adr : self._ball_qvel_adr + 2]

        dist_robot_ball = np.linalg.norm(ball_xy - robot_xy)
        dist_ball_target = np.linalg.norm(target_xy - ball_xy)

        reward, terminated, info = self._compute_reward(
            dist_robot_ball, dist_ball_target, ball_vel_xy, delta_cmd, robot_z, ball_xy, robot_xy, target_xy
        )
        self._prev_dist_robot_ball = dist_robot_ball
        self._prev_dist_ball_target = dist_ball_target

        truncated = self._hl_step_count >= self._max_hl_steps
        obs = self._compute_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="sideline")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- helpers ------------------------------------------------------------

    def _sample_layout(self, rng):
        fcfg = self.cfg.field_cfg
        lo_x, hi_x = -fcfg.half_x + fcfg.margin, fcfg.half_x - fcfg.margin
        lo_y, hi_y = -fcfg.half_y + fcfg.margin, fcfg.half_y - fcfg.margin

        robot_xy = np.array([rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)])

        for _ in range(50):
            angle = rng.uniform(-np.pi, np.pi)
            dist = rng.uniform(fcfg.min_robot_ball_dist, fcfg.min_robot_ball_dist + 1.0)
            ball_xy = robot_xy + dist * np.array([np.cos(angle), np.sin(angle)])
            if lo_x <= ball_xy[0] <= hi_x and lo_y <= ball_xy[1] <= hi_y:
                break
        else:
            ball_xy = np.clip(ball_xy, [lo_x, lo_y], [hi_x, hi_y])

        for _ in range(50):
            angle = rng.uniform(-np.pi, np.pi)
            dist = rng.uniform(fcfg.min_ball_target_dist, fcfg.min_ball_target_dist + 1.5)
            target_xy = ball_xy + dist * np.array([np.cos(angle), np.sin(angle)])
            if lo_x <= target_xy[0] <= hi_x and lo_y <= target_xy[1] <= hi_y:
                break
        else:
            target_xy = np.clip(target_xy, [lo_x, lo_y], [hi_x, hi_y])

        return robot_xy, ball_xy, target_xy

    def _compute_obs(self):
        robot_xy = self.data.qpos[self._base_qpos_adr : self._base_qpos_adr + 2]
        robot_quat = self.data.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7]
        robot_yaw = _quat_to_yaw(robot_quat)
        robot_linvel_world = self.data.qvel[self._base_qvel_adr : self._base_qvel_adr + 2]

        ball_xy = self.data.qpos[self._ball_qpos_adr : self._ball_qpos_adr + 2]
        target_xy = self.data.mocap_pos[self._target_mocap_id][:2]

        to_ball_body = _world_to_body(ball_xy - robot_xy, robot_yaw)
        to_target_body = _world_to_body(target_xy - robot_xy, robot_yaw)
        linvel_body = _world_to_body(robot_linvel_world, robot_yaw)

        obs = np.concatenate(
            [
                to_ball_body,
                to_target_body,
                [np.linalg.norm(ball_xy - robot_xy)],
                [np.linalg.norm(target_xy - ball_xy)],
                linvel_body,
                [np.sin(robot_yaw), np.cos(robot_yaw)],
                self._cmd,
            ]
        ).astype(np.float32)
        return obs

    def _compute_alignment(self, robot_xy, ball_xy, target_xy):
        """cos(robot->ball, ball->target): 1 = robot is directly behind the
        ball on the line to the target, -1 = robot is beyond the ball
        already. Returns 0.0 (neutral) if either vector is degenerate."""
        robot_to_ball = ball_xy - robot_xy
        ball_to_target = target_xy - ball_xy
        rb_norm = np.linalg.norm(robot_to_ball)
        bt_norm = np.linalg.norm(ball_to_target)
        if rb_norm > 1e-6 and bt_norm > 1e-6:
            return float(np.dot(robot_to_ball, ball_to_target) / (rb_norm * bt_norm))
        return 0.0

    def _compute_reward(
        self, dist_robot_ball, dist_ball_target, ball_vel_xy, delta_cmd, robot_z, ball_xy, robot_xy, target_xy
    ):
        rcfg = self.cfg.reward
        hcfg = self.cfg.high_level
        fcfg = self.cfg.field_cfg
        info = {}

        reward = 0.0
        if dist_robot_ball > rcfg.dribble_range:
            reward += rcfg.approach_weight * (self._prev_dist_robot_ball - dist_robot_ball)
            # Reward *improving* alignment (this step's cos vs last step's),
            # not the absolute cos value: an absolute-value reward pays out
            # just as well for standing still at a lucky spawn as for
            # actually moving into position, which taught the policy to
            # stand still (observed: eval episodes with high reward and
            # dist_ball_target frozen for the whole episode). A delta-based
            # reward pays only for genuinely improving the approach angle.
            alignment = self._compute_alignment(robot_xy, ball_xy, target_xy)
            reward += rcfg.alignment_weight * (alignment - self._prev_alignment)
            self._prev_alignment = alignment
        else:
            ball_speed = np.linalg.norm(ball_vel_xy)
            reward -= rcfg.dribble_speed_penalty * max(0.0, ball_speed - hcfg.dribble_speed_soft_limit)

        # Ball-toward-target progress is rewarded unconditionally (not just
        # once the robot has formally entered dribble_range). Gating it
        # behind that threshold made "approach the ball, then discover the
        # right push" a rare compound event for random exploration to ever
        # stumble into — a stray nudge while still approaching produced zero
        # signal, so there was nothing to reinforce. Since Δdist_ball_target
        # is ~0 whenever the robot isn't close enough to actually move the
        # ball, making the term unconditional adds no exploitable reward
        # (nothing to gain by not touching the ball) — it only widens *when*
        # a real push gets credited.
        reward += rcfg.dribble_weight * (self._prev_dist_ball_target - dist_ball_target)

        reward -= rcfg.cmd_smooth_penalty * float(np.dot(delta_cmd, delta_cmd))
        reward += rcfg.alive_bonus

        terminated = False
        if robot_z < hcfg.fall_height_threshold:
            reward -= rcfg.fall_penalty
            terminated = True
            info["termination"] = "fall"
        elif abs(ball_xy[0]) > fcfg.half_x or abs(ball_xy[1]) > fcfg.half_y:
            reward -= rcfg.fall_penalty
            terminated = True
            info["termination"] = "ball_out_of_bounds"
        elif dist_ball_target < fcfg.target_radius:
            reward += rcfg.success_bonus
            terminated = True
            info["termination"] = "success"

        # Always set explicitly (not just on success): SB3's Monitor/rollout
        # success_rate logger averages this key across whatever episodes
        # happen to *have* it, so leaving it unset on failure/timeout makes
        # the metric silently average over successes only (always reads 1.0).
        info["is_success"] = terminated and info.get("termination") == "success"

        return float(reward), terminated, info
