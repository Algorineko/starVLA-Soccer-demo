"""Frozen low-level locomotion controller for the SoccerDribble demo.

This wraps Unitree's official pretrained G1 walking policy (TorchScript,
trained with RL in Isaac Gym — see assets/UNITREE_RL_GYM_LICENSE), following
unitree_rl_gym's standard sim2sim PD-control loop, minus the interactive
viewer/keyboard handling. It is never trained or fine-tuned here: the only
trainable part of this demo is the high-level policy in soccer_env.py, which
calls into this class purely as a black box that turns a velocity command
into joint torques.
"""

from __future__ import annotations

import numpy as np
import torch

from config import LowLevelConfig

# Joint order must match motion.pt's training order, which is the actuator
# order declared in assets/g1_description/g1_12dof.xml.
_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
_BASE_FREE_JOINT_NAME = "floating_base_joint"


def _gravity_orientation(quaternion):
    """Projected gravity vector in the base frame, from base orientation quat (w,x,y,z)."""
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


class LowLevelPolicy:
    """Drives one G1 humanoid in a shared MuJoCo `model`/`data` toward a
    velocity command, using the frozen pretrained policy for PD joint targets.

    Usage per physics tick (dt = cfg.simulation_dt):
        low_level.physics_step(model, data, cmd)   # cmd = [vx, vy, yaw_rate]
    Call `reset()` whenever `data` has been reset (e.g. at episode start).
    """

    def __init__(self, cfg: LowLevelConfig, policy_path):
        self.cfg = cfg
        self.policy = torch.jit.load(str(policy_path))
        self.policy.eval()

        self._default_angles = np.array(cfg.default_angles, dtype=np.float32)
        self._kps = np.array(cfg.kps, dtype=np.float32)
        self._kds = np.array(cfg.kds, dtype=np.float32)
        self._cmd_scale = np.array(cfg.cmd_scale, dtype=np.float32)

        # Resolved lazily on first physics_step call, once `model` is known.
        self._qpos_adr = None
        self._qvel_adr = None
        self._base_qpos_adr = None
        self._base_qvel_adr = None

        self.reset()

    def reset(self):
        self._action = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self._target_dof_pos = self._default_angles.copy()
        self._counter = 0
        # motion.pt is a PolicyExporterLSTM: it carries hidden/cell state as
        # module buffers across calls, so a new episode must zero them or it
        # silently inherits memory from wherever the previous episode ended.
        if hasattr(self.policy, "hidden_state"):
            self.policy.hidden_state.zero_()
            self.policy.cell_state.zero_()

    def _resolve_addresses(self, model):
        if self._qpos_adr is not None:
            return
        self._qpos_adr = np.array([model.joint(name).qposadr[0] for name in _JOINT_NAMES])
        self._qvel_adr = np.array([model.joint(name).dofadr[0] for name in _JOINT_NAMES])
        base_joint = model.joint(_BASE_FREE_JOINT_NAME)
        self._base_qpos_adr = base_joint.qposadr[0]  # [pos(3), quat(4)]
        self._base_qvel_adr = base_joint.dofadr[0]  # [linvel(3), angvel(3)]

    def physics_step(self, model, data, cmd):
        """Advance the simulation by one physics dt. `cmd` = [vx, vy, yaw_rate]
        in the robot's target-velocity units (same convention as
        vendor/unitree_rl_gym's sim2sim cmd_init)."""
        import mujoco  # local import keeps this module importable without mujoco for unit tests

        self._resolve_addresses(model)

        qj = data.qpos[self._qpos_adr]
        dqj = data.qvel[self._qvel_adr]
        tau = (self._target_dof_pos - qj) * self._kps - dqj * self._kds
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        self._counter += 1
        if self._counter % self.cfg.control_decimation == 0:
            self._update_target(data, cmd)

    def _update_target(self, data, cmd):
        cfg = self.cfg
        n = cfg.num_actions

        qj = data.qpos[self._qpos_adr]
        dqj = data.qvel[self._qvel_adr]
        quat = data.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7]
        omega = data.qvel[self._base_qvel_adr + 3 : self._base_qvel_adr + 6]

        qj_scaled = (qj - self._default_angles) * cfg.dof_pos_scale
        dqj_scaled = dqj * cfg.dof_vel_scale
        gravity = _gravity_orientation(quat)
        omega_scaled = omega * cfg.ang_vel_scale

        t = self._counter * cfg.simulation_dt
        phase = (t % cfg.gait_period) / cfg.gait_period
        sin_phase, cos_phase = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

        obs = np.zeros(cfg.num_obs, dtype=np.float32)
        obs[0:3] = omega_scaled
        obs[3:6] = gravity
        obs[6:9] = np.asarray(cmd, dtype=np.float32) * self._cmd_scale
        obs[9 : 9 + n] = qj_scaled
        obs[9 + n : 9 + 2 * n] = dqj_scaled
        obs[9 + 2 * n : 9 + 3 * n] = self._action
        obs[9 + 3 * n : 9 + 3 * n + 2] = [sin_phase, cos_phase]

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(obs_tensor).detach().numpy().squeeze()
        self._action = action
        self._target_dof_pos = action * cfg.action_scale + self._default_angles
