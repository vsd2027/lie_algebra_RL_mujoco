"""
franka_orientation.py — EE Orientation Control GoalEnv

Uses MuJoCo Menagerie Franka Panda (loaded via robot_descriptions).
Task: rotate the end-effector to a goal orientation.
Action: orientation delta in chosen representation → Jacobian IK → joint commands.

GoalEnv compatible with SB3's HerReplayBuffer.

Key facts about Menagerie Panda (from XML):
  - Bodies: link0..link7, hand, left_finger, right_finger
  - NO sites — must use data.xmat/xpos on "hand" body
  - Joints: joint1..joint7 (arm), finger_joint1/2 (gripper)
  - Actuators: actuator1..actuator7 (arm), actuator8 (gripper tendon)
  - Home keyframe: qpos=[0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]
  - nq=9, nv=9, nu=8

Install:
    pip install robot_descriptions mujoco gymnasium stable-baselines3
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rotations import (
    REPR_DIM, rotmat_to_repr, repr_to_rotation, geodesic_distance_batch
)


def _load_panda_model():
    """Load Franka Panda from MuJoCo Menagerie via robot_descriptions."""
    from robot_descriptions.loaders.mujoco import load_robot_description
    return load_robot_description("panda_mj_description")


class FrankaOrientationGoalEnv(gym.Env):
    """
    End-effector orientation control with Franka Panda.

    Observation dict:
        'observation':    joint_pos(7) + joint_vel(7) = 14D
        'achieved_goal':  current EE orientation in action_repr
        'desired_goal':   target EE orientation in action_repr

    Action:
        Orientation delta in action_repr → damped Jacobian IK → joint position targets.

    Reward:
        Sparse: 0 if geodesic distance < threshold, -1 otherwise.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        action_repr: str = 'rotvec',
        max_angle: float = 0.1 * np.pi,
        threshold: float = 0.15,
        max_steps: int = 100,
        ik_damping: float = 0.05,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.action_repr = action_repr
        self.max_angle = max_angle
        self.threshold = threshold
        self.max_steps = max_steps
        self.ik_damping = ik_damping
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        # Load model
        self.model = _load_panda_model()
        self.data = mujoco.MjData(self.model)

        # ── Robot structure ──────────────────────────────────────────────
        self.n_joints = 7
        self.joint_names = [f'joint{i+1}' for i in range(self.n_joints)]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in self.joint_names]

        # EE = "hand" body (Menagerie Panda has NO sites)
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        assert self.hand_body_id >= 0, "Body 'hand' not found in Menagerie Panda"

        # Arm actuators: actuator1..actuator7 (indices 0..6)
        self.arm_actuator_ids = list(range(7))

        # Home position from keyframe (matches XML exactly)
        self.q_home = self.model.key_qpos[0, :self.n_joints].copy()
        self.ctrl_home = self.model.key_ctrl[0, :self.n_joints].copy()

        # DOF indices for arm joints (for Jacobian column extraction)
        self.arm_dof_ids = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]

        # ── Spaces ───────────────────────────────────────────────────────
        goal_dim = REPR_DIM[action_repr]

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(self.n_joints * 2,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
        })

        bound = max_angle if action_repr in ('rotvec', 'euler') else 1.0
        self.action_space = spaces.Box(-bound, bound, shape=(goal_dim,), dtype=np.float32)

        # Pre-allocate Jacobian buffers: (3, nv)
        self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        # State
        self.goal_R = None
        self.step_count = 0
        self._viewer = None

    # ── EE accessors (body-based, not site) ──────────────────────────────

    def _get_ee_rotmat(self) -> np.ndarray:
        """Hand body rotation matrix (3, 3)."""
        return self.data.xmat[self.hand_body_id].reshape(3, 3).copy()

    def _get_ee_pos(self) -> np.ndarray:
        """Hand body position (3,)."""
        return self.data.xpos[self.hand_body_id].copy()

    def _get_qpos(self) -> np.ndarray:
        """Arm joint positions (7,). Joints are at qpos indices 0..6."""
        return self.data.qpos[:self.n_joints].astype(np.float32).copy()

    def _get_qvel(self) -> np.ndarray:
        """Arm joint velocities (7,). DOFs are at indices 0..6."""
        return self.data.qvel[:self.n_joints].astype(np.float32).copy()

    # ── Observation ──────────────────────────────────────────────────────

    def _make_obs(self) -> dict:
        qpos = self._get_qpos()
        qvel = self._get_qvel()
        achieved = rotmat_to_repr(self._get_ee_rotmat(), self.action_repr)
        desired = rotmat_to_repr(self.goal_R, self.action_repr)
        return {
            'observation': np.concatenate([qpos, qvel]).astype(np.float32),
            'achieved_goal': achieved,
            'desired_goal': desired,
        }

    # ── IK ───────────────────────────────────────────────────────────────

    def _ik_orientation(self, omega_error: np.ndarray) -> np.ndarray:
        """Damped pseudoinverse IK: rotation error → joint deltas.

        Args:
            omega_error: axis-angle error vector (3,) in WORLD frame

        Returns:
            dq: joint position deltas (7,)
        """
        self._jacr[:] = 0
        mujoco.mj_jacBody(self.model, self.data, None, self._jacr, self.hand_body_id)

        # Extract arm joint columns only (indices 0..6)
        Jr = self._jacr[:, self.arm_dof_ids]  # (3, 7)

        lam = self.ik_damping
        JJT = Jr @ Jr.T + lam**2 * np.eye(3)
        dq = Jr.T @ np.linalg.solve(JJT, omega_error.astype(np.float64))
        return dq

    # ── Reward (vectorized for HER) ──────────────────────────────────────

    def compute_reward(self, achieved_goal, desired_goal, info) -> np.ndarray:
        """Vectorized sparse reward. Called by HER on large batches."""
        dist = geodesic_distance_batch(achieved_goal, desired_goal, self.action_repr)
        return np.where(dist <= self.threshold, 0.0, -1.0).astype(np.float32)

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Set to home keyframe
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.ctrl[:] = self.model.key_ctrl[0]
        mujoco.mj_forward(self.model, self.data)

        # Sample goal: random rotation offset from current EE
        ee_R = self._get_ee_rotmat()
        axis = self.rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-8
        angle = self.rng.uniform(0.3, np.pi / 2)
        delta = Rotation.from_rotvec(axis * angle)
        self.goal_R = (Rotation.from_matrix(ee_R) * delta).as_matrix()

        self.step_count = 0
        obs = self._make_obs()
        dist = geodesic_distance_batch(obs['achieved_goal'], obs['desired_goal'], self.action_repr)
        return obs, {'distance': float(dist), 'is_success': False}

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float64)

        # Action → rotation delta
        if self.action_repr == 'rotvec':
            angle = np.linalg.norm(action)
            if angle > self.max_angle:
                action = action * (self.max_angle / angle)
            R_delta = Rotation.from_rotvec(action)
        else:
            R_delta = repr_to_rotation(action, self.action_repr)

        # Desired EE orientation
        R_current = Rotation.from_matrix(self._get_ee_rotmat())
        R_desired = R_current * R_delta

        # Orientation error as rotvec (for IK — always axis-angle)
        # R_error is in body frame: R_current⁻¹ · R_desired
        R_error = R_current.inv() * R_desired
        omega_body = R_error.as_rotvec()

        # mj_jacBody maps dq → world-frame angular velocity,
        # so IK needs world-frame error: ω_world = R_current @ ω_body
        omega_world = R_current.as_matrix() @ omega_body

        # IK → joint deltas
        dq = self._ik_orientation(omega_world)

        # Apply to actuators (ctrl = target joint position)
        for i in range(self.n_joints):
            jid = self.joint_ids[i]
            aid = self.arm_actuator_ids[i]
            current = self.data.qpos[self.model.jnt_qposadr[jid]]
            new_val = current + dq[i]
            # Clip to joint limits
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                new_val = np.clip(new_val, lo, hi)
            self.data.ctrl[aid] = new_val

        # Step simulation (10 substeps for stability)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._make_obs()
        dist = geodesic_distance_batch(obs['achieved_goal'], obs['desired_goal'], self.action_repr)
        reward = 0.0 if dist <= self.threshold else -1.0
        truncated = self.step_count >= self.max_steps

        info = {
            'distance': float(dist),
            'is_success': bool(dist <= self.threshold),
            'ee_pos': self._get_ee_pos(),
        }
        return obs, reward, False, truncated, info

    # ── Render ───────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == 'human':
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == 'rgb_array':
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None