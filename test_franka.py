"""
test_franka.py — Tests for Menagerie Franka Panda in MuJoCo

Run: python test_franka.py
      (or) pytest test_franka.py -v

Tests model loading, body/joint/actuator discovery, forward kinematics,
Jacobian computation, IK sanity, and GoalEnv contract.
"""

import numpy as np
import mujoco
import sys

# ─── Helpers ────────────────────────────────────────────────────────────────

def load_panda():
    """Load Menagerie Panda. Returns (model, data)."""
    from robot_descriptions.loaders.mujoco import load_robot_description
    model = load_robot_description("panda_mj_description")
    data = mujoco.MjData(model)
    return model, data

def get_body_id(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"Body '{name}' not found"
    return bid

def get_joint_id(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"Joint '{name}' not found"
    return jid

# ─── Tests ──────────────────────────────────────────────────────────────────

def test_model_loads():
    """Model loads without error."""
    model, data = load_panda()
    assert model is not None
    assert data is not None
    print(f"  Model: nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}")
    print("  PASS: model loads")

def test_expected_bodies_exist():
    """All expected bodies exist: link0-7, hand, left/right finger."""
    model, _ = load_panda()
    expected = ['link0', 'link1', 'link2', 'link3', 'link4',
                'link5', 'link6', 'link7', 'hand',
                'left_finger', 'right_finger']
    for name in expected:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"Body '{name}' not found"
    print("  PASS: all expected bodies exist")

def test_expected_joints_exist():
    """7 arm joints (joint1-7) and 2 finger joints."""
    model, _ = load_panda()
    arm_joints = [f'joint{i}' for i in range(1, 8)]
    finger_joints = ['finger_joint1', 'finger_joint2']
    for name in arm_joints + finger_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"Joint '{name}' not found"
    print("  PASS: all expected joints exist")

def test_expected_actuators_exist():
    """8 actuators: actuator1-7 for arm, actuator8 for gripper."""
    model, _ = load_panda()
    for i in range(1, 9):
        name = f'actuator{i}'
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, f"Actuator '{name}' not found"
    assert model.nu == 8, f"Expected 8 actuators, got {model.nu}"
    print("  PASS: all expected actuators exist")

def test_no_sites():
    """Menagerie Panda has NO sites — code must not rely on sites."""
    model, _ = load_panda()
    assert model.nsite == 0, f"Expected 0 sites, got {model.nsite}"
    print("  PASS: confirmed no sites (must use body xpos/xmat)")

def test_hand_body_pose_at_home():
    """At home config, hand body has a reasonable position and valid rotation."""
    model, data = load_panda()

    # Set home keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0, "Keyframe 'home' not found"
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    hand_id = get_body_id(model, "hand")

    # Position
    pos = data.xpos[hand_id].copy()
    print(f"  Hand position at home: {pos}")
    assert pos.shape == (3,)
    # Hand should be somewhere reasonable (not at origin, not at infinity)
    assert np.linalg.norm(pos) > 0.1, "Hand too close to origin"
    assert np.linalg.norm(pos) < 2.0, "Hand too far from origin"
    assert pos[2] > 0.0, "Hand should be above ground"

    # Rotation matrix
    xmat = data.xmat[hand_id].copy()
    assert xmat.shape == (9,), f"Expected flat (9,), got {xmat.shape}"
    R = xmat.reshape(3, 3)
    # Should be a valid rotation: R @ R.T ≈ I, det(R) ≈ 1
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "xmat not orthogonal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "xmat det != 1"

    print("  PASS: hand body pose at home is valid")

def test_jacobian_shape_and_computation():
    """mj_jacBody returns (3, nv) Jacobians for translation and rotation."""
    model, data = load_panda()

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    hand_id = get_body_id(model, "hand")
    nv = model.nv

    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)

    print(f"  jacp shape: {jacp.shape}, jacr shape: {jacr.shape}")
    # Jacobian should not be all zeros (hand is connected to joints)
    assert np.any(jacp != 0), "Translation Jacobian is all zeros"
    assert np.any(jacr != 0), "Rotation Jacobian is all zeros"

    # First 7 columns correspond to arm joints — should have nonzero entries
    # (columns 7,8 are finger joints — irrelevant to hand pose)
    arm_jacp = jacp[:, :7]
    arm_jacr = jacr[:, :7]
    assert np.any(arm_jacp != 0), "Arm columns of jacp are all zeros"
    assert np.any(arm_jacr != 0), "Arm columns of jacr are all zeros"

    print("  PASS: Jacobian shape and content correct")

def test_ik_moves_hand():
    """Applying damped pseudoinverse IK actually changes the hand orientation."""
    model, data = load_panda()

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    hand_id = get_body_id(model, "hand")
    nv = model.nv

    # Record initial hand orientation
    R_before = data.xmat[hand_id].reshape(3, 3).copy()

    # Compute Jacobian
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)

    # Desired rotation error (small rotation around z-axis)
    omega_error = np.array([0.0, 0.0, 0.1])

    # Damped pseudoinverse IK (arm joints only)
    arm_dof_indices = list(range(7))  # first 7 DOFs are arm joints
    Jr = jacr[:, arm_dof_indices]  # (3, 7)
    lam = 0.05
    JJT = Jr @ Jr.T + lam**2 * np.eye(3)
    dq = Jr.T @ np.linalg.solve(JJT, omega_error)

    # Apply joint deltas to actuator controls
    for i in range(7):
        jid = get_joint_id(model, f'joint{i+1}')
        qpos_adr = model.jnt_qposadr[jid]
        current = data.qpos[qpos_adr]
        new_val = current + dq[i]
        # Clip to joint limits
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            new_val = np.clip(new_val, lo, hi)
        # Set actuator control (actuator i maps to joint i+1)
        data.ctrl[i] = new_val

    # Step simulation
    for _ in range(20):
        mujoco.mj_step(model, data)

    # Check hand moved
    R_after = data.xmat[hand_id].reshape(3, 3).copy()
    R_diff = R_before.T @ R_after
    trace = R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]
    angle_change = np.arccos(np.clip((trace - 1) / 2, -1, 1))

    print(f"  Orientation change after IK: {angle_change:.4f} rad ({np.degrees(angle_change):.1f}deg)")
    assert angle_change > 0.001, f"Hand didn't move: angle_change={angle_change}"

    print("  PASS: IK moves the hand")

def test_joint_qpos_mapping():
    """Verify joint name to qpos address mapping is correct for arm joints."""
    model, data = load_panda()

    for i in range(1, 8):
        jid = get_joint_id(model, f'joint{i}')
        qpos_adr = model.jnt_qposadr[jid]
        dof_adr = model.jnt_dofadr[jid]
        print(f"  joint{i}: id={jid}, qpos_adr={qpos_adr}, dof_adr={dof_adr}")
        # Arm joints should be hinge (1 DOF each)
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE, \
            f"joint{i} is not a hinge joint"

    # Finger joints should be slide
    for name in ['finger_joint1', 'finger_joint2']:
        jid = get_joint_id(model, name)
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE, \
            f"{name} is not a slide joint"

    print("  PASS: joint qpos mapping correct")

def test_home_keyframe_matches_xml():
    """Home keyframe qpos matches the XML definition."""
    model, data = load_panda()

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    expected_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    actual_qpos = data.qpos[:9].copy()

    print(f"  Expected: {expected_qpos}")
    print(f"  Actual:   {actual_qpos}")
    assert np.allclose(actual_qpos, expected_qpos, atol=1e-4), \
        f"Home keyframe mismatch"

    print("  PASS: home keyframe matches XML")

def test_scipy_rotation_consistency():
    """scipy Rotation matches MuJoCo xmat for the hand body."""
    from scipy.spatial.transform import Rotation

    model, data = load_panda()
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    hand_id = get_body_id(model, "hand")
    R_mujoco = data.xmat[hand_id].reshape(3, 3).copy()

    # Convert to rotvec and back
    r = Rotation.from_matrix(R_mujoco)
    rv = r.as_rotvec()
    R_roundtrip = Rotation.from_rotvec(rv).as_matrix()

    assert np.allclose(R_mujoco, R_roundtrip, atol=1e-6), \
        "scipy rotvec roundtrip doesn't match MuJoCo xmat"

    # Convert to euler and back
    euler = r.as_euler('xyz')
    R_euler_rt = Rotation.from_euler('xyz', euler).as_matrix()
    assert np.allclose(R_mujoco, R_euler_rt, atol=1e-6), \
        "scipy euler roundtrip doesn't match MuJoCo xmat"

    # Convert to quat and back
    q = r.as_quat()  # scalar-last
    R_quat_rt = Rotation.from_quat(q).as_matrix()
    assert np.allclose(R_mujoco, R_quat_rt, atol=1e-6), \
        "scipy quat roundtrip doesn't match MuJoCo xmat"

    print(f"  rotvec: {rv}")
    print(f"  euler:  {euler}")
    print(f"  quat:   {q}")
    print("  PASS: scipy rotation roundtrips match MuJoCo xmat")

def test_dof_indices_match_arm_joints():
    """Verify that DOF indices 0-6 correspond to joint1-7."""
    model, _ = load_panda()

    for i in range(1, 8):
        jid = get_joint_id(model, f'joint{i}')
        dof_adr = model.jnt_dofadr[jid]
        assert dof_adr == i - 1, \
            f"joint{i} dof_adr={dof_adr}, expected {i-1}"

    print("  PASS: DOF indices 0-6 map to joint1-7")


# ─── Run all ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_model_loads,
        test_expected_bodies_exist,
        test_expected_joints_exist,
        test_expected_actuators_exist,
        test_no_sites,
        test_home_keyframe_matches_xml,
        test_joint_qpos_mapping,
        test_dof_indices_match_arm_joints,
        test_hand_body_pose_at_home,
        test_jacobian_shape_and_computation,
        test_ik_moves_hand,
        test_scipy_rotation_consistency,
    ]

    # ── Env-level tests ────────────────────────────────────────────────

    def test_env_creates():
        """GoalEnv creates without crashing."""
        from envs.franka_orientation import FrankaOrientationGoalEnv
        env = FrankaOrientationGoalEnv(action_repr='rotvec', seed=42)
        assert env.hand_body_id >= 0
        assert env.n_joints == 7
        assert env.action_space.shape == (3,)
        assert env.observation_space['observation'].shape == (14,)
        assert env.observation_space['achieved_goal'].shape == (3,)
        env.close()
        print("  PASS: env creates for rotvec")

    def test_env_creates_all_reprs():
        """GoalEnv creates for all representations."""
        from envs.franka_orientation import FrankaOrientationGoalEnv
        for repr_name in ['rotvec', 'euler', 'quat']:
            env = FrankaOrientationGoalEnv(action_repr=repr_name, seed=42)
            obs, info = env.reset()
            assert 'distance' in info
            assert info['distance'] > 0
            env.close()
        print("  PASS: env creates for rotvec, euler, quat")

    def test_env_reset_returns_valid_obs():
        """Reset returns valid obs dict with correct shapes."""
        from envs.franka_orientation import FrankaOrientationGoalEnv
        env = FrankaOrientationGoalEnv(action_repr='rotvec', seed=42)
        obs, info = env.reset()
        assert obs['observation'].shape == (14,)
        assert obs['achieved_goal'].shape == (3,)
        assert obs['desired_goal'].shape == (3,)
        assert not np.any(np.isnan(obs['observation']))
        assert not np.any(np.isnan(obs['achieved_goal']))
        assert not np.any(np.isnan(obs['desired_goal']))
        assert 0.1 < info['distance'] < np.pi, f"Init distance out of range: {info['distance']}"
        env.close()
        print(f"  Init distance: {info['distance']:.3f} rad")
        print("  PASS: reset returns valid obs")

    def test_env_step_changes_state():
        """Taking a step should change the observation."""
        from envs.franka_orientation import FrankaOrientationGoalEnv
        env = FrankaOrientationGoalEnv(action_repr='rotvec', seed=42)
        obs0, _ = env.reset()
        action = env.action_space.sample() * 0.5
        obs1, reward, term, trunc, info = env.step(action)
        assert not np.allclose(obs0['observation'], obs1['observation'], atol=1e-4), \
            "Observation didn't change after step"
        assert reward in (0.0, -1.0), f"Unexpected reward: {reward}"
        assert isinstance(info['is_success'], bool)
        assert isinstance(info['distance'], float)
        env.close()
        print("  PASS: step changes state")

    def test_env_10_steps_moves_ee():
        """10 directed steps should reduce distance (IK is working)."""
        from envs.franka_orientation import FrankaOrientationGoalEnv
        from scipy.spatial.transform import Rotation
        env = FrankaOrientationGoalEnv(action_repr='rotvec', seed=42)
        obs, info = env.reset()
        dist_init = info['distance']

        # Take 10 steps with action = proper relative rotation toward goal
        for _ in range(10):
            # Correct: Log(R_current⁻¹ · R_goal), NOT (desired - achieved)
            R_current = Rotation.from_rotvec(obs['achieved_goal'])
            R_goal = Rotation.from_rotvec(obs['desired_goal'])
            R_error = R_current.inv() * R_goal
            action = R_error.as_rotvec() * 0.3  # proportional gain
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, term, trunc, info = env.step(action)

        dist_final = info['distance']
        print(f"  Distance: {dist_init:.3f} → {dist_final:.3f} rad")
        assert dist_final < dist_init, \
            f"IK not working: distance didn't decrease ({dist_init:.3f} → {dist_final:.3f})"
        env.close()
        print("  PASS: 10 steps reduces distance (IK works)")

    def test_env_compute_reward_vectorized():
        """compute_reward handles batched inputs correctly."""
        from envs.franka_orientation import FrankaOrientationGoalEnv
        env = FrankaOrientationGoalEnv(action_repr='rotvec', seed=42)
        env.reset()

        # Batch of goals
        achieved = np.random.randn(100, 3).astype(np.float32) * 0.5
        desired = achieved.copy()  # same → should all be 0 reward
        rewards = env.compute_reward(achieved, desired, {})
        assert rewards.shape == (100,)
        np.testing.assert_allclose(rewards, 0.0, err_msg="Self-distance should give 0 reward")

        # Far apart → should all be -1
        desired_far = achieved + 1.0
        rewards_far = env.compute_reward(achieved, desired_far, {})
        assert np.all(rewards_far == -1.0), "Far goals should give -1 reward"

        env.close()
        print("  PASS: compute_reward is vectorized and correct")

    def test_env_quat_obs_dim():
        """Quaternion env should have goal dim 4."""
        from envs.franka_orientation import FrankaOrientationGoalEnv
        env = FrankaOrientationGoalEnv(action_repr='quat', seed=42)
        obs, _ = env.reset()
        assert obs['achieved_goal'].shape == (4,)
        assert obs['desired_goal'].shape == (4,)
        assert env.action_space.shape == (4,)
        env.close()
        print("  PASS: quat env has 4D goals and actions")

    tests.extend([
        test_env_creates,
        test_env_creates_all_reprs,
        test_env_reset_returns_valid_obs,
        test_env_step_changes_state,
        test_env_10_steps_moves_ee,
        test_env_compute_reward_vectorized,
        test_env_quat_obs_dim,
    ])

    passed = 0
    failed = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*50}")
    sys.exit(1 if failed > 0 else 0)