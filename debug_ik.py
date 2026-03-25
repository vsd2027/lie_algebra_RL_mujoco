"""
debug_ik.py — Isolate IK frame issue.

Tests raw IK outside the env to find what's broken.
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from robot_descriptions.loaders.mujoco import load_robot_description


def main():
    model = load_robot_description("panda_mj_description")
    data = mujoco.MjData(model)
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # Reset to home
    data.qpos[:] = model.key_qpos[0]
    data.ctrl[:] = model.key_ctrl[0]
    mujoco.mj_forward(model, data)

    R_init = data.xmat[hand_id].reshape(3, 3).copy()
    print(f"Initial EE rotmat:\n{R_init}")

    # Goal: rotate 0.3 rad about world z
    R_target = (Rotation.from_matrix(R_init) * Rotation.from_rotvec([0, 0, 0.3])).as_matrix()

    print(f"\nTarget distance: {(Rotation.from_matrix(R_init).inv() * Rotation.from_matrix(R_target)).magnitude():.3f} rad")

    # Try IK with different frame conventions
    for label, get_omega in [
        ("body-frame error (R_cur.inv * R_tgt)", lambda R_cur, R_tgt: (Rotation.from_matrix(R_cur).inv() * Rotation.from_matrix(R_tgt)).as_rotvec()),
        ("world-frame error (R_tgt * R_cur.inv)", lambda R_cur, R_tgt: (Rotation.from_matrix(R_tgt) * Rotation.from_matrix(R_cur).inv()).as_rotvec()),
    ]:
        # Fresh reset
        data.qpos[:] = model.key_qpos[0]
        data.ctrl[:] = model.key_ctrl[0]
        mujoco.mj_forward(model, data)

        print(f"\n--- {label} ---")
        for step in range(20):
            R_cur = data.xmat[hand_id].reshape(3, 3).copy()
            omega = get_omega(R_cur, R_target) * 0.5  # proportional gain

            # Jacobian
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, None, jacr, hand_id)
            Jr = jacr[:, :7]

            # Damped pseudoinverse
            lam = 0.05
            JJT = Jr @ Jr.T + lam**2 * np.eye(3)
            dq = Jr.T @ np.linalg.solve(JJT, omega)

            # Apply
            data.qpos[:7] += dq
            for j in range(7):
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j+1}")
                if model.jnt_limited[jnt_id]:
                    lo, hi = model.jnt_range[jnt_id]
                    data.qpos[j] = np.clip(data.qpos[j], lo, hi)
            mujoco.mj_forward(model, data)

            R_now = data.xmat[hand_id].reshape(3, 3)
            dist = (Rotation.from_matrix(R_now).inv() * Rotation.from_matrix(R_target)).magnitude()
            if step % 5 == 0 or step == 19:
                print(f"  step {step:2d}: dist={dist:.4f} rad  |omega|={np.linalg.norm(omega):.4f}")

    # Also test: does the env's step() work with direct qpos manipulation (no actuators)?
    print("\n--- Direct qpos (no actuators/simulation) ---")
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)

    for step in range(20):
        R_cur = data.xmat[hand_id].reshape(3, 3).copy()
        R_err = Rotation.from_matrix(R_cur).inv() * Rotation.from_matrix(R_target)
        omega = R_err.as_rotvec() * 0.5

        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, None, jacr, hand_id)
        Jr = jacr[:, :7]
        lam = 0.05
        dq = Jr.T @ np.linalg.solve(Jr @ Jr.T + lam**2 * np.eye(3), omega)

        # Apply directly to qpos, NO mj_step, just forward kinematics
        data.qpos[:7] += dq
        for j in range(7):
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j+1}")
            if model.jnt_limited[jnt_id]:
                lo, hi = model.jnt_range[jnt_id]
                data.qpos[j] = np.clip(data.qpos[j], lo, hi)
        mujoco.mj_forward(model, data)

        dist = (Rotation.from_matrix(data.xmat[hand_id].reshape(3, 3)).inv() * Rotation.from_matrix(R_target)).magnitude()
        if step % 5 == 0 or step == 19:
            print(f"  step {step:2d}: dist={dist:.4f} rad")


if __name__ == "__main__":
    main()
