"""
visualize.py — Watch a trained policy on the Franka arm with goal frame visualization.

Draws RGB axes at the EE showing the goal orientation:
  Red = X, Green = Y, Blue = Z

Usage:
    mjpython visualize.py --model runs/td3_her_a-rotvec*/best_model.zip
    mjpython visualize.py --random
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def draw_frame(viewer, pos, rotmat, axis_len=0.15, axis_radius=0.005, alpha=0.8):
    """Draw RGB axes at pos with given rotation matrix using viewer.user_scn."""
    colors = [
        np.array([1, 0, 0, alpha], dtype=np.float32),  # X = red
        np.array([0, 1, 0, alpha], dtype=np.float32),  # Y = green
        np.array([0, 0, 1, alpha], dtype=np.float32),  # Z = blue
    ]

    for axis_idx in range(3):
        i = viewer.user_scn.ngeom
        if i >= viewer.user_scn.maxgeom:
            break

        # Axis direction in world frame
        axis_dir = rotmat[:, axis_idx]

        # Cylinder center = pos + axis_dir * axis_len/2
        center = pos + axis_dir * axis_len / 2

        # Build rotation matrix that aligns Z with axis_dir
        # MuJoCo cylinders are along Z by default
        z = axis_dir / (np.linalg.norm(axis_dir) + 1e-8)
        # Find a vector not parallel to z
        if abs(z[0]) < 0.9:
            x = np.cross(z, np.array([1, 0, 0]))
        else:
            x = np.cross(z, np.array([0, 1, 0]))
        x /= np.linalg.norm(x) + 1e-8
        y = np.cross(z, x)
        mat = np.column_stack([x, y, z]).flatten()

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.array([axis_radius, axis_radius, axis_len / 2]),
            pos=center.astype(np.float64),
            mat=mat.astype(np.float64),
            rgba=colors[axis_idx],
        )
        viewer.user_scn.ngeom += 1


def draw_sphere(viewer, pos, radius=0.01, rgba=None):
    """Draw a small sphere marker."""
    i = viewer.user_scn.ngeom
    if i >= viewer.user_scn.maxgeom:
        return
    if rgba is None:
        rgba = np.array([1, 1, 0, 0.6], dtype=np.float32)
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[i],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0]),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    viewer.user_scn.ngeom += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--action-repr", type=str, default="rotvec")
    p.add_argument("--random", action="store_true")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from envs.franka_orientation import FrankaOrientationGoalEnv

    env = FrankaOrientationGoalEnv(
        action_repr=args.action_repr,
        seed=args.seed,
    )

    model = None
    if args.model and not args.random:
        from stable_baselines3 import TD3
        model = TD3.load(args.model, env=env)
        print(f"Loaded: {args.model}")

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    for ep in range(args.episodes):
        obs, info = env.reset()
        print(f"\nEpisode {ep+1} | init dist: {info['distance']:.2f} rad ({np.degrees(info['distance']):.0f}°)")

        for step in range(200):
            if not viewer.is_running():
                break

            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample() * 0.3

            obs, reward, term, trunc, info = env.step(action)

            # ── Draw goal and current frames ─────────────────────────
            viewer.user_scn.ngeom = 0  # clear custom geoms

            ee_pos = env._get_ee_pos()
            ee_R = env._get_ee_rotmat()
            goal_R = env.goal_R

            # Current EE frame (thin, semi-transparent)
            draw_frame(viewer, ee_pos, ee_R, axis_len=0.25, axis_radius=0.01, alpha=0.3)

            # Goal frame (thick, bright) — drawn at same EE position
            draw_frame(viewer, ee_pos, goal_R, axis_len=0.35, axis_radius=0.02, alpha=0.9)

            # Small yellow sphere at EE
            draw_sphere(viewer, ee_pos, radius=0.02)

            viewer.sync()
            time.sleep(0.02)

            if step % 25 == 0 or info['is_success']:
                print(f"  step {step:3d} | dist: {info['distance']:.3f} rad | {'SUCCESS' if info['is_success'] else ''}")

            if term or trunc:
                break

        if not viewer.is_running():
            break

    viewer.close()
    env.close()


if __name__ == '__main__':
    main()