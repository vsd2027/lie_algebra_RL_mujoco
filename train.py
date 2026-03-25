"""
train.py — TD3 + HER training for orientation control experiments

Usage:
    python train.py --config configs/rotvec.yaml
    python train.py --config configs/euler.yaml
    python train.py --config configs/quat.yaml
    python train.py --action-repr rotvec --total-timesteps 500000
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    defaults = {}
    if pre_args.config:
        import yaml
        with open(pre_args.config) as f:
            defaults = yaml.safe_load(f)

    p = argparse.ArgumentParser(description="TD3+HER Orientation Control")
    p.add_argument("--config", type=str, default=None)

    # Environment
    p.add_argument("--action-repr", type=str, default="rotvec",
        choices=["rotvec", "euler", "quat"],
        help="rotvec = Lie algebra (paper's method), euler, quat")
    p.add_argument("--max-angle", type=float, default=0.1 * np.pi)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--max-steps", type=int, default=100)

    # Training
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--n-sampled-goal", type=int, default=4)
    p.add_argument("--noise-sigma", type=float, default=0.1)
    p.add_argument("--net-arch", type=int, nargs='+', default=[256, 256, 256])

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=5000)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--log-dir", type=str, default="./runs")
    p.add_argument("--device", type=str, default="cpu",
        help="cpu recommended — small network, single env")

    if defaults:
        p.set_defaults(**defaults)
    return p.parse_args()


def main():
    args = parse_args()

    from stable_baselines3 import TD3, HerReplayBuffer
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from envs.franka_orientation import FrankaOrientationGoalEnv

    exp_name = f"td3_her_a-{args.action_repr}__{args.seed}__{int(time.time())}"
    log_path = os.path.join(args.log_dir, exp_name)
    os.makedirs(log_path, exist_ok=True)

    # Save args
    with open(os.path.join(log_path, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    print(f"{'='*50}")
    print(f"  TD3 + HER — Orientation Control")
    print(f"  Action repr: {args.action_repr}")
    print(f"  Timesteps:   {args.total_timesteps:,}")
    print(f"  Device:      {args.device}")
    print(f"  Log:         {log_path}")
    print(f"{'='*50}\n")

    # Environments
    env = Monitor(FrankaOrientationGoalEnv(
        action_repr=args.action_repr,
        max_angle=args.max_angle,
        threshold=args.threshold,
        max_steps=args.max_steps,
        seed=args.seed,
    ), log_path)

    eval_env = Monitor(FrankaOrientationGoalEnv(
        action_repr=args.action_repr,
        max_angle=args.max_angle,
        threshold=args.threshold,
        max_steps=args.max_steps,
        seed=args.seed + 1000,
    ))

    # Action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.noise_sigma * np.ones(n_actions),
    )

    # Model
    model = TD3(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=args.n_sampled_goal,
            goal_selection_strategy="future",
        ),
        action_noise=action_noise,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        policy_kwargs=dict(net_arch=args.net_arch),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    # Eval callback
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    # Train
    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=eval_cb, progress_bar=True)
    elapsed = time.time() - t0

    model.save(os.path.join(log_path, "final_model"))
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Best model:  {log_path}/best_model.zip")
    print(f"Eval data:   {log_path}/evaluations.npz")
    print(f"Plot:        python plot.py {log_path}/evaluations.npz")


if __name__ == "__main__":
    main()