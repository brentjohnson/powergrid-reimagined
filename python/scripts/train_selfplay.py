"""
Self-play training: shared-policy MaskablePPO across all seats.

Usage:
    python scripts/train_selfplay.py --num-players 4 --total-timesteps 1_000_000
    python scripts/train_selfplay.py --num-envs 8 --total-timesteps 5_000_000

Performance notes:
  - Each env step now calls Rust directly (no JSON serialisation) — ~14× faster
    raw env throughput vs the old JSON-bridge + PettingZoo wrapper chain.
  - All rollout transitions are real game steps (no black_death padding waste).
  - SubprocVecEnv is not used: each Rust step is so fast (~200 µs) that IPC
    overhead dominates. DummyVecEnv (sequential, in-process) is faster for this
    workload. A few envs (4–8) gives the best balance between env throughput and
    policy-forward amortisation.
"""

import argparse
import os

import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from powergrid_env import PowerGridSelfPlayEnv, PowerGridSingleAgentEnv
from powergrid_env.callbacks import PersistentBestEvalCallback


def make_env(num_players: int, seed: int, reward_shaping: bool):
    def _init():
        return PowerGridSelfPlayEnv(
            num_players=num_players, seed=seed, reward_shaping=reward_shaping
        )
    return _init


def make_eval_env(num_players: int, seed: int):
    """Eval vs Rust bots — an external yardstick for self-play progress."""
    def _init():
        env = PowerGridSingleAgentEnv(
            num_players=num_players,
            bot_difficulty="normal",
            seed=seed,
            reward_shaping=False,
        )
        # A policy that always passes can stall a game forever; truncate
        # such episodes instead of hanging the eval pass.
        return Monitor(gym.wrappers.TimeLimit(env, max_episode_steps=2000))
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of parallel envs (DummyVecEnv).")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        help="PyTorch device ('auto', 'cpu', 'cuda'). 'auto' picks GPU if available.")
    parser.add_argument("--run-dir", default="runs/selfplay")
    parser.add_argument("--resume-from", default=None,
                        help="Path to a saved MaskablePPO .zip (without .zip suffix) "
                             "to continue training from. If unset, training starts fresh.")
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Save an intermediate checkpoint every N vec-env steps. "
                             "0 disables.")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Evaluate vs normal Rust bots every N steps per env. 0 disables. "
                             "Logs eval/mean_reward to TensorBoard and keeps best_model.zip.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--reward-shaping", action=argparse.BooleanOptionalAction, default=True,
                        help="Add a per-round bonus proportional to cities powered to the "
                             "acting seat's step. Eval is always unshaped.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="PPO entropy bonus coefficient. SB3's default is 0.0, which let "
                             "long runs collapse to a near-deterministic policy. Unlike other "
                             "hyperparameters, this is applied on --resume-from too "
                             "(overrides the checkpoint's value).")
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    env_fns = [make_env(args.num_players, args.seed + i, args.reward_shaping)
               for i in range(args.num_envs)]
    vec_env = DummyVecEnv(env_fns)

    if args.resume_from:
        model = MaskablePPO.load(args.resume_from, env=vec_env, device=args.device)
        model.tensorboard_log = os.path.join(args.run_dir, "tb")
        model.ent_coef = args.ent_coef
        print(f"Resumed from {args.resume_from} at {model.num_timesteps} timesteps "
              f"(ent_coef={args.ent_coef})")
    else:
        # n_epochs/batch_size are the dominant cost on CPU.
        # Default PPO (n_epochs=10, batch=64) does 1280 mini-batch updates per
        # rollout; these settings do 64 (8192/512 * 4 epochs), giving ~3s/iter
        # instead of ~18s/iter with no significant quality loss in practice.
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=args.seed,
            device=args.device,
            n_steps=512,
            batch_size=512,
            n_epochs=4,
            ent_coef=args.ent_coef,
            tensorboard_log=os.path.join(args.run_dir, "tb"),
        )

    callbacks = []
    if args.save_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=args.save_freq,
            save_path=args.run_dir,
            name_prefix="ckpt",
        ))
    if args.eval_freq > 0:
        # eval/mean_reward in [-1, 1] maps directly to win rate vs bots:
        # win_rate = (mean_reward + 1) / 2.
        eval_env = DummyVecEnv([make_eval_env(args.num_players, args.seed + 10_000)])
        callbacks.append(PersistentBestEvalCallback(
            eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            best_model_save_path=args.run_dir,
            deterministic=False,
        ))

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks or None,
        reset_num_timesteps=not bool(args.resume_from),
    )
    model.save(os.path.join(args.run_dir, "final_model"))
    print(f"Saved to {args.run_dir}/final_model")
    vec_env.close()


if __name__ == "__main__":
    main()
