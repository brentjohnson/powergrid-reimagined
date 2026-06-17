"""
Train a single-agent MaskablePPO policy vs Rust strategy bots.

Usage:
    python scripts/train_vs_bots.py --total-timesteps 500_000 --bot-difficulty normal
    python scripts/train_vs_bots.py --resume-from runs/vs_bots/ckpt_450000_steps

Performance notes:
  - Each env step calls Rust directly (step_vs_bots): the learner action and all
    bot turns are applied in one PyO3 round-trip with no JSON serialisation.
  - DummyVecEnv (sequential, in-process) beats SubprocVecEnv here: Rust steps
    are so fast that IPC overhead dominates.
"""

import argparse
import os

import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from powergrid_env import PowerGridSingleAgentEnv
from powergrid_env.callbacks import PersistentBestEvalCallback


def make_env(args, seed: int, reward_shaping: bool, max_episode_steps: int | None = None):
    def _init():
        env = PowerGridSingleAgentEnv(
            num_players=args.num_players,
            learner_seat=args.learner_seat,
            bot_difficulty=args.bot_difficulty,
            seed=seed,
            reward_shaping=reward_shaping,
            end_game_cities=args.end_game_cities,
        )
        if max_episode_steps:
            # A policy that always passes can stall a game forever; truncate
            # such episodes instead of hanging the eval pass.
            env = Monitor(gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps))
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--learner-seat", type=int, default=0)
    parser.add_argument("--bot-difficulty", default="normal", choices=["easy", "normal", "hard"])
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of parallel envs (DummyVecEnv).")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        help="PyTorch device ('auto', 'cpu', 'cuda'). 'auto' picks GPU if available.")
    parser.add_argument("--run-dir", default="runs/vs_bots")
    parser.add_argument("--resume-from", default=None,
                        help="Path to a saved MaskablePPO .zip (without .zip suffix) "
                             "to continue training from. If unset, training starts fresh.")
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Save an intermediate checkpoint every N steps per env. 0 disables.")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Run an eval (win-rate) pass every N steps per env. 0 disables. "
                             "Logs eval/mean_reward to TensorBoard and keeps best_model.zip.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--reward-shaping", action=argparse.BooleanOptionalAction, default=True,
                        help="Add a per-round bonus proportional to cities powered, granted when "
                             "the learner's powering resolves. Disable with --no-reward-shaping "
                             "for pure win/loss reward.")
    parser.add_argument("--end-game-cities", type=int, default=None,
                        help="Play every game (training AND eval) to this fixed end-game "
                             "city trigger instead of the rulebook number. Eval scores at "
                             "different triggers aren't comparable: delete "
                             "best_mean_reward.json in the run dir when changing this "
                             "between runs.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="PPO entropy bonus coefficient. SB3's default is 0.0, which let "
                             "long runs collapse to a near-deterministic policy. Unlike other "
                             "hyperparameters, this is applied on --resume-from too "
                             "(overrides the checkpoint's value).")
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    env_fns = [make_env(args, args.seed + i, args.reward_shaping) for i in range(args.num_envs)]
    vec_env = DummyVecEnv(env_fns)

    if args.resume_from:
        model = MaskablePPO.load(args.resume_from, env=vec_env, device=args.device)
        model.tensorboard_log = os.path.join(args.run_dir, "tb")
        model.ent_coef = args.ent_coef
        print(f"Resumed from {args.resume_from} at {model.num_timesteps} timesteps "
              f"(ent_coef={args.ent_coef})")
    else:
        # n_epochs/batch_size are the dominant cost on CPU; these settings match
        # train_selfplay.py (fewer, larger mini-batch updates per rollout).
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
        # Eval without shaping so eval/mean_reward in [-1, 1] maps directly to
        # win rate: win_rate = (mean_reward + 1) / 2. Stochastic actions: a
        # deterministic pass-everything policy never finishes a game.
        eval_env = DummyVecEnv([
            make_env(args, args.seed + 10_000, reward_shaping=False, max_episode_steps=2000)
        ])
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
