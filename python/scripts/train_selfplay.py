"""
Frozen-opponent self-play: MaskablePPO vs periodic snapshots of itself.

The learner occupies one seat of a PowerGridSingleAgentEnv; the other seats
are driven by a frozen copy of its own policy network, running natively in
Rust (bot_difficulty="policy"). Every --snapshot-every timesteps the current
weights are frozen and pushed to the envs, so the opposition improves with
the learner. Rewards are learner-centric (+1 win / -1 loss on the learner's
final transition), which keeps credit assignment correct — unlike the old
single-stream shared-policy env, whose terminal reward went to whichever
seat happened to finish the round's bookkeeping.

Usage:
    python scripts/train_selfplay.py --num-players 4 --total-timesteps 1_000_000
    python scripts/train_selfplay.py --num-envs 8 --total-timesteps 5_000_000

Performance notes:
  - Each env step calls Rust directly; opponent moves (including snapshot
    inference) run inside Rust, so there is no per-move Python overhead.
  - SubprocVecEnv is not used: each Rust step is so fast that IPC overhead
    dominates. DummyVecEnv (sequential, in-process) is faster for this
    workload. A few envs (4–8) gives the best balance between env throughput
    and policy-forward amortisation.
"""

import argparse
import os

import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from powergrid_env import PowerGridSingleAgentEnv
from powergrid_env.callbacks import (
    RULEBOOK_END_GAME_CITIES,
    EndGameCurriculumCallback,
    LeagueSnapshotCallback,
    OpponentSnapshotCallback,
    PersistentBestEvalCallback,
    ShapingAnnealCallback,
)
from powergrid_env.export import policy_state_dict_to_bytes


def make_env(num_players: int, seed: int, reward_shaping: bool,
             shaping_mode: str = "absolute",
             end_game_cities: int | None = None, bot_mix: float = 0.0,
             terminal_reward: str = "winloss"):
    def _init():
        return PowerGridSingleAgentEnv(
            num_players=num_players, bot_difficulty="policy", seed=seed,
            reward_shaping=reward_shaping, shaping_mode=shaping_mode,
            end_game_cities=end_game_cities, bot_mix=bot_mix,
            terminal_reward=terminal_reward,
        )
    return _init


def league_mix(value: str) -> tuple[float, float, float]:
    parts = tuple(float(x) for x in value.split(","))
    if len(parts) != 3 or any(w < 0 for w in parts) or sum(parts) <= 0:
        raise ValueError(f"expected three non-negative weights, got {value!r}")
    return parts


def make_eval_env(num_players: int, seed: int, end_game_cities: int | None = None):
    """Eval vs Rust bots — an external yardstick for self-play progress."""
    def _init():
        env = PowerGridSingleAgentEnv(
            num_players=num_players,
            bot_difficulty="normal",
            seed=seed,
            reward_shaping=False,
            end_game_cities=end_game_cities,
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
    parser.add_argument("--snapshot-every", type=int, default=100_000,
                        help="Freeze the current policy and hand it to the training envs "
                             "as the opponent every N total timesteps.")
    parser.add_argument("--bot-mix", type=float, default=0.0,
                        help="Per-episode probability of facing 'hard' heuristic bots "
                             "instead of the policy snapshot (grounding/diversity knob). "
                             "Only used with --no-league; the league mix subsumes it.")
    parser.add_argument("--league", action=argparse.BooleanOptionalAction, default=True,
                        help="Population-based opponents: sample each episode's opponent "
                             "from a league of past snapshots + heuristic bots instead of "
                             "only the latest snapshot (avoids the self-play echo "
                             "chamber). Snapshots persist in <run-dir>/league/ and are "
                             "reloaded on --resume-from. --no-league restores the old "
                             "latest-snapshot-only behaviour.")
    parser.add_argument("--league-past-k", type=int, default=4,
                        help="How many past snapshots (besides the latest) are in the "
                             "opponent pool at a time, resampled at every snapshot.")
    parser.add_argument("--league-mix", type=league_mix, default=(0.5, 0.3, 0.2),
                        metavar="LATEST,PAST,BOTS",
                        help="Sampling weights for the opponent pool: latest snapshot, "
                             "past snapshots (shared), heuristic hard bots. "
                             "Default 0.5,0.3,0.2.")
    parser.add_argument("--anneal-shaping-steps", type=int, default=0,
                        help="Linearly anneal the shaping bonus to zero over this many "
                             "timesteps (shaping should bootstrap, not steer the final "
                             "policy). 0 = constant shaping (old behaviour). The scale "
                             "is derived from num_timesteps, so it resumes correctly.")
    parser.add_argument("--terminal-reward", choices=["winloss", "placement"],
                        default="winloss",
                        help="Terminal reward. 'winloss' = +1/-1. 'placement' = final "
                             "rank mapped onto [-1, +1] (4p: +1/+1/3/-1/3/-1) — denser "
                             "signal, values 2nd over last. Eval is always winloss, so "
                             "eval/mean_reward stays comparable.")
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Save an intermediate checkpoint every N vec-env steps. "
                             "0 disables.")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Evaluate vs normal Rust bots every N steps per env. 0 disables. "
                             "Logs eval/mean_reward to TensorBoard and keeps best_model.zip.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--reward-shaping", action=argparse.BooleanOptionalAction, default=True,
                        help="Add a per-round powered-cities bonus to the learner's step. "
                             "Eval is always unshaped.")
    parser.add_argument("--shaping-mode", choices=["absolute", "relative"], default="absolute",
                        help="Powered-cities shaping quantity. 'absolute' (default) rewards "
                             "the learner's own powered count — a clean 'build more = more "
                             "reward' teacher for from-scratch runs. 'relative' rewards the "
                             "lead over the best opponent (aligned with winning, can go "
                             "negative) but is a poor cold-start teacher. Bootstrap with "
                             "absolute, then --resume-from with relative to fine-tune.")
    parser.add_argument("--end-game-cities", type=int, default=None,
                        help="Play every game (training AND eval) to this fixed end-game "
                             "city trigger instead of the rulebook number. Mutually "
                             "exclusive with --curriculum-start. Eval scores at different "
                             "triggers aren't comparable: delete best_mean_reward.json in "
                             "the run dir when changing this between runs.")
    parser.add_argument("--curriculum-start", type=int, default=None,
                        help="Enable an end-game-cities curriculum starting at this trigger "
                             "(e.g. 3). Games end when a player builds this many cities, so "
                             "wins are frequent and the terminal signal is dense. Unset = "
                             "always play to the rulebook trigger.")
    parser.add_argument("--curriculum-step", type=int, default=2,
                        help="How much to raise the trigger at each curriculum bump.")
    parser.add_argument("--curriculum-every", type=int, default=5_000_000,
                        help="Raise the trigger every N total timesteps until it reaches "
                             "the rulebook value. The stage is derived from num_timesteps, "
                             "so --resume-from lands on the right stage.")
    parser.add_argument("--ent-coef", type=float, default=0.03,
                        help="PPO entropy bonus coefficient. SB3's default is 0.0, which let "
                             "long runs collapse to a near-deterministic policy; even 0.01 "
                             "collapsed (entropy → ~0.15 nats by 20M steps), so the default "
                             "is now 0.03 to keep exploration alive. Unlike other "
                             "hyperparameters, this is applied on --resume-from too "
                             "(overrides the checkpoint's value).")
    parser.add_argument("--net-width", type=int, default=128,
                        help="Hidden width of the policy/value MLP (two equal-width hidden "
                             "layers) for a fresh run. Default 128. (SB3's own default and "
                             "the pre-2026-06 checkpoints were 64; the embedded expert.bin "
                             "is still 64-wide.) Ignored with --resume-from (architecture "
                             "comes from the checkpoint). The Rust Expert port reads the "
                             "width from the exported policy header, so widening needs no "
                             "Rust changes — but the net must stay two equal-width layers "
                             "(the PGRLPOL1 format constraint).")
    args = parser.parse_args()

    if args.end_game_cities is not None and args.curriculum_start is not None:
        parser.error("--end-game-cities and --curriculum-start are mutually exclusive: "
                     "the curriculum would overwrite the fixed trigger at training start.")

    os.makedirs(args.run_dir, exist_ok=True)

    env_fns = [make_env(args.num_players, args.seed + i, args.reward_shaping,
                        shaping_mode=args.shaping_mode,
                        end_game_cities=args.end_game_cities, bot_mix=args.bot_mix,
                        terminal_reward=args.terminal_reward)
               for i in range(args.num_envs)]
    vec_env = DummyVecEnv(env_fns)

    if args.resume_from:
        model = MaskablePPO.load(args.resume_from, env=vec_env, device=args.device)
        model.tensorboard_log = os.path.join(args.run_dir, "tb")
        model.ent_coef = args.ent_coef
        print(f"Resumed from {args.resume_from} at {model.num_timesteps} timesteps "
              f"(ent_coef={args.ent_coef}, "
              f"shaping={'off' if not args.reward_shaping else args.shaping_mode})")
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
            # Two equal-width hidden layers (dict form keeps the separate
            # policy_net/value_net heads the PGRLPOL1 exporter reads).
            policy_kwargs=dict(net_arch=dict(pi=[args.net_width, args.net_width],
                                             vf=[args.net_width, args.net_width])),
            tensorboard_log=os.path.join(args.run_dir, "tb"),
        )
        print(f"Fresh model (ent_coef={args.ent_coef}, net_width={args.net_width}, "
              f"shaping={'off' if not args.reward_shaping else args.shaping_mode})")

    # Seed the envs with an initial opponent snapshot before learn() resets
    # them (SB3 resets envs before any callback fires); the callback keeps
    # refreshing it from there.
    vec_env.env_method(
        "set_opponent_policy", policy_state_dict_to_bytes(model.policy.state_dict())
    )

    if args.league:
        callbacks = [LeagueSnapshotCallback(
            vec_env,
            snapshot_every=args.snapshot_every,
            league_dir=os.path.join(args.run_dir, "league"),
            past_k=args.league_past_k,
            mix=args.league_mix,
            seed=args.seed,
            verbose=1,
        )]
    else:
        callbacks = [OpponentSnapshotCallback(vec_env, snapshot_every=args.snapshot_every,
                                              verbose=1)]
    if args.reward_shaping and args.anneal_shaping_steps > 0:
        callbacks.append(ShapingAnnealCallback(
            vec_env, anneal_steps=args.anneal_shaping_steps))
    if args.save_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=args.save_freq,
            save_path=args.run_dir,
            name_prefix="ckpt",
        ))
    eval_env = None
    eval_callback = None
    if args.eval_freq > 0:
        # eval/mean_reward in [-1, 1] maps directly to win rate vs bots:
        # win_rate = (mean_reward + 1) / 2.
        eval_env = DummyVecEnv([make_eval_env(args.num_players, args.seed + 10_000,
                                              args.end_game_cities)])
        eval_callback = PersistentBestEvalCallback(
            eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            best_model_save_path=args.run_dir,
            deterministic=False,
        )
    if args.curriculum_start is not None:
        # Placed before the eval callback so a stage bump retargets the eval
        # env (and resets the best bar) before any eval at the same step.
        callbacks.append(EndGameCurriculumCallback(
            vec_env,
            eval_env,
            eval_callback,
            start=args.curriculum_start,
            step=args.curriculum_step,
            bump_every=args.curriculum_every,
            target=RULEBOOK_END_GAME_CITIES[args.num_players],
        ))
    if eval_callback is not None:
        callbacks.append(eval_callback)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not bool(args.resume_from),
    )
    model.save(os.path.join(args.run_dir, "final_model"))
    print(f"Saved to {args.run_dir}/final_model")
    vec_env.close()


if __name__ == "__main__":
    main()
