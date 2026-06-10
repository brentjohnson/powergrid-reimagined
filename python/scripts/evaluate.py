"""
Evaluate a trained MaskablePPO checkpoint against the Rust strategy bots.

Plays N full games with the deterministic masked policy and reports win rate,
average learner cities, and average episode length.

Usage:
    python scripts/evaluate.py --model runs/vs_bots/best_model --games 100
    python scripts/evaluate.py --model runs/selfplay/final_model --bot-difficulty hard
"""

import argparse

from sb3_contrib import MaskablePPO

from powergrid_env import PowerGridSingleAgentEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to a saved MaskablePPO .zip (without .zip suffix).")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--learner-seat", type=int, default=0)
    parser.add_argument("--bot-difficulty", default="normal", choices=["easy", "normal", "hard"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true",
                        help="Greedy action selection. Default is stochastic: a "
                             "deterministic pass-everything policy can stall a game forever.")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Per-game step cap; a game hitting it counts as a loss.")
    args = parser.parse_args()

    env = PowerGridSingleAgentEnv(
        num_players=args.num_players,
        learner_seat=args.learner_seat,
        bot_difficulty=args.bot_difficulty,
        seed=args.seed,
        reward_shaping=False,
    )
    model = MaskablePPO.load(args.model, device=args.device)

    wins = 0
    timeouts = 0
    total_cities = 0
    total_steps = 0
    for g in range(args.games):
        obs, info = env.reset()
        terminated = False
        steps = 0
        reward = 0.0
        while not terminated and steps < args.max_steps:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=args.deterministic
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps += 1
        won = terminated and reward > 0
        wins += won
        timeouts += not terminated
        total_cities += env.learner_cities
        total_steps += steps
        outcome = "WIN " if won else ("stall" if not terminated else "loss")
        print(f"game {g + 1:3d}/{args.games}: {outcome}  "
              f"cities={env.learner_cities:2d}  steps={steps}")

    n = args.games
    print()
    print(f"model:          {args.model}")
    print(f"opponents:      {args.num_players - 1} × {args.bot_difficulty} bot")
    print(f"win rate:       {wins}/{n} = {wins / n:.1%}")
    print(f"avg cities:     {total_cities / n:.1f}")
    print(f"avg steps/game: {total_steps / n:.1f}")
    if timeouts:
        print(f"stalled games:  {timeouts} (hit --max-steps, counted as losses)")
    env.close()


if __name__ == "__main__":
    main()
