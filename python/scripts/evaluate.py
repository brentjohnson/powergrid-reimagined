"""
Evaluate a trained MaskablePPO checkpoint against the Rust strategy bots.

Plays N full games with the masked policy and reports win rate plus
per-game and aggregate stats: cities, cities powered, money, plants,
placement, rounds, and episode length.

Usage:
    python scripts/evaluate.py --model runs/vs_bots/best_model --games 100
    python scripts/evaluate.py --model runs/selfplay/final_model --bot-difficulty hard
"""

import argparse
import json
from collections import defaultdict

from sb3_contrib import MaskablePPO

from powergrid_env import PowerGridSingleAgentEnv
from powergrid_env.stats import learner_stats


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
    parser.add_argument("--end-game-cities", type=int, default=None,
                        help="Play to this fixed end-game city trigger instead of the "
                             "rulebook number. Use the value the model was trained at.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-game output; show only the summary.")
    args = parser.parse_args()

    env = PowerGridSingleAgentEnv(
        num_players=args.num_players,
        learner_seat=args.learner_seat,
        bot_difficulty=args.bot_difficulty,
        seed=args.seed,
        reward_shaping=False,
        end_game_cities=args.end_game_cities,
    )
    model = MaskablePPO.load(args.model, device=args.device)

    wins = 0
    timeouts = 0
    total_cities = 0
    total_steps = 0
    total_powered = 0
    total_money = 0
    total_plants = 0
    total_capacity = 0
    total_rounds = 0
    placements = [0] * args.num_players
    plant_games = defaultdict(int)
    plant_wins = defaultdict(int)
    plant_lasts = defaultdict(int)
    for g in range(args.games):
        obs, info = env.reset()
        learner_id = env.game.player_ids()[args.learner_seat]
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

        winner_id = env.game.winner() if terminated else None
        stats = learner_stats(
            json.loads(env.game.state_json(learner_id)), learner_id, winner_id
        )
        total_powered += stats["powered"]
        total_money += stats["money"]
        total_plants += len(stats["plants"])
        total_capacity += stats["capacity"]
        total_rounds += stats["round"]
        if terminated and winner_id is not None:
            placements[stats["rank"] - 1] += 1
            for plant in stats["plants"]:
                plant_games[plant] += 1
                if won:
                    plant_wins[plant] += 1
                if stats["rank"] == args.num_players:
                    plant_lasts[plant] += 1
        if not args.quiet:
            outcome = "WIN " if won else ("stall" if not terminated else "loss")
            # winner_id is None both for non-terminated games and for the
            # degenerate invalid-action termination (game never reached a real
            # GameOver), so rank is meaningless in either case.
            rank = f"{stats['rank']}/{args.num_players}" if winner_id is not None else " - "
            print(f"game {g + 1:3d}/{args.games}: {outcome}  rank={rank}  "
                  f"cities={env.learner_cities:2d}  powered={stats['powered']:2d}  "
                  f"money={stats['money']:3d}  plants={stats['plants']}  "
                  f"round={stats['round']:2d}  steps={steps}")

    n = args.games
    ordinal = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, args.num_players + 1)]
    print()
    print(f"model:           {args.model}")
    print(f"opponents:       {args.num_players - 1} × {args.bot_difficulty} bot")
    if args.end_game_cities is not None:
        print(f"end-game cities: {args.end_game_cities} (rulebook override)")
    print(f"win rate:        {wins}/{n} = {wins / n:.1%}")
    print("placements:      " + "  ".join(
        f"{ordinal[i]}={placements[i]}" for i in range(args.num_players)))
    print(f"avg cities:      {total_cities / n:.1f}")
    print(f"avg powered:     {total_powered / n:.1f}")
    print(f"avg money:       {total_money / n:.1f}")
    print(f"avg plants:      {total_plants / n:.1f} (capacity {total_capacity / n:.1f})")
    print(f"avg rounds/game: {total_rounds / n:.1f}")
    print(f"avg steps/game:  {total_steps / n:.1f}")
    if timeouts:
        print(f"stalled games:   {timeouts} (hit --max-steps, counted as losses, no rank)")

    min_occurrences = 3

    def top_plants(hits: dict) -> list[tuple[int, float, int]]:
        ranked = [
            (plant, hits[plant] / plant_games[plant], plant_games[plant])
            for plant in plant_games
            if plant_games[plant] >= min_occurrences
        ]
        ranked.sort(key=lambda x: (-x[1], -x[2]))
        return ranked[:3]

    def format_plants(ranked: list[tuple[int, float, int]]) -> str:
        if not ranked:
            return "(insufficient data)"
        return ", ".join(f"#{p} ({rate:.0%}, n={n})" for p, rate, n in ranked)

    print(f"top plants (win):   {format_plants(top_plants(plant_wins))}")
    print(f"top plants (last):  {format_plants(top_plants(plant_lasts))}")
    env.close()


if __name__ == "__main__":
    main()
