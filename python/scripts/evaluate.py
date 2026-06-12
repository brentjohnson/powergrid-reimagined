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

from sb3_contrib import MaskablePPO

from powergrid_env import PowerGridSingleAgentEnv


def learner_stats(state: dict, learner_id: str) -> dict:
    """Final-state stats for the learner, plus its placement among all players.

    Placement is ranked by (cities powered last round, cities connected).
    Opponent money is hidden from the view, so the money tiebreak is not
    applied; tied players share the better rank.
    """
    city_owners = state["city_owners"]

    def cities_of(pid: str) -> int:
        return sum(pid in owners for owners in city_owners.values())

    scores = {p["id"]: (p["last_cities_powered"], cities_of(p["id"]))
              for p in state["players"]}
    me = next(p for p in state["players"] if p["id"] == learner_id)
    rank = 1 + sum(s > scores[learner_id] for s in scores.values())
    return {
        "money": me["money"],
        "powered": me["last_cities_powered"],
        "plants": [pl["number"] for pl in me["plants"]],
        "capacity": sum(pl["cities"] for pl in me["plants"]),
        "rank": rank,
        "round": state["round"],
    }


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

        stats = learner_stats(json.loads(env.game.state_json(learner_id)), learner_id)
        if won:
            stats["rank"] = 1
        total_powered += stats["powered"]
        total_money += stats["money"]
        total_plants += len(stats["plants"])
        total_capacity += stats["capacity"]
        total_rounds += stats["round"]
        if terminated:
            placements[stats["rank"] - 1] += 1
        outcome = "WIN " if won else ("stall" if not terminated else "loss")
        rank = f"{stats['rank']}/{args.num_players}" if terminated else " - "
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
    env.close()


if __name__ == "__main__":
    main()
