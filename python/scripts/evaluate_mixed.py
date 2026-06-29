"""
Evaluate a four-player game: easy vs normal vs hard Rust bots vs the trained
expert (a MaskablePPO checkpoint driven from Python).

Plays N full games and reports, per bot, how often it finished 1st/2nd/3rd/4th
plus its average cities powered and average plant capacity.

Usage:
    python scripts/evaluate_mixed.py --model runs/vs_bots/best_model --games 100
"""

import argparse
import json

import numpy as np
from sb3_contrib import MaskablePPO

from powergrid_py import Game

ROSTER = ["easy", "normal", "hard", "expert"]
COLORS = ["red", "blue", "green", "yellow"]


def final_standings(game: Game) -> dict[str, dict]:
    """Per-player final stats keyed by player id.

    Rank follows the game's scoring: cities powered last round, then money,
    then cities connected. Opponent money is zeroed in any single view, so
    each player's own view is fetched to recover their money.
    """
    state = json.loads(game.state_json())
    city_owners = state["city_owners"]

    stats = {}
    for p in state["players"]:
        own = next(q for q in json.loads(game.state_json(p["id"]))["players"]
                   if q["id"] == p["id"])
        stats[p["id"]] = {
            "powered": p["last_cities_powered"],
            "cities": sum(p["id"] in owners for owners in city_owners.values()),
            "money": own["money"],
            "capacity": sum(pl["cities"] for pl in p["plants"]),
        }
    scores = {pid: (s["powered"], s["money"], s["cities"]) for pid, s in stats.items()}
    for pid, s in stats.items():
        s["rank"] = 1 + sum(other > scores[pid] for other in scores.values())
    return stats


def play_game(game: Game, model, seat_difficulty: dict[str, str],
              deterministic: bool, max_steps: int) -> int:
    """Run one game to completion; returns steps taken (== max_steps on stall)."""
    steps = 0
    while not game.is_terminal() and steps < max_steps:
        actor = game.current_actor()
        if actor is None:
            break
        if seat_difficulty[actor] == "expert":
            obs = game.observation(actor)
            mask = game.action_mask(actor)
            action, _ = model.predict(obs, action_masks=mask,
                                      deterministic=deterministic)
            game.apply_action_id(actor, int(action))
        else:
            action_json = game.bot_decide(actor, seat_difficulty[actor])
            if action_json is not None:
                game.apply(actor, action_json)
            else:
                # Heuristic has no move (shouldn't happen on its turn);
                # take the first legal action so the game keeps moving.
                game.apply_action_id(actor, int(np.argmax(game.action_mask(actor))))
        steps += 1
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to a saved MaskablePPO .zip (without .zip suffix).")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true",
                        help="Greedy action selection for the expert. Default is "
                             "stochastic: greedy pass-everything play can stall forever.")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Per-game step cap; a game hitting it is dropped from rankings.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-game output; show only the summary.")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model, device=args.device)

    placements = {d: [0] * 4 for d in ROSTER}
    powered = {d: 0 for d in ROSTER}
    capacity = {d: 0 for d in ROSTER}
    owned = {d: 0 for d in ROSTER}
    stalls = 0
    ranked_games = 0

    for g in range(args.games):
        # Rotate seating each game so join order doesn't favour one bot.
        roster = ROSTER[g % 4:] + ROSTER[:g % 4]
        game = Game(4, args.seed + g if args.seed else None)
        game.start([d.capitalize() for d in roster], COLORS)
        seat_difficulty = dict(zip(game.player_ids(), roster))

        steps = play_game(game, model, seat_difficulty, args.deterministic,
                          args.max_steps)
        if not game.is_terminal():
            stalls += 1
            if not args.quiet:
                print(f"game {g + 1:3d}/{args.games}: stalled after {steps} steps, dropped")
            continue

        ranked_games += 1
        stats = final_standings(game)
        parts = []
        for pid, diff in seat_difficulty.items():
            s = stats[pid]
            placements[diff][s["rank"] - 1] += 1
            powered[diff] += s["powered"]
            capacity[diff] += s["capacity"]
            owned[diff] += s["cities"]
            parts.append(f"{diff}={s['rank']}({s['powered']}p)")
        if not args.quiet:
            print(f"game {g + 1:3d}/{args.games}: steps={steps:4d}  "
                  + "  ".join(sorted(parts)))

    print()
    print(f"model:  {args.model} (expert seat)")
    print(f"games:  {ranked_games} ranked"
          + (f", {stalls} stalled (dropped)" if stalls else ""))
    if ranked_games == 0:
        return
    print()
    print(f"{'bot':<8} {'1st':>5} {'2nd':>5} {'3rd':>5} {'4th':>5} "
          f"{'avg powered':>12} {'avg owned':>10} {'avg capacity':>13}")
    for diff in ROSTER:
        p = placements[diff]
        print(f"{diff:<8} {p[0]:>5} {p[1]:>5} {p[2]:>5} {p[3]:>5} "
              f"{powered[diff] / ranked_games:>12.1f} "
              f"{owned[diff] / ranked_games:>10.1f} "
              f"{capacity[diff] / ranked_games:>13.1f}")


if __name__ == "__main__":
    main()
