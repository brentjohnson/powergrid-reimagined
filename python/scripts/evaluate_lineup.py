"""
Evaluate any mix of RL policy checkpoints and Rust heuristic bots in the same game.

Each --player argument is a difficulty keyword (easy / normal / hard) or a path
to a MaskablePPO .zip checkpoint (without the .zip suffix).  Repeat the flag for
each seat; player count is 2–6.

Examples:
    # Two model checkpoints vs easy and hard bots
    python scripts/evaluate_lineup.py \\
        --player runs/selfplay/best_model \\
        --player runs/vs_bots/best_model \\
        --player easy \\
        --player hard

    # Four RL models head-to-head
    python scripts/evaluate_lineup.py \\
        --player runs/run_a/best_model \\
        --player runs/run_b/best_model \\
        --player runs/run_c/best_model \\
        --player runs/run_d/best_model

    # Two models against two bots
    python scripts/evaluate_lineup.py \\
        --player runs/best_model --player runs/best_model \\
        --player hard --player hard --games 200
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from powergrid_py import Game

DIFFICULTY_KEYWORDS = {"easy", "normal", "hard"}
COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]


def make_labels(specs: list[str]) -> list[str]:
    """Return a display label per spec, adding #N suffixes for duplicates."""
    raw = [s if s in DIFFICULTY_KEYWORDS else Path(s).name for s in specs]
    freq = Counter(raw)
    seen: dict[str, int] = {}
    labels = []
    for r in raw:
        if freq[r] == 1:
            labels.append(r)
        else:
            seen[r] = seen.get(r, 0) + 1
            labels.append(f"{r}#{seen[r]}")
    return labels


def final_standings(game: Game) -> dict[str, dict]:
    """Per-player final stats keyed by player id, with authoritative money tiebreak."""
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
            "round": state["round"],
        }
    scores = {pid: (s["powered"], s["money"], s["cities"]) for pid, s in stats.items()}
    for pid, s in stats.items():
        s["rank"] = 1 + sum(other > scores[pid] for other in scores.values())
    return stats


def play_game(
    game: Game,
    seat_specs: dict[str, str],
    seat_models: dict[str, MaskablePPO | None],
    deterministic: bool,
    max_steps: int,
) -> int:
    """Run one game to completion; returns step count (== max_steps on stall)."""
    steps = 0
    while not game.is_terminal() and steps < max_steps:
        actor = game.current_actor()
        if actor is None:
            break
        model = seat_models[actor]
        if model is not None:
            mask = game.action_mask(actor)
            if not mask.any():
                # Auto-phase with no macro (e.g. Bureaucracy powering — removed
                # from the action space; training resolves it natively in
                # resolve_auto_phases). Resolve with the heuristic, which fires
                # the same optimal subset.
                action_json = game.bot_decide(actor, "hard")
                if action_json is None:
                    break
                game.apply(actor, action_json)
            else:
                obs = game.observation(actor)
                action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
                game.apply_action_id(actor, int(action))
        else:
            action_json = game.bot_decide(actor, seat_specs[actor])
            if action_json is not None:
                game.apply(actor, action_json)
            else:
                # Heuristic has no move (shouldn't happen); take first legal action.
                game.apply_action_id(actor, int(np.argmax(game.action_mask(actor))))
        steps += 1
    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate any mix of RL models and heuristic bots in the same game.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--player", dest="players", action="append", required=True, metavar="SPEC",
        help="Difficulty keyword (easy/normal/hard) or path to a MaskablePPO checkpoint "
             "(without .zip suffix).  Repeat for each seat (2–6 total).",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed; each game uses seed+g.  0 = random seeds.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Greedy action selection for RL models.  Default is stochastic.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000,
        help="Per-game step cap; stalled games are dropped from rankings.",
    )
    parser.add_argument(
        "--end-game-cities", type=int, default=None,
        help="Override the end-game city trigger (useful when models were trained at "
             "a non-rulebook value).",
    )
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-game output; show only the summary.")
    args = parser.parse_args()

    specs = args.players
    num_players = len(specs)
    if not 2 <= num_players <= 6:
        parser.error(f"Need 2–6 --player flags, got {num_players}.")

    labels = make_labels(specs)

    print("Lineup:")
    for i, (spec, lbl) in enumerate(zip(specs, labels)):
        kind = "bot" if spec in DIFFICULTY_KEYWORDS else "model"
        src = spec if spec in DIFFICULTY_KEYWORDS else str(spec)
        print(f"  seat {i}: [{kind}] {lbl}  ({src})")
    print()

    models: list[MaskablePPO | None] = []
    for spec in specs:
        if spec in DIFFICULTY_KEYWORDS:
            models.append(None)
        else:
            print(f"Loading {spec} ...")
            models.append(MaskablePPO.load(spec, device=args.device))
    print()

    placements = {lbl: [0] * num_players for lbl in labels}
    powered_total = {lbl: 0 for lbl in labels}
    cities_total = {lbl: 0 for lbl in labels}
    capacity_total = {lbl: 0 for lbl in labels}
    rounds_total = {lbl: 0 for lbl in labels}
    stalls = 0
    ranked_games = 0

    ordinals = (["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, num_players + 1)])[:num_players]

    for g in range(args.games):
        # Rotate seating each game so join order doesn't bias any player.
        offset = g % num_players
        rot_specs = specs[offset:] + specs[:offset]
        rot_labels = labels[offset:] + labels[:offset]
        rot_models = models[offset:] + models[:offset]

        seed = args.seed + g if args.seed else None
        game = Game(num_players, seed)
        # Player names must be unique short strings; truncate model basenames.
        names = []
        seen_names: set[str] = set()
        for lbl in rot_labels:
            base = lbl.replace("#", "")[:16]
            name = base
            suffix = 2
            while name in seen_names:
                name = f"{base[:14]}{suffix}"
                suffix += 1
            seen_names.add(name)
            names.append(name)
        game.start(names, COLORS[:num_players])
        if args.end_game_cities is not None:
            game.set_end_game_cities(args.end_game_cities)

        player_ids = game.player_ids()
        seat_specs = dict(zip(player_ids, rot_specs))
        seat_labels = dict(zip(player_ids, rot_labels))
        seat_models_map = dict(zip(player_ids, rot_models))

        steps = play_game(game, seat_specs, seat_models_map, args.deterministic, args.max_steps)
        if not game.is_terminal():
            stalls += 1
            if not args.quiet:
                print(f"game {g + 1:3d}/{args.games}: stalled after {steps} steps, dropped")
            continue

        ranked_games += 1
        stats = final_standings(game)
        parts = []
        for pid in player_ids:
            s = stats[pid]
            lbl = seat_labels[pid]
            placements[lbl][s["rank"] - 1] += 1
            powered_total[lbl] += s["powered"]
            cities_total[lbl] += s["cities"]
            capacity_total[lbl] += s["capacity"]
            rounds_total[lbl] += s["round"]
            parts.append(f"{lbl}={s['rank']}({s['powered']}p)")
        if not args.quiet:
            print(f"game {g + 1:3d}/{args.games}: steps={steps:4d}  " + "  ".join(sorted(parts)))

    print()
    print(f"games: {ranked_games} ranked" + (f", {stalls} stalled (dropped)" if stalls else ""))
    if args.end_game_cities is not None:
        print(f"end-game cities: {args.end_game_cities} (override)")
    if ranked_games == 0:
        return

    rg = ranked_games
    col_w = max(max(len(lbl) for lbl in labels), 8)
    header = f"{'player':<{col_w}}"
    for o in ordinals:
        header += f" {o:>5}"
    header += f"  {'win%':>6}  {'avg powered':>11}  {'avg cities':>10}  {'avg capacity':>12}  {'avg rounds':>10}"
    print()
    print(header)
    print("-" * len(header))
    for lbl in labels:
        p = placements[lbl]
        wins = p[0]
        row = f"{lbl:<{col_w}}"
        for cnt in p:
            row += f" {cnt:>5}"
        row += f"  {wins / rg:>6.1%}"
        row += f"  {powered_total[lbl] / rg:>11.1f}"
        row += f"  {cities_total[lbl] / rg:>10.1f}"
        row += f"  {capacity_total[lbl] / rg:>12.1f}"
        row += f"  {rounds_total[lbl] / rg:>10.1f}"
        print(row)


if __name__ == "__main__":
    main()
