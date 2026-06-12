"""
Roll out one game and print the event log.

Usage:
    # Random vs random:
    python scripts/play_game.py

    # Trained model vs bots:
    python scripts/play_game.py --model runs/vs_bots/final_model --render

    # Bot vs bot (all seats Rust strategy):
    python scripts/play_game.py --all-bots
"""

import argparse
import json
import numpy as np

from sb3_contrib import MaskablePPO
from powergrid_env import PowerGridAECEnv, RandomPolicy, RustBotPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=None, help="Path to a MaskablePPO checkpoint")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--all-bots", action="store_true", help="All seats use Rust bot")
    parser.add_argument("--bot-difficulty", default="normal", choices=["easy", "normal", "hard"])
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Stop after this many steps; some games can stall indefinitely.")
    parser.add_argument("--end-game-cities", type=int, default=None,
                        help="Play to this fixed end-game city trigger instead of the "
                             "rulebook number. Use the value the model was trained at.")
    args = parser.parse_args()

    env = PowerGridAECEnv(
        num_players=args.num_players,
        seed=args.seed,
        render_mode="ansi" if args.render else None,
        end_game_cities=args.end_game_cities,
    )
    env.reset(seed=args.seed)

    model = None
    if args.model:
        model = MaskablePPO.load(args.model)
        print(f"Loaded model from {args.model}")

    bot_policy = RustBotPolicy(difficulty=args.bot_difficulty)
    random_policy = RandomPolicy()

    step_count = 0
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        uuid = env._id_to_uuid.get(agent, agent)
        if terminated or truncated:
            action = None
        else:
            mask = info.get("action_mask", np.ones(env.action_space(agent).n, dtype=np.int8))
            if model and not args.all_bots and agent == env.possible_agents[0]:
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            elif args.all_bots or model:
                state = json.loads(env.game.state_json())
                action = bot_policy.act(env.game, uuid, state, obs, mask)
            else:
                action = random_policy.act(obs, mask)

        env.step(action)
        step_count += 1

        if args.render and env.game and not env.game.is_terminal():
            print(env.render())
            print(f"  → agent={agent} action={action}")
            print()

        if step_count >= args.max_steps:
            print(f"\nStopping after {step_count} steps (--max-steps); game has stalled.")
            break

    state = json.loads(env.game.state_json()) if env.game else {}
    print("\n=== Game Over ===")
    winner = env.game.winner() if env.game else None
    if winner:
        city_owners = state.get("city_owners", {})
        for p in state.get("players", []):
            if p["id"] == winner:
                n_cities = sum(1 for owners in city_owners.values() if p["id"] in owners)
                print(f"Winner: {p['name']} (${p['money']}, {n_cities} cities)")
                break
    print(f"Total steps: {step_count}")
    print("\nEvent log:")
    for msg in state.get("event_log", []):
        print(f"  {msg}")

    env.close()


if __name__ == "__main__":
    main()
