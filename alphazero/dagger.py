"""DAgger (Dataset Aggregation) — expert iteration with the Rust hard bot as
the oracle.

Behavior cloning (`pretrain.py`) plateaus at ~8-11% vs normal because it only
sees the *teacher's* states; over a ~600-move game the clone's small per-move
errors compound into positions the teacher never demonstrated, and it flounders
there. AlphaZero finetuning made this worse (as a ~90% underdog the value head
sees nearly every position as losing, so MCTS visit-count targets carry little
signal and flatten the clone — measured 10.7%->2.0%).

DAgger fixes the compounding-error gap directly and reliably: roll out games
with the *current net* driving the learner seat (seats 1-3 are hard bots, which
also guarantee the game terminates), and at every state the net visits, record
the move the *hard bot* would play there as a one-hot target. Retrain on the
aggregate, repeat. The label is a sharp, high-signal expert action — there is
no value head or search in the decision loop to flatten it — and the rollout
distribution (net in seat 0 vs 3 bots) is exactly the distribution we evaluate
on. So the net learns the teacher's move precisely in the states it actually
reaches when playing the way it's scored.

Run from the repo root with the `python/` venv (see `alphazero/README.md`):

    python -m alphazero.dagger --resume alphazero/runs/clone3/cloned.pt \
        --iters 30 --games-per-iter 40 --run-dir alphazero/runs/dagger1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque

import numpy as np
from powergrid_env.constants import (
    BUILD_CITY_BASE,
    BUY_RESOURCE_BASE,
    CITY_INDEX,
    DONE_BUILDING,
    DONE_BUYING,
    N_ACTIONS,
    RESOURCE_IDX,
)

from . import arena
from .config import AZConfig
from .game import PowerGridGame, to_relative_vector
from .network import NNetWrapper
from .selfplay import Example, _sample_action


def _one_hot(action_id: int) -> np.ndarray:
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[action_id] = 1.0
    return pi


def bot_first_action_id(game: PowerGridGame, difficulty: str) -> int | None:
    """The single 94-action-space id the `difficulty` heuristic bot would play
    *next* in `game`'s current state — the DAgger label for that state.

    Most phases map one bot decision to one id. The two whole-turn *batch*
    phases (`build_cities`, `buy_resources`) don't have a single id, but their
    handlers don't end the turn, so the bot's turn is really a sequence of
    single-id steps; the label at this state is the *first* of them (the bot's
    highest-priority city / first resource unit — same ordering `imitation.py`
    decomposes). After the net applies its own move, the bot re-decides from
    the new state next turn, which is exactly the DAgger relabeling. Returns
    `None` if the bot has no move or its choice isn't representable as an id."""
    action_json = game.bot_decide_json(difficulty)
    if action_json is None:
        return None
    action = json.loads(action_json)
    kind = action["type"]
    if kind == "build_cities":
        cids = action["city_ids"]
        return BUILD_CITY_BASE + CITY_INDEX[cids[0]] if cids else DONE_BUILDING
    if kind == "build_city":
        return BUILD_CITY_BASE + CITY_INDEX[action["city_id"]]
    if kind == "done_building":
        return DONE_BUILDING
    if kind in ("buy_resources", "buy_resource_batch"):
        purchases = (
            action["purchases"]
            if kind == "buy_resource_batch"
            else [(action["resource"], action["amount"])]
        )
        if not purchases:
            return DONE_BUYING
        return BUY_RESOURCE_BASE + RESOURCE_IDX[purchases[0][0]]
    if kind == "done_buying":
        return DONE_BUYING
    return game.bot_decide_id(difficulty)


def generate_dagger_examples(
    nnet: NNetWrapper,
    cfg: AZConfig,
    n_games: int,
    seed: int,
    difficulty: str,
    end_game_cities: int | None,
) -> tuple[list[Example], int]:
    """Play `n_games` games with the net driving the learner seat (masked-
    softmax sampling — the deployed Rust bot's stochastic play, and it widens
    state coverage) and `difficulty` hard bots on the other seats. Record one
    `(obs, mask, one-hot bot-label, rank-outcome value)` example per learner
    state. Returns `(examples, skipped)` — `skipped` counts states where the
    bot had no representable move (should be ~0)."""
    examples: list[Example] = []
    skipped = 0
    for g in range(n_games):
        game = PowerGridGame(
            seed=seed + g, num_players=cfg.num_players, end_game_cities=end_game_cities
        )
        learner = game.player_ids()[0]
        history: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

        term = game.advance_bots(learner, difficulty)
        moves = 0
        while not term and moves < cfg.max_moves:
            obs = game.observation()
            mask = game.action_mask()
            label = bot_first_action_id(game, difficulty)
            if label is None or mask[label] == 0:
                skipped += 1
            else:
                history.append((obs, mask, _one_hot(label), learner))
            # Advance the learner seat by the NET's own (sampled) move, so the
            # states we collect are the ones the net actually reaches.
            probs, _ = nnet.predict(obs, mask)
            game.apply(_sample_action(probs))
            moves += 1
            term = game.advance_bots(learner, difficulty)

        if not game.is_terminal():
            # 3 hard bots have anti-stall guarantees, so this is a rare safety
            # net; drop the game rather than label it with a non-terminal value.
            continue
        outcome = game.outcome()
        player_ids = game.player_ids()
        examples.extend(
            (obs, mask, pi, to_relative_vector(player_ids, to_move, outcome))
            for obs, mask, pi, to_move in history
        )
    return examples, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--games-per-iter", type=int, default=40)
    parser.add_argument(
        "--difficulty",
        default="hard",
        choices=["easy", "normal", "hard"],
        help="Teacher/oracle bot that labels states (and fills the other seats).",
    )
    parser.add_argument("--train-batches", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--buffer-cap",
        type=int,
        default=300_000,
        help="Max aggregated examples kept (DAgger aggregates across iters).",
    )
    parser.add_argument("--eval-games", type=int, default=60)
    parser.add_argument("--net-width", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--end-game-cities", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", default="alphazero/runs/dagger1")
    parser.add_argument(
        "--resume",
        default=None,
        help="Warm-start checkpoint (normally the behavior-cloning clone).",
    )
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    cfg = AZConfig(
        net_width=args.net_width,
        value_hidden=args.value_hidden,
        lr=args.lr,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        device=args.device,
        seed=args.seed,
        run_dir=args.run_dir,
    )
    nnet = (
        NNetWrapper.load(args.resume, device=cfg.device, cfg=cfg)
        if args.resume
        else NNetWrapper(cfg)
    )

    metrics_path = os.path.join(args.run_dir, "metrics.csv")
    fields = [
        "iter",
        "new_examples",
        "skipped",
        "buffer_size",
        "policy_loss",
        "value_loss",
        "win_easy",
        "win_normal",
        "win_hard",
        "is_best",
        "elapsed_s",
    ]
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    buffer: deque[Example] = deque(maxlen=args.buffer_cap)
    best_win_normal = -1.0
    for it in range(1, args.iters + 1):
        t0 = time.time()
        new, skipped = generate_dagger_examples(
            nnet,
            cfg,
            n_games=args.games_per_iter,
            seed=args.seed + it * 10_000,
            difficulty=args.difficulty,
            end_game_cities=args.end_game_cities,
        )
        buffer.extend(new)
        losses = nnet.train(list(buffer), num_batches=cfg.train_batches)

        seed_base = args.seed + it * 99_991
        win = {
            d: arena.net_vs_bots(
                nnet,
                cfg,
                n_games=args.eval_games,
                difficulty=d,
                seed_base=seed_base,
                end_game_cities=args.end_game_cities,
                num_sims=0,  # net-only greedy — the exported artifact
            )
            for d in ("easy", "normal", "hard")
        }
        is_best = win["normal"] > best_win_normal
        ckpt = os.path.join(args.run_dir, f"iter_{it:04d}.pt")
        nnet.save(ckpt)
        if is_best:
            best_win_normal = win["normal"]
            nnet.save(os.path.join(args.run_dir, "dagger.pt"))

        row = {
            "iter": it,
            "new_examples": len(new),
            "skipped": skipped,
            "buffer_size": len(buffer),
            "policy_loss": losses["policy_loss"],
            "value_loss": losses["value_loss"],
            "win_easy": win["easy"],
            "win_normal": win["normal"],
            "win_hard": win["hard"],
            "is_best": is_best,
            "elapsed_s": time.time() - t0,
        }
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(
            f"[iter {it:3d}] examples+={len(new):5d} (buf={len(buffer)}, skip={skipped})  "
            f"policy_loss={losses['policy_loss']:.4f}  "
            f"win easy/normal/hard = {win['easy']:.1%}/{win['normal']:.1%}/{win['hard']:.1%}  "
            f"(best normal={best_win_normal:.1%}){' *' if is_best else ''}  "
            f"{row['elapsed_s']:.1f}s"
        )

    print(f"Wrote {args.run_dir}/dagger.pt (best vs normal={best_win_normal:.1%})")


if __name__ == "__main__":
    main()
