"""Behavior-clone the Rust heuristic bot into a `NNetWrapper`, producing a
warm-start checkpoint for AlphaZero finetuning (`alphazero/train.py --resume`).

Generates a fixed dataset of (observation, mask, one-hot action, outcome)
pairs from `--games` hard-vs-hard games (see `imitation.py`), then trains on
it epoch-by-epoch, periodically evaluating with `arena.net_vs_bots`
(network-only greedy) against easy/normal/hard bots. `cloned.pt` is
overwritten whenever the vs-normal win rate improves — this is the file
Phase 2 finetuning resumes from. `metrics.csv` records every eval point.

Usage:
    python/.venv/bin/python -m alphazero.pretrain \
        --games 400 --epochs 20 --run-dir alphazero/runs/clone1
"""

from __future__ import annotations

import argparse
import csv
import os
import time

from powergrid_env.export import policy_state_dict_to_bytes

from . import arena
from .config import AZConfig
from .imitation import generate_examples
from .network import NNetWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--difficulty", default="hard", choices=["easy", "normal", "hard"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--net-width", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--end-game-cities", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", default="alphazero/runs/clone1")
    parser.add_argument("--export", default=None, help="Optional path to also write a PGRLPOL1 .bin")
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    cfg = AZConfig(
        net_width=args.net_width,
        value_hidden=args.value_hidden,
        lr=args.lr,
        batch_size=args.batch_size,
        train_epochs=1,
        device=args.device,
        seed=args.seed,
        run_dir=args.run_dir,
    )

    print(f"Generating {args.games} {args.difficulty}-vs-{args.difficulty} games...")
    t0 = time.time()
    examples, skipped = generate_examples(
        n_games=args.games,
        seed=args.seed,
        cfg=cfg,
        difficulty=args.difficulty,
        end_game_cities=args.end_game_cities,
    )
    print(
        f"Collected {len(examples)} examples ({skipped} moves skipped: "
        f"not representable as an action id) in {time.time() - t0:.1f}s"
    )
    if not examples:
        raise SystemExit("No examples collected — nothing to train on.")

    nnet = NNetWrapper(cfg)
    metrics_path = os.path.join(args.run_dir, "metrics.csv")
    fieldnames = [
        "epoch",
        "policy_loss",
        "value_loss",
        "win_easy",
        "win_normal",
        "win_hard",
        "is_best",
        "elapsed_s",
    ]
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_win_normal = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = nnet.train(examples)

        row = {
            "epoch": epoch,
            "policy_loss": losses["policy_loss"],
            "value_loss": losses["value_loss"],
            "win_easy": "",
            "win_normal": "",
            "win_hard": "",
            "is_best": False,
            "elapsed_s": None,
        }
        is_eval = epoch % args.eval_every == 0 or epoch == args.epochs
        if is_eval:
            seed_base = args.seed + epoch * 99_991
            win_easy = arena.net_vs_bots(
                nnet, cfg, n_games=args.eval_games, difficulty="easy",
                seed_base=seed_base, end_game_cities=args.end_game_cities,
            )
            win_normal = arena.net_vs_bots(
                nnet, cfg, n_games=args.eval_games, difficulty="normal",
                seed_base=seed_base, end_game_cities=args.end_game_cities,
            )
            win_hard = arena.net_vs_bots(
                nnet, cfg, n_games=args.eval_games, difficulty="hard",
                seed_base=seed_base, end_game_cities=args.end_game_cities,
            )
            is_best = win_normal > best_win_normal
            if is_best:
                best_win_normal = win_normal
                nnet.save(os.path.join(args.run_dir, "cloned.pt"))
            row.update(
                win_easy=win_easy, win_normal=win_normal, win_hard=win_hard, is_best=is_best
            )
            print(
                f"[epoch {epoch:3d}] policy_loss={losses['policy_loss']:.4f} "
                f"value_loss={losses['value_loss']:.4f}  "
                f"win vs easy/normal/hard = {win_easy:.1%}/{win_normal:.1%}/{win_hard:.1%} "
                f"(best vs normal={best_win_normal:.1%}){' *' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {epoch:3d}] policy_loss={losses['policy_loss']:.4f} "
                f"value_loss={losses['value_loss']:.4f}"
            )

        row["elapsed_s"] = time.time() - t0
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    if best_win_normal < 0:
        # eval never ran (epochs < eval_every and epochs != final) — shouldn't
        # happen since the last epoch always evaluates, but guard anyway.
        nnet.save(os.path.join(args.run_dir, "cloned.pt"))

    print(f"Wrote {args.run_dir}/cloned.pt and {metrics_path}")

    if args.export:
        sd = nnet.net.policy_state_dict()
        with open(args.export, "wb") as f:
            f.write(policy_state_dict_to_bytes(sd))
        print(f"Wrote {args.export}")


if __name__ == "__main__":
    main()
