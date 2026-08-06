#!/usr/bin/env python3
"""
Migrate PGRLPOL6 / PGRLVAL1 .bin files (and their golden JSONs) across an
append-only observation growth (e.g. 582 -> 600, the 2026-08-06 end-game-race
section).

Because new features are appended at the END of the observation vector, an old
policy is migrated by zero-padding each layer-1 weight row: the padded network
computes bit-identical logits/values on any observation whose new features it
ignores (weight 0). This keeps the embedded expert, league snapshots, and any
exported champion playable and warm-startable (--init-policy-from) with no
retraining and no behavior change.

Golden JSONs ({"obs": [...], "logits"/"value": ...}) are migrated by appending
zeros to "obs" — with zero weights on the new inputs the recorded outputs stay
exact.

Usage:
    python scripts/migrate_policy_obs.py FILE_OR_DIR [...]   # in-place
    python scripts/migrate_policy_obs.py --assets            # embedded assets

    # old-format sb3 checkpoint -> migrated OBS_SIZE-wide .bin (policy layers;
    # the natural --init-policy-from warm-start source for new-format runs):
    python scripts/migrate_policy_obs.py --from-ckpt runs/x/best_model --out champ.bin

    # migrated .bin -> runnable sb3 checkpoint (policy layers loaded, value
    # head fresh) so evaluate_lineup.py can seat the frozen champion:
    python scripts/migrate_policy_obs.py --bin-to-ckpt champ.bin --out baseline/best_model

Directories are scanned for *.bin (league dirs). Files already at the current
OBS_SIZE are left untouched.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from powergrid_env.constants import OBS_SIZE  # noqa: E402

POLICY_MAGIC = b"PGRLPOL6"
VALUE_MAGIC = b"PGRLVAL1"
HEADER = 8 + 3 * 4


def migrate_bin(path: Path) -> bool:
    blob = path.read_bytes()
    magic = blob[:8]
    if magic not in (POLICY_MAGIC, VALUE_MAGIC):
        print(f"  skip {path}: unknown magic {magic!r}")
        return False
    obs, hidden, out = struct.unpack("<III", blob[8:HEADER])
    if obs == OBS_SIZE:
        print(f"  ok   {path}: already {OBS_SIZE}-wide")
        return False
    if obs > OBS_SIZE:
        raise SystemExit(f"{path}: file obs {obs} > current OBS_SIZE {OBS_SIZE}; "
                         "cannot shrink an observation append-only")
    counts = [hidden * obs, hidden, hidden * hidden, hidden, out * hidden, out]
    expected = HEADER + sum(counts) * 4
    if len(blob) != expected:
        raise SystemExit(f"{path}: length {len(blob)} != expected {expected}")

    body = blob[HEADER:]
    l1_bytes = counts[0] * 4
    l1, rest = body[:l1_bytes], body[l1_bytes:]
    pad = b"\x00" * ((OBS_SIZE - obs) * 4)
    rows = [l1[r * obs * 4:(r + 1) * obs * 4] + pad for r in range(hidden)]

    out_blob = bytearray(magic)
    out_blob += struct.pack("<III", OBS_SIZE, hidden, out)
    out_blob += b"".join(rows)
    out_blob += rest
    path.write_bytes(bytes(out_blob))
    print(f"  pad  {path}: {obs} -> {OBS_SIZE}")
    return True


def migrate_golden(path: Path) -> bool:
    golden = json.loads(path.read_text())
    obs = golden.get("obs")
    if obs is None:
        print(f"  skip {path}: no obs field")
        return False
    if len(obs) == OBS_SIZE:
        print(f"  ok   {path}: already {OBS_SIZE}-wide")
        return False
    if len(obs) > OBS_SIZE:
        raise SystemExit(f"{path}: golden obs {len(obs)} > OBS_SIZE {OBS_SIZE}")
    golden["obs"] = obs + [0.0] * (OBS_SIZE - len(obs))
    path.write_text(json.dumps(golden))
    print(f"  pad  {path}: obs {len(obs)} -> {OBS_SIZE}")
    return True


def migrate_path(path: Path) -> None:
    if path.is_dir():
        for p in sorted(path.glob("*.bin")):
            migrate_bin(p)
        for p in sorted(path.glob("*.golden.json")):
            migrate_golden(p)
    elif path.suffix == ".json":
        migrate_golden(path)
    else:
        migrate_bin(path)


def ckpt_to_migrated_bin(stem: str, out: Path) -> None:
    """sb3 checkpoint (any older append-only obs width) -> OBS_SIZE-wide
    PGRLPOL6 .bin, zero-padding l1 rows. Reads policy.pth straight from the
    zip, so no env/space reconstruction is needed."""
    import io
    import zipfile

    import torch

    from powergrid_env.export import policy_state_dict_to_bytes

    path = stem if stem.endswith(".zip") else stem + ".zip"
    with zipfile.ZipFile(path) as zf:
        sd = torch.load(io.BytesIO(zf.read("policy.pth")),
                        map_location="cpu", weights_only=True)
    w = sd["mlp_extractor.policy_net.0.weight"]
    old_obs = w.shape[1]
    if old_obs > OBS_SIZE:
        raise SystemExit(f"{path}: l1 width {old_obs} > OBS_SIZE {OBS_SIZE}")
    if old_obs < OBS_SIZE:
        pad = torch.zeros(w.shape[0], OBS_SIZE - old_obs, dtype=w.dtype)
        sd = dict(sd)
        sd["mlp_extractor.policy_net.0.weight"] = torch.cat([w, pad], dim=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(policy_state_dict_to_bytes(sd))
    print(f"  bin  {path} (obs {old_obs}) -> {out} (obs {OBS_SIZE})")


def bin_to_ckpt(bin_path: Path, out_stem: str) -> None:
    """Migrated .bin -> a minimal runnable MaskablePPO checkpoint: the three
    policy layers are loaded, everything else (value head, optimizer) is fresh.
    Play behavior (predict) is exactly the .bin policy — used to materialise a
    frozen champion as an evaluate_lineup.py / --h2h baseline."""
    from sb3_contrib import MaskablePPO

    from powergrid_env.export import policy_bytes_to_state_dict
    from powergrid_env.single_agent import PowerGridSingleAgentEnv

    blob = bin_path.read_bytes()
    if not blob.startswith(POLICY_MAGIC):
        raise SystemExit(f"{bin_path}: not a {POLICY_MAGIC.decode()} policy file")
    obs, hidden, _ = struct.unpack("<III", blob[8:HEADER])
    if obs != OBS_SIZE:
        raise SystemExit(f"{bin_path}: obs {obs} != OBS_SIZE {OBS_SIZE}; "
                         "migrate the .bin first")
    clone = policy_bytes_to_state_dict(blob)
    env = PowerGridSingleAgentEnv(num_players=4, bot_difficulty="hard", seed=0)
    model = MaskablePPO(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=dict(pi=[hidden, hidden], vf=[hidden, hidden])),
        device="cpu",
    )
    missing, unexpected = model.policy.load_state_dict(clone, strict=False)
    if unexpected:
        raise SystemExit(f"clone has keys sb3 does not: {unexpected}")
    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_stem)
    print(f"  ckpt {bin_path} -> {out_stem}.zip (policy loaded, "
          f"{len(missing)} value/log-std keys fresh)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help=".bin / golden.json files or dirs")
    parser.add_argument("--assets", action="store_true",
                        help="migrate the embedded assets/policies files")
    parser.add_argument("--from-ckpt", metavar="STEM",
                        help="sb3 checkpoint stem to convert to a migrated .bin "
                             "(requires --out)")
    parser.add_argument("--bin-to-ckpt", metavar="BIN",
                        help="migrated .bin to materialise as a runnable sb3 "
                             "checkpoint (requires --out, a stem without .zip)")
    parser.add_argument("--out", help="output path for --from-ckpt / --bin-to-ckpt")
    args = parser.parse_args()

    if args.from_ckpt or args.bin_to_ckpt:
        if args.paths or args.assets or (args.from_ckpt and args.bin_to_ckpt):
            parser.error("--from-ckpt / --bin-to-ckpt are exclusive, single-file modes")
        if not args.out:
            parser.error("--out is required with --from-ckpt / --bin-to-ckpt")
        if args.from_ckpt:
            ckpt_to_migrated_bin(args.from_ckpt, Path(args.out))
        else:
            bin_to_ckpt(Path(args.bin_to_ckpt), args.out)
        return

    paths = [Path(p) for p in args.paths]
    if args.assets:
        paths.append(Path(__file__).resolve().parents[2] / "assets" / "policies")
    if not paths:
        parser.error("give paths or --assets")
    for p in paths:
        if not p.exists():
            raise SystemExit(f"{p}: does not exist")
        migrate_path(p)


if __name__ == "__main__":
    main()
