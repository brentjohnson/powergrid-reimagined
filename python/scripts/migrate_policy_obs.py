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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help=".bin / golden.json files or dirs")
    parser.add_argument("--assets", action="store_true",
                        help="migrate the embedded assets/policies files")
    args = parser.parse_args()

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
