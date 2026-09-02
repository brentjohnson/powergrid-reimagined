"""
Model soup: uniformly average the policy weights of several MaskablePPO
checkpoints into one PGRLPOL6 ``.bin`` usable as a ``--init-policy-from`` warm
start.

Souping only works when the inputs are fine-tunes of a *shared* checkpoint —
they then sit in one loss basin and their weight-space average is itself a valid
(often flatter/better) minimum (Wortsman et al., "Model Soups"). The wave-14
cross-lineage arms x2/x3/x4 all fork the same donor state (s4-y3 @750M) and
differ only in gamma (0.997/0.999/0.9995), so they are exactly this setting: an
average across the gamma sweep.

Only the three policy layers are written (same as export_policy.py); the value
head is not part of the .bin and the trainer re-learns it fresh from the soup.

Usage:
    python scripts/make_soup.py --out runs/sweep4/soup-x234.bin \
        --model runs/sweep4/x2-y3-g997/best_model \
        --model runs/sweep4/x3-y3-g999/best_model \
        --model runs/sweep4/x4-y3-g9995/best_model
"""

import argparse
import io
import zipfile

import torch

from powergrid_env.export import policy_bytes_to_state_dict, policy_state_dict_to_bytes


def load_policy_state_dict(model_path: str) -> dict:
    # A PGRLPOL6 .bin (migrated to the current OBS_SIZE) carries only the three
    # policy layers — exactly what the soup averages — so accept it directly,
    # which lets prepare() rebuild soups after an obs bump when the source .zip
    # checkpoints are the old width and can no longer be re-serialized.
    if model_path.endswith(".bin"):
        with open(model_path, "rb") as f:
            return policy_bytes_to_state_dict(f.read())
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    with zipfile.ZipFile(model_path) as zf:
        return torch.load(
            io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="Checkpoint stem (repeatable, >=2). Must be same net width.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.model) < 2:
        raise SystemExit("need at least two --model checkpoints to soup")

    sds = [load_policy_state_dict(m) for m in args.model]

    # Average over the keys common to every input (the policy path plus whatever
    # else they share); policy_state_dict_to_bytes only reads the three policy
    # layers, so a shared value head averaging along too is harmless.
    keys = set(sds[0])
    for sd in sds[1:]:
        keys &= set(sd)

    soup = {}
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in sds], dim=0)
        soup[k] = stacked.mean(dim=0)

    blob = policy_state_dict_to_bytes(soup)
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"souped {len(sds)} checkpoints -> {args.out} ({len(blob)} bytes)")
    for m in args.model:
        print(f"  + {m}")


if __name__ == "__main__":
    main()
