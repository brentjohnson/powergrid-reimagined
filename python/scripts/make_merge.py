"""
Weighted weight-space merge of MaskablePPO policy checkpoints into one
PGRLPOL6 ``.bin`` usable as a ``--init-policy-from`` warm start.

This generalizes ``make_soup.py`` (which does an equal-weight average) to
arbitrary per-model coefficients, which unlocks two things a plain soup cannot:

  * weighted interpolation (coeffs in [0,1] summing to 1), and
  * **extrapolation** — coeffs OUTSIDE [0,1] that step PAST a model along the
    direction from one checkpoint to another. E.g. to step 1.5x from a base B
    toward an improved child C (out = B + 1.5*(C - B) = -0.5*B + 1.5*C):

        python scripts/make_merge.py --out runs/sweep4/extrap.bin \
            --model runs/sweep4/x5-champ-g999/best_model --coeff -0.5 \
            --model runs/sweep4/a3-nsteps4096/best_model --coeff  1.5

Like souping, this is only meaningful when the inputs are fine-tunes of a
*shared* checkpoint so they sit in one basin and share a coordinate system
(Wortsman et al., "Model Soups"; Ilharco et al., "Editing Models with Task
Arithmetic" for the extrapolation case). Interpolation lands inside the basin;
extrapolation bets that the improvement direction generalizes a little past the
child — a non-gradient move to weights that forking+annealing the base cannot
reach.

Coefficients should sum to 1.0 so the merged weights stay on the affine hull of
the inputs (this preserves the activation scale the net was trained at); the
script warns if they do not. Only the three policy layers are written (same as
export_policy.py / make_soup.py); the value head is not part of the .bin and the
trainer re-learns it fresh.
"""

import argparse
import io
import zipfile

import torch

from powergrid_env.export import policy_state_dict_to_bytes


def load_policy_state_dict(model_path: str) -> dict:
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
    ap.add_argument("--coeff", action="append", required=True, type=float,
                    help="Per-model coefficient, one per --model, in the same order. "
                         "Should sum to 1.0 (interpolation stays in [0,1]; "
                         "extrapolation goes outside).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.model) < 2:
        raise SystemExit("need at least two --model checkpoints to merge")
    if len(args.coeff) != len(args.model):
        raise SystemExit(f"got {len(args.model)} --model but {len(args.coeff)} --coeff; "
                         "supply exactly one --coeff per --model")

    total = sum(args.coeff)
    if abs(total - 1.0) > 1e-6:
        print(f"WARNING: coefficients sum to {total:.4f}, not 1.0 — merged weights "
              "leave the affine hull and the activation scale will shift.")

    sds = [load_policy_state_dict(m) for m in args.model]

    keys = set(sds[0])
    for sd in sds[1:]:
        keys &= set(sd)

    merged = {}
    for k in keys:
        acc = None
        for c, sd in zip(args.coeff, sds):
            term = sd[k].float() * c
            acc = term if acc is None else acc + term
        merged[k] = acc

    blob = policy_state_dict_to_bytes(merged)
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"merged {len(sds)} checkpoints (coeff sum {total:.4f}) -> {args.out} "
          f"({len(blob)} bytes)")
    for c, m in zip(args.coeff, args.model):
        print(f"  {c:+.3f} * {m}")


if __name__ == "__main__":
    main()
