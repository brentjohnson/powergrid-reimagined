#!/usr/bin/env python3
"""
Interpret and compare Expert policy networks in *game terms*.

The Expert bot is a small MLP: OBS_SIZE(600) -> H -> tanh -> H -> tanh ->
N_ACTIONS(26) logits (H read from the file header; the shipped expert.bin is
128-wide). Its inputs and outputs already carry game meaning (obs_layout.rs /
action_labels.rs); this tool answers the two questions those labels *don't*:

  Q1  What does the network compute, in game terms?
      - Rigorous: local input->logit attribution (an exact chain-rule Jacobian,
        since it's only two tanh layers), averaged over real game states and
        grouped by observation section, giving per-macro "driven +by / -by
        <game feature>" statements.
      - Exploratory (clearly fenced): per-hidden-unit fingerprints — what each
        unit reads, what it pushes, and when it fires. Dense tanh units are
        often polysemantic, so treat these as suggestive, not definitive.

  Q2  What did one network learn relative to another? (--compare)
      - Behavioral diff: both policies' masked-softmax action distributions over
        one shared corpus, per macro and per phase (mean probability shift + KL).
      - Attribution diff: J_champion - J_ancestor grouped by (macro, section) —
        the crisp "sensitivity to <feature> went up/down" statement.
      - Per-unit drift: valid here *because* the champions are a warm-started
        lineage (--init-policy-from), so hidden units stay index-aligned; for
        two independently trained nets this section is meaningless (permutation
        symmetry) and it says so.

The corpus is generated on-policy by driving the *first* --policy through real
games vs Rust heuristic bots (PowerGridSingleAgentEnv), so it reflects states
that policy actually visits.

Usage:
    # Q1 only (single policy)
    python scripts/analyze_policy.py --policy ../assets/policies/expert.bin

    # Q1 + Q2 (champion vs its warm-start ancestor)
    #   first export the ancestor sb3 checkpoint to a .bin:
    #   python scripts/migrate_policy_obs.py --from-ckpt \
    #       runs/sweep4/w5-y3-gamma/best_model --out /tmp/w5-y3-gamma.bin
    python scripts/analyze_policy.py \
        --policy ../assets/policies/expert.bin \
        --compare /tmp/w5-y3-gamma.bin \
        --label-a x5-champ-g999 --label-b w5-y3-gamma

Writes a self-contained report.html (--out) and prints a short summary.
"""

import argparse
import html
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "src"))

from powergrid_env.constants import (  # noqa: E402
    CITY_IDS,
    N_ACTIONS,
    OBS_SIZE,
    REGION_NAMES,
)

POLICY_MAGIC = b"PGRLPOL6"
HEADER = 8 + 3 * 4

# --------------------------------------------------------------------------- #
# Observation labels — port of crates/powergrid-netviz/src/obs_layout.rs.
# Section table: (name, start, len, per-local-index label fn).
# --------------------------------------------------------------------------- #
PLANT_FEATS = ["number", "kind", "cost", "cities", "capacity"]
ACTUAL_FEATS = ["number", "kind", "cost", "cities", "present", "discount"]
FUTURE_FEATS = ["number", "kind", "cost", "cities", "present"]
OPP_FEATS = ["plants", "cities", "capacity", "last_powered"]
RESOURCE_NAMES = ["coal", "oil", "gas", "uranium"]
MARKET_META = ["step3_triggered", "in_step3", "deck_size"]
PHASE_SCALARS = ["phase_id", "step", "round", "end_game_cities", "turn_order_pos"]
ENDGAME_FEATS = [
    "self progress", "self deficit (sat 6)",
    "opp 0 progress", "opp 1 progress", "opp 2 progress", "opp 3 progress",
    "opp 4 progress", "min opp deficit (sat 6)",
    "self powerable now",
    "opp 0 powerable now", "opp 1 powerable now", "opp 2 powerable now",
    "opp 3 powerable now", "opp 4 powerable now",
    "self last powered", "powered margin", "can finish now", "money after finish",
]
NC = len(CITY_IDS)  # 49

_SECTIONS = [
    ("Self money", 1, lambda i: "money"),
    ("Self resources", 4, lambda i: RESOURCE_NAMES[i]),
    ("Self plants", 15, lambda i: f"slot {i // 5}: {PLANT_FEATS[i % 5]}"),
    ("Self cities", NC, lambda i: CITY_IDS[i]),
    ("Opponents", 20, lambda i: f"opp {i // 4}: {OPP_FEATS[i % 4]}"),
    ("Opponent cities", 5 * NC, lambda i: f"opp {i // NC}: {CITY_IDS[i % NC]}"),
    ("City slot counts", NC, lambda i: CITY_IDS[i]),
    ("Active regions", 7, lambda i: REGION_NAMES[i]),
    ("Plant market (cards 1-4)", 24, lambda i: f"card {i // 6 + 1}: {ACTUAL_FEATS[i % 6]}"),
    ("Plant market (cards 5-8)", 20, lambda i: f"card {i // 5 + 5}: {FUTURE_FEATS[i % 5]}"),
    ("Market meta", 3, lambda i: MARKET_META[i]),
    ("Resource market", 4, lambda i: RESOURCE_NAMES[i]),
    ("Phase scalars", 5, lambda i: PHASE_SCALARS[i]),
    ("Phase scratch", 8, lambda i: f"scratch[{i}]"),
    ("Connection cost to city", NC, lambda i: CITY_IDS[i]),
    ("Opponent fuel demand", 4, lambda i: RESOURCE_NAMES[i]),
    ("Opponent plants", 5 * 3 * 5, lambda i: f"opp {i // 15}: slot {(i % 15) // 5}: {PLANT_FEATS[i % 5]}"),
    ("End-game race", 18, lambda i: ENDGAME_FEATS[i]),
]


def _build_labels():
    names = [""] * OBS_SIZE  # section name per global index
    labels = [""] * OBS_SIZE  # "section: local label" per global index
    bounds = []  # (name, start, end)
    pos = 0
    for name, ln, fn in _SECTIONS:
        bounds.append((name, pos, pos + ln))
        for k in range(ln):
            names[pos + k] = name
            labels[pos + k] = f"{name}: {fn(k)}"
        pos += ln
    assert pos == OBS_SIZE, f"section table covers {pos}, expected {OBS_SIZE}"
    return names, labels, bounds


SECTION_NAME, OBS_LABEL, SECTION_BOUNDS = _build_labels()
SECTION_ORDER = [b[0] for b in SECTION_BOUNDS]


# Macro labels — port of crates/powergrid-netviz/src/action_labels.rs.
NOMINATE_BASE, N_NOMINATE = 0, 6
AUCTION_PASS, AUCTION_RAISE = 6, 7
BUILD_COUNT_BASE, N_BUILD_COUNT = 8, 7
BUY_SUBSET_BASE, N_BUY_SUBSETS = 15, 8
DISCARD_PLANT_BASE, N_DISCARD_PLANT = 23, 3


def macro_label(i: int) -> str:
    if NOMINATE_BASE <= i < NOMINATE_BASE + N_NOMINATE:
        return f"Nominate[slot {i - NOMINATE_BASE}]"
    if i == AUCTION_PASS:
        return "Auction:Pass"
    if i == AUCTION_RAISE:
        return "Auction:Raise+1"
    if BUILD_COUNT_BASE <= i < BUILD_COUNT_BASE + N_BUILD_COUNT:
        n = i - BUILD_COUNT_BASE
        return "Build:Nothing" if n == 0 else f"Build:{n} cheapest"
    if BUY_SUBSET_BASE <= i < BUY_SUBSET_BASE + N_BUY_SUBSETS:
        mask = i - BUY_SUBSET_BASE
        if mask == 0:
            return "Buy:Nothing"
        bits = ",".join(str(b) for b in range(3) if mask & (1 << b))
        return f"Buy:fuel plants [{bits}]"
    if DISCARD_PLANT_BASE <= i < DISCARD_PLANT_BASE + N_DISCARD_PLANT:
        return f"DiscardPlant[slot {i - DISCARD_PLANT_BASE}]"
    return f"unknown[{i}]"


MACRO_LABELS = [macro_label(i) for i in range(N_ACTIONS)]

# Which phase each macro belongs to, for phase-bucketed reporting.
MACRO_PHASE = {}
for i in range(N_ACTIONS):
    if i < AUCTION_PASS or i in (AUCTION_PASS, AUCTION_RAISE):
        MACRO_PHASE[i] = "Auction"
    elif BUILD_COUNT_BASE <= i < BUILD_COUNT_BASE + N_BUILD_COUNT:
        MACRO_PHASE[i] = "BuildCities"
    elif BUY_SUBSET_BASE <= i < BUY_SUBSET_BASE + N_BUY_SUBSETS:
        MACRO_PHASE[i] = "BuyResources"
    else:
        MACRO_PHASE[i] = "DiscardPlant"


# --------------------------------------------------------------------------- #
# Policy loading + forward pass (numpy; mirrors MlpPolicy::forward_trace).
# --------------------------------------------------------------------------- #
class Policy:
    def __init__(self, path: Path):
        blob = Path(path).read_bytes()
        if blob[:8] != POLICY_MAGIC:
            raise SystemExit(f"{path}: not a {POLICY_MAGIC.decode()} policy (magic {blob[:8]!r})")
        obs, hidden, out = struct.unpack("<III", blob[8:HEADER])
        if obs != OBS_SIZE or out != N_ACTIONS:
            raise SystemExit(f"{path}: dims (obs={obs}, out={out}) != ({OBS_SIZE}, {N_ACTIONS})")
        f = np.frombuffer(blob[HEADER:], dtype="<f4").astype(np.float64)
        counts = [hidden * obs, hidden, hidden * hidden, hidden, out * hidden, out]
        if f.size != sum(counts):
            raise SystemExit(f"{path}: payload {f.size} floats != expected {sum(counts)}")
        o = 0
        parts = []
        for c in counts:
            parts.append(f[o:o + c])
            o += c
        self.path = Path(path)
        self.obs, self.hidden, self.out = obs, hidden, out
        self.l1_w = parts[0].reshape(hidden, obs)   # [H, 600]
        self.l1_b = parts[1]
        self.l2_w = parts[2].reshape(hidden, hidden)  # [H, H]
        self.l2_b = parts[3]
        self.out_w = parts[4].reshape(out, hidden)   # [26, H]
        self.out_b = parts[5]

    def forward(self, X: np.ndarray) -> dict:
        """Batched forward. X: [K, 600]. Returns z1/z2/logits (+ tanh derivs)."""
        a1 = X @ self.l1_w.T + self.l1_b
        z1 = np.tanh(a1)
        a2 = z1 @ self.l2_w.T + self.l2_b
        z2 = np.tanh(a2)
        logits = z2 @ self.out_w.T + self.out_b
        return {"z1": z1, "z2": z2, "logits": logits,
                "d1": 1.0 - z1 * z1, "d2": 1.0 - z2 * z2}


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-wise softmax over legal (mask==1) entries. logits/mask: [K, 26]."""
    z = np.where(mask > 0, logits, -1e30)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z) * (mask > 0)
    s = e.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return e / s


def golden_check(pol: Policy) -> str:
    golden = REPO / "assets" / "policies" / "expert.golden.json"
    if not golden.exists():
        return "golden file not found (skipped)"
    g = json.loads(golden.read_text())
    obs = np.array(g["obs"], dtype=np.float64)[None, :]
    got = pol.forward(obs)["logits"][0]
    ref = np.array(g["logits"], dtype=np.float64)
    zref = np.array(g["zeros_logits"], dtype=np.float64)
    zgot = pol.forward(np.zeros((1, OBS_SIZE)))["logits"][0]
    d = max(np.abs(got - ref).max(), np.abs(zgot - zref).max())
    ok = d < 1e-3
    return f"max|Δlogit| vs torch golden = {d:.2e} ({'PASS' if ok else 'FAIL'})"


# --------------------------------------------------------------------------- #
# Corpus generation — drive real games with `driver`, record (obs, mask, facts).
# --------------------------------------------------------------------------- #
def build_corpus(driver: Policy, games: int, num_players: int, seed: int,
                 difficulty: str, max_states: int) -> dict:
    from powergrid_env import PowerGridSingleAgentEnv

    env = PowerGridSingleAgentEnv(
        num_players=num_players, learner_seat=0, bot_difficulty=difficulty,
        seed=seed, reward_shaping=False,
    )
    rng = np.random.default_rng(seed)
    Xs, Ms = [], []
    for _ in range(games):
        obs, info = env.reset()
        mask = info["action_mask"]
        terminal = False
        steps = 0
        while not terminal and steps < 2000:
            if mask.sum() == 0:
                break
            Xs.append(obs.astype(np.float64))
            Ms.append(mask.astype(np.float64))
            logit = driver.forward(obs[None, :])["logits"][0]
            p = masked_softmax(logit[None, :], mask[None, :].astype(np.float64))[0]
            action = int(rng.choice(N_ACTIONS, p=p))
            obs, _, terminal, _, info = env.step(action)
            mask = info["action_mask"]
            steps += 1
        if len(Xs) >= max_states:
            break
    env.close()
    X = np.array(Xs[:max_states])
    M = np.array(Ms[:max_states])
    # Derived game facts (all read straight from the observation).
    facts = {
        "phase_id": np.rint(X[:, 441] * 9).astype(int),
        "round": X[:, 443] * 50,
        "self cities": X[:, 20:20 + NC].sum(axis=1),
        "self money (norm)": X[:, 0],
        "end-game deficit": X[:, 583] * 6,
        "powered margin": X[:, 597],
        "self powerable now": X[:, 590] * 21,
        "can finish now": X[:, 598],
    }
    return {"X": X, "M": M, "facts": facts}


# --------------------------------------------------------------------------- #
# Attribution: exact local Jacobian d logit / d input, averaged over states.
# --------------------------------------------------------------------------- #
def mean_jacobian(pol: Policy, X: np.ndarray, M: np.ndarray, chunk: int) -> np.ndarray:
    """Mean signed J[a, i] over states where macro a is legal. Shape [26, 600]."""
    sumJ = np.zeros((pol.out, pol.obs))
    cnt = np.zeros(pol.out)
    for s in range(0, len(X), chunk):
        xb = X[s:s + chunk]
        mb = M[s:s + chunk]
        f = pol.forward(xb)
        # M1 = diag(d1) @ l1_w                    -> [b, H, 600]
        M1 = f["d1"][:, :, None] * pol.l1_w[None, :, :]
        # T  = l2_w @ M1                          -> [b, H, 600]
        T = np.einsum("hj,bjk->bhk", pol.l2_w, M1, optimize=True)
        # M2 = diag(d2) @ T                       -> [b, H, 600]
        M2 = f["d2"][:, :, None] * T
        # Jc = out_w @ M2                         -> [b, 26, 600]
        Jc = np.einsum("ah,bhk->bak", pol.out_w, M2, optimize=True)
        sumJ += np.einsum("bak,ba->ak", Jc, mb, optimize=True)
        cnt += mb.sum(axis=0)
    cnt = np.where(cnt == 0, 1.0, cnt)
    return sumJ / cnt[:, None], cnt


def finite_diff_check(pol: Policy, X: np.ndarray, J: np.ndarray, n: int = 20) -> str:
    """Spot-check the analytic Jacobian at one state against finite differences."""
    rng = np.random.default_rng(0)
    x = X[rng.integers(len(X))].copy()
    Jx = _jac_one(pol, x)
    eps = 1e-4
    errs = []
    for _ in range(n):
        i = int(rng.integers(pol.obs))
        a = int(rng.integers(pol.out))
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        num = (pol.forward(xp[None, :])["logits"][0][a]
               - pol.forward(xm[None, :])["logits"][0][a]) / (2 * eps)
        errs.append(abs(num - Jx[a, i]))
    return f"finite-diff |analytic - numeric| max = {max(errs):.2e} (mean {np.mean(errs):.2e})"


def _jac_one(pol: Policy, x: np.ndarray) -> np.ndarray:
    f = pol.forward(x[None, :])
    d1, d2 = f["d1"][0], f["d2"][0]
    return pol.out_w @ (d2[:, None] * (pol.l2_w @ (d1[:, None] * pol.l1_w)))


def hidden_push(pol: Policy, X: np.ndarray, chunk: int) -> np.ndarray:
    """Mean effective push of hidden-unit j onto logit a: d logit_a / d z1[j],
    averaged over the corpus. Shape [26, H]."""
    sP = np.zeros((pol.out, pol.hidden))
    for s in range(0, len(X), chunk):
        f = pol.forward(X[s:s + chunk])
        G = f["d2"][:, :, None] * pol.l2_w[None, :, :]      # [b, H, H]
        Pc = np.einsum("ah,bhj->baj", pol.out_w, G, optimize=True)  # [b, 26, H]
        sP += Pc.sum(axis=0)
    return sP / len(X)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


# --------------------------------------------------------------------------- #
# HTML report helpers
# --------------------------------------------------------------------------- #
def esc(s) -> str:
    return html.escape(str(s))


def bar(v: float, vmax: float, width: int = 120) -> str:
    """A center-anchored diverging bar cell (blue negative / red positive)."""
    if vmax <= 0:
        vmax = 1.0
    frac = max(-1.0, min(1.0, v / vmax))
    half = width / 2
    w = abs(frac) * half
    color = "#d1495b" if v >= 0 else "#3a7ca5"
    if v >= 0:
        left, bw = half, w
    else:
        left, bw = half - w, w
    return (f'<span class="barbox" style="width:{width}px">'
            f'<span class="bartick"></span>'
            f'<span class="bar" style="left:{left:.1f}px;width:{bw:.1f}px;'
            f'background:{color}"></span></span>')


def top_features(J_row: np.ndarray, k: int):
    order = np.argsort(-np.abs(J_row))
    return [(int(i), float(J_row[i])) for i in order[:k]]


def macro_attr_table(J: np.ndarray, cnt: np.ndarray, k: int, phase: str) -> str:
    rows = []
    ids = [i for i in range(N_ACTIONS) if MACRO_PHASE[i] == phase and cnt[i] >= 20]
    if not ids:
        return ""
    vmax = max(np.abs(J[i]).max() for i in ids) or 1.0
    for i in ids:
        feats = top_features(J[i], k)
        cells = "".join(
            f'<tr><td class="feat">{esc(OBS_LABEL[fi])}</td>'
            f'<td class="num">{v:+.3f}</td><td>{bar(v, vmax)}</td></tr>'
            for fi, v in feats
        )
        rows.append(
            f'<div class="macrocard"><h4>{esc(MACRO_LABELS[i])} '
            f'<span class="muted">(legal in {int(cnt[i])} states)</span></h4>'
            f'<table class="attr">{cells}</table></div>'
        )
    return f'<h3>{esc(phase)}</h3><div class="macrogrid">' + "".join(rows) + "</div>"


def section_attention(J: np.ndarray, cnt: np.ndarray, k: int) -> str:
    """For each macro, which observation sections it is most sensitive to."""
    rows = []
    for i in range(N_ACTIONS):
        if cnt[i] < 20:
            continue
        sums = {}
        for name, a, b in SECTION_BOUNDS:
            sums[name] = float(np.abs(J[i, a:b]).sum())
        top = sorted(sums.items(), key=lambda kv: -kv[1])[:k]
        tot = sum(sums.values()) or 1.0
        chips = " ".join(
            f'<span class="chip">{esc(n)} <b>{100 * v / tot:.0f}%</b></span>'
            for n, v in top
        )
        rows.append(f'<tr><td>{esc(MACRO_LABELS[i])}</td><td>{chips}</td></tr>')
    return ('<table class="sectbl"><tr><th>Macro</th>'
            '<th>Most-attended observation sections (share of |sensitivity|)</th></tr>'
            + "".join(rows) + "</table>")


def hidden_fingerprints(pol: Policy, cor: dict, push: np.ndarray,
                        z1: np.ndarray, facts: dict, n_units: int, k: int) -> str:
    """Exploratory per-unit cards. Rank units by downstream importance."""
    importance = np.abs(push).sum(axis=0) * z1.std(axis=0)
    order = np.argsort(-importance)[:n_units]
    fact_names = list(facts.keys())
    fact_mat = np.stack([facts[n] for n in fact_names], axis=1)
    cards = []
    for j in order:
        j = int(j)
        w = pol.l1_w[j]
        reads = top_features(w, k)
        reads_html = " ".join(
            f'<span class="chip {"pos" if v >= 0 else "neg"}">{esc(OBS_LABEL[fi])} '
            f'{v:+.2f}</span>' for fi, v in reads
        )
        pv = push[:, j]
        po = np.argsort(-np.abs(pv))[:k]
        push_html = " ".join(
            f'<span class="chip {"pos" if pv[a] >= 0 else "neg"}">{esc(MACRO_LABELS[a])} '
            f'{pv[a]:+.2f}</span>' for a in po
        )
        fc = [(fact_names[c], corr(z1[:, j], fact_mat[:, c])) for c in range(len(fact_names))]
        fc.sort(key=lambda t: -abs(t[1]))
        fires_html = " ".join(
            f'<span class="chip {"pos" if v >= 0 else "neg"}">{esc(n)} '
            f'{v:+.2f}</span>' for n, v in fc[:4] if abs(v) > 0.08
        ) or '<span class="muted">no strong game-fact correlate</span>'
        sat = float((np.abs(z1[:, j]) > 0.9).mean())
        cards.append(
            f'<div class="unitcard"><h4>hidden unit {j} '
            f'<span class="muted">(|activation| std {z1[:, j].std():.2f}, '
            f'{100 * sat:.0f}% saturated)</span></h4>'
            f'<div class="urow"><span class="ulab">reads</span>{reads_html}</div>'
            f'<div class="urow"><span class="ulab">pushes</span>{push_html}</div>'
            f'<div class="urow"><span class="ulab">fires when</span>{fires_html}</div>'
            f'</div>'
        )
    return '<div class="unitgrid">' + "".join(cards) + "</div>"


# --------------------------------------------------------------------------- #
# Q2 — comparison
# --------------------------------------------------------------------------- #
def behavioral_diff(pa: Policy, pb: Policy, X: np.ndarray, M: np.ndarray,
                    facts: dict) -> dict:
    La = pa.forward(X)["logits"]
    Lb = pb.forward(X)["logits"]
    Pa = masked_softmax(La, M)
    Pb = masked_softmax(Lb, M)
    # Per-macro mean probability shift over states where the macro is legal.
    legal = M > 0
    dp = Pa - Pb
    per_macro = []
    for i in range(N_ACTIONS):
        sel = legal[:, i]
        if sel.sum() < 20:
            continue
        per_macro.append((i, float(dp[sel, i].mean()), int(sel.sum())))
    per_macro.sort(key=lambda t: -abs(t[1]))
    # KL(A||B) per state over legal support, overall and per phase.
    eps = 1e-9
    kl = (Pa * (np.log(Pa + eps) - np.log(Pb + eps))).sum(axis=1)
    phase = facts["phase_id"]
    phase_kl = {}
    for pid, nm in [(2, "Auction"), (5, "BuyResources"), (6, "BuildCities"), (3, "DiscardPlant")]:
        m = phase == pid
        if m.sum() > 0:
            phase_kl[nm] = (float(kl[m].mean()), int(m.sum()))
    return {"per_macro": per_macro, "kl_overall": float(kl.mean()), "phase_kl": phase_kl}


def render_behavioral(bd: dict, la: str, lb: str) -> str:
    pm = bd["per_macro"]
    vmax = max((abs(v) for _, v, _ in pm), default=1.0)
    rows = "".join(
        f'<tr><td>{esc(MACRO_LABELS[i])}</td><td class="muted">{esc(MACRO_PHASE[i])}</td>'
        f'<td class="num">{v:+.3f}</td><td>{bar(v, vmax)}</td>'
        f'<td class="num muted">{n}</td></tr>'
        for i, v, n in pm[:18]
    )
    pk = "".join(
        f'<tr><td>{esc(nm)}</td><td class="num">{v:.4f}</td>'
        f'<td class="num muted">{n}</td></tr>'
        for nm, (v, n) in bd["phase_kl"].items()
    )
    return (
        f'<p>Mean masked-softmax probability shift (<b>{esc(la)}</b> − <b>{esc(lb)}</b>) '
        f'per macro, over states where the macro is legal. Positive ⇒ <b>{esc(la)}</b> '
        f'favours it more.</p>'
        f'<table class="cmptbl"><tr><th>Macro</th><th>Phase</th><th>Δprob</th>'
        f'<th></th><th>n</th></tr>{rows}</table>'
        f'<h4>Divergence by phase — KL({esc(la)} ‖ {esc(lb)})</h4>'
        f'<table class="cmptbl"><tr><th>Phase</th><th>mean KL</th><th>n</th></tr>{pk}'
        f'<tr><td><b>overall</b></td><td class="num"><b>{bd["kl_overall"]:.4f}</b></td>'
        f'<td></td></tr></table>'
    )


def attribution_diff(Ja: np.ndarray, Jb: np.ndarray, cnt: np.ndarray,
                     la: str, lb: str, k: int) -> str:
    dJ = Ja - Jb
    # Top (macro, section) cells by |Δ sum|.
    cells = []
    for i in range(N_ACTIONS):
        if cnt[i] < 20:
            continue
        for name, a, b in SECTION_BOUNDS:
            cells.append((i, name, float(dJ[i, a:b].sum()), float(np.abs(dJ[i, a:b]).sum())))
    cells.sort(key=lambda c: -c[3])
    vmax = max((abs(c[2]) for c in cells[:20]), default=1.0)
    top = "".join(
        f'<tr><td>{esc(MACRO_LABELS[i])}</td><td>{esc(name)}</td>'
        f'<td class="num">{s:+.3f}</td><td>{bar(s, vmax)}</td></tr>'
        for i, name, s, _ in cells[:20]
    )
    # Per-macro single most-changed feature.
    feat = []
    for i in range(N_ACTIONS):
        if cnt[i] < 20:
            continue
        fi = int(np.argmax(np.abs(dJ[i])))
        feat.append((i, fi, float(dJ[i, fi])))
    feat.sort(key=lambda t: -abs(t[2]))
    fmax = max((abs(v) for _, _, v in feat), default=1.0)
    fr = "".join(
        f'<tr><td>{esc(MACRO_LABELS[i])}</td><td class="feat">{esc(OBS_LABEL[fi])}</td>'
        f'<td class="num">{v:+.3f}</td><td>{bar(v, fmax)}</td></tr>'
        for i, fi, v in feat[:14]
    )
    return (
        f'<p>Change in local sensitivity ∂logit/∂input, <b>{esc(la)}</b> − <b>{esc(lb)}</b>, '
        f'summed per observation section. This is the crisp "what it learned": a positive '
        f'value means <b>{esc(la)}</b> made that macro <i>more</i> sensitive to that group of '
        f'game features.</p>'
        f'<h4>Biggest (macro, section) sensitivity shifts</h4>'
        f'<table class="cmptbl"><tr><th>Macro</th><th>Obs section</th><th>Δ (signed sum)</th>'
        f'<th></th></tr>{top}</table>'
        f'<h4>Each macro\'s single most-changed input feature</h4>'
        f'<table class="cmptbl"><tr><th>Macro</th><th>Feature</th><th>ΔJ</th><th></th></tr>{fr}'
        f'</table>'
    )


def unit_drift(pa: Policy, pb: Policy, za: np.ndarray, zb: np.ndarray,
               pusha: np.ndarray, pushb: np.ndarray, n_units: int, k: int) -> str:
    if pa.hidden != pb.hidden:
        return '<p class="muted">Hidden widths differ; per-unit comparison N/A.</p>'
    wdrift = np.linalg.norm(pa.l1_w - pb.l1_w, axis=1)
    adrift = np.abs(za - zb).mean(axis=0)
    score = wdrift * (adrift + 1e-6)
    order = np.argsort(-score)[:n_units]
    rows = []
    for j in order:
        j = int(j)
        ra = top_features(pa.l1_w[j], k)
        rb = top_features(pb.l1_w[j], k)
        pa_top = np.argsort(-np.abs(pusha[:, j]))[:3]
        pb_top = np.argsort(-np.abs(pushb[:, j]))[:3]
        reads_a = ", ".join(f"{esc(OBS_LABEL[fi].split(': ', 1)[-1])} {v:+.2f}" for fi, v in ra)
        reads_b = ", ".join(f"{esc(OBS_LABEL[fi].split(': ', 1)[-1])} {v:+.2f}" for fi, v in rb)
        push_a = ", ".join(f"{esc(MACRO_LABELS[a])} {pusha[a, j]:+.2f}" for a in pa_top)
        push_b = ", ".join(f"{esc(MACRO_LABELS[a])} {pushb[a, j]:+.2f}" for a in pb_top)
        rows.append(
            f'<div class="unitcard"><h4>hidden unit {j} '
            f'<span class="muted">(‖Δw‖ {wdrift[j]:.2f}, mean |Δact| {adrift[j]:.3f})</span></h4>'
            f'<table class="drift"><tr><th></th><th>A</th><th>B</th></tr>'
            f'<tr><td class="ulab">reads</td><td>{reads_a}</td><td>{reads_b}</td></tr>'
            f'<tr><td class="ulab">pushes</td><td>{push_a}</td><td>{push_b}</td></tr>'
            f'</table></div>'
        )
    return '<div class="unitgrid">' + "".join(rows) + "</div>"


# --------------------------------------------------------------------------- #
CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
  background:#12141a;color:#e6e8ee}
.wrap{max-width:1180px;margin:0 auto;padding:28px}
h1{font-size:26px;margin:0 0 4px} h2{border-bottom:1px solid #2a2e38;padding-bottom:6px;
  margin-top:40px;font-size:20px} h3{margin:24px 0 8px;color:#9fd0ff}
h4{margin:14px 0 6px;font-size:14px}
.muted{color:#8b93a7;font-weight:400} .num{text-align:right;font-variant-numeric:tabular-nums}
.feat{color:#cdd3e0}
.note{background:#1b1f29;border:1px solid #2a2e38;border-left:3px solid #e0a458;
  padding:12px 16px;border-radius:6px;margin:16px 0}
.meta{background:#1b1f29;border:1px solid #2a2e38;border-radius:6px;padding:12px 16px;margin:12px 0}
.meta code{color:#9fd0ff}
table{border-collapse:collapse;width:100%} td,th{padding:3px 8px;text-align:left}
th{color:#8b93a7;font-weight:600;border-bottom:1px solid #2a2e38}
.attr td{border:0;padding:2px 6px}
.macrogrid,.unitgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.macrocard,.unitcard{background:#1b1f29;border:1px solid #2a2e38;border-radius:6px;padding:10px 12px}
.barbox{position:relative;display:inline-block;height:11px;vertical-align:middle;
  background:#0e1016;border-radius:2px}
.bartick{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#3a3f4c}
.bar{position:absolute;top:1px;height:9px;border-radius:2px}
.chip{display:inline-block;background:#232734;border:1px solid #2f3543;border-radius:10px;
  padding:1px 7px;margin:2px 2px;font-size:12px}
.chip.pos{border-color:#5a2f3a} .chip.neg{border-color:#274152}
.urow{margin:4px 0} .ulab{display:inline-block;width:74px;color:#8b93a7;font-size:12px;
  vertical-align:top}
.sectbl td{border-bottom:1px solid #20242e}
.cmptbl th,.cmptbl td{border-bottom:1px solid #20242e}
.drift td{border-bottom:1px solid #20242e;font-size:12px} .drift th{font-size:12px}
"""


def build_report(ctx: dict) -> str:
    p = []
    p.append(f'<div class="wrap"><h1>Policy network — game-term analysis</h1>')
    p.append(f'<p class="muted">Expert MLP: {OBS_SIZE} → {ctx["pa"].hidden} → tanh → '
             f'{ctx["pa"].hidden} → tanh → {N_ACTIONS} macros.</p>')
    p.append('<div class="meta">')
    p.append(f'<div>Policy A: <code>{esc(ctx["la"])}</code> — {esc(ctx["pa"].path)}</div>')
    if ctx.get("pb"):
        p.append(f'<div>Policy B: <code>{esc(ctx["lb"])}</code> — {esc(ctx["pb"].path)}</div>')
    p.append(f'<div>Corpus: {len(ctx["X"])} on-policy decision states from '
             f'{ctx["games"]} games vs {ctx["difficulty"]} bots (driven by A).</div>')
    p.append(f'<div class="muted">Loader check — {esc(ctx["golden"])}. '
             f'{esc(ctx["fdcheck"])}.</div>')
    p.append('</div>')

    # Q1
    p.append('<h2>Q1 · What the network computes, in game terms</h2>')
    p.append('<p>For each macro, the observation features whose value most changes that '
             'macro\'s logit, averaged over states where the macro is legal (exact local '
             'Jacobian). <span style="color:#d1495b">Red = raises</span> the logit as the '
             'feature increases; <span style="color:#3a7ca5">blue = lowers</span> it.</p>')
    for phase in ["Auction", "BuyResources", "BuildCities", "DiscardPlant"]:
        p.append(macro_attr_table(ctx["Ja"], ctx["cnt"], ctx["k"], phase))
    p.append('<h3>Where each macro looks</h3>')
    p.append(section_attention(ctx["Ja"], ctx["cnt"], 4))

    # Q1 appendix — hidden units
    p.append('<h2>Appendix · Hidden-unit fingerprints <span class="muted">(exploratory)</span></h2>')
    p.append('<div class="note"><b>Read with caution.</b> Individual tanh units in a dense '
             'network are frequently <i>polysemantic</i> — one unit can encode several unrelated '
             'things — so these fingerprints are suggestive, not definitions. The rigorous '
             'game-term claims are the input→macro attributions above. Units are ranked by '
             'downstream importance (‖push‖ × activation spread).</div>')
    p.append(hidden_fingerprints(ctx["pa"], ctx["cor"], ctx["pusha"], ctx["za"],
                                 ctx["facts"], ctx["units"], ctx["k"]))

    # Q2
    if ctx.get("pb"):
        p.append(f'<h2>Q2 · What {esc(ctx["la"])} learned relative to {esc(ctx["lb"])}</h2>')
        p.append('<h3>Behavioral difference</h3>')
        p.append(render_behavioral(ctx["bd"], ctx["la"], ctx["lb"]))
        p.append('<h3>Sensitivity difference (what it learned)</h3>')
        p.append(attribution_diff(ctx["Ja"], ctx["Jb"], ctx["cnt"], ctx["la"], ctx["lb"], ctx["k"]))
        p.append('<h3>Per-unit drift <span class="muted">(exploratory; valid only for a '
                 'warm-started lineage)</span></h3>')
        p.append('<div class="note">Comparing hidden units by index is meaningful <b>only</b> '
                 'because these two policies share initialization (one warm-started from the '
                 'other). For two independently trained networks the hidden layer is the same '
                 'function up to a permutation of units, and this section would be noise.</div>')
        p.append(unit_drift(ctx["pa"], ctx["pb"], ctx["za"], ctx["zb"],
                            ctx["pusha"], ctx["pushb"], ctx["units"], ctx["k"]))

    p.append('</div>')
    return f'<!doctype html><html><head><meta charset="utf-8">' \
           f'<title>Policy analysis</title><style>{CSS}</style></head><body>' \
           + "".join(p) + '</body></html>'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="Policy A .bin (PGRLPOL6). Drives the corpus.")
    ap.add_argument("--compare", help="Policy B .bin for Q2 comparison.")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--bot-difficulty", default="hard", choices=["easy", "normal", "hard"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-states", type=int, default=12000)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--top", type=int, default=6, help="Top-k features/units to show.")
    ap.add_argument("--units", type=int, default=16, help="Hidden-unit cards to show.")
    ap.add_argument("--out", default=str(REPO / "python" / "runs" / "analysis" / "report.html"))
    args = ap.parse_args()

    pa = Policy(args.policy)
    pb = Policy(args.compare) if args.compare else None
    la = args.label_a if args.label_a != "A" else Path(args.policy).stem
    lb = args.label_b if args.label_b != "B" else (Path(args.compare).stem if pb else "B")

    golden = golden_check(pa)
    print(f"[loader] {golden}")
    if pb and pa.hidden != pb.hidden:
        print(f"[warn] hidden widths differ ({pa.hidden} vs {pb.hidden}); "
              "per-unit comparison will be skipped.")

    print(f"[corpus] driving {args.games} games vs {args.bot_difficulty} bots "
          f"(driver = {la})...")
    cp = build_corpus(pa, args.games, args.num_players, args.seed,
                      args.bot_difficulty, args.max_states)
    X, M, facts = cp["X"], cp["M"], cp["facts"]
    print(f"[corpus] {len(X)} decision states collected")

    Ja, cnt = mean_jacobian(pa, X, M, args.chunk)
    fdcheck = finite_diff_check(pa, X, Ja)
    print(f"[jacobian] {fdcheck}")
    pusha = hidden_push(pa, X, args.chunk)
    za = pa.forward(X)["z1"]

    ctx = dict(pa=pa, pb=pb, la=la, lb=lb, X=X, M=M, facts=facts,
               Ja=Ja, cnt=cnt, pusha=pusha, za=za, cor={},
               games=args.games, difficulty=args.bot_difficulty,
               golden=golden, fdcheck=fdcheck, k=args.top, units=args.units)

    if pb:
        Jb, _ = mean_jacobian(pb, X, M, args.chunk)
        pushb = hidden_push(pb, X, args.chunk)
        zb = pb.forward(X)["z1"]
        bd = behavioral_diff(pa, pb, X, M, facts)
        ctx.update(Jb=Jb, pushb=pushb, zb=zb, bd=bd)
        print(f"[compare] overall KL({la}||{lb}) = {bd['kl_overall']:.4f}")
        print(f"[compare] biggest behavior shifts:")
        for i, v, n in bd["per_macro"][:6]:
            print(f"           {MACRO_LABELS[i]:<24} Δprob {v:+.3f}  (n={n})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(ctx))
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
