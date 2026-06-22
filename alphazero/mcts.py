"""Multiplayer masked PUCT MCTS.

Each node is backed by a forked `PowerGridGame` (a real, independent copy of
the Rust engine state) rather than a transposition-table entry keyed by a
canonical board string — Power Grid's state is too large/hidden-info-bearing
to canonicalize cheaply, and forking is cheap (a `GameState::clone`), so a
plain search tree of nodes is simpler and just as correct.

Because Power Grid is strictly single-actor-per-turn (never simultaneous),
every node has one well-defined "player to move", so the usual N-player
AlphaZero backup is simple: the network's leaf value is a vector relative to
the leaf's mover, converted once to an absolute `{player_id: value}` dict
(see `game.to_absolute_dict`), and that *same* dict is propagated unchanged
up the path — each ancestor just reads its own mover's entry out of it when
updating its Q/N statistics. No per-player Q arrays are needed.
"""

from __future__ import annotations

import math

import numpy as np
from powergrid_env.constants import N_ACTIONS

from .config import AZConfig
from .game import PowerGridGame, to_absolute_dict
from .network import NNetWrapper


class Node:
    __slots__ = ("game", "parent", "prior", "children", "N", "W", "is_expanded", "actor")

    def __init__(self, game: PowerGridGame | None, parent: Node | None = None, prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.children: dict[int, Node] = {}
        self.N = 0
        self.W = 0.0
        self.is_expanded = False
        self.actor: str | None = None

    def q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class MCTS:
    def __init__(self, nnet: NNetWrapper, cfg: AZConfig):
        self.nnet = nnet
        self.cfg = cfg

    def get_action_probs(
        self, root_game: PowerGridGame, temp: float = 1.0, add_noise: bool = True
    ) -> np.ndarray:
        """Run `cfg.num_sims` simulations from `root_game` and return the
        visit-count distribution over the 143 actions (masked to legal
        moves), tempered by `temp` (0 = greedy one-hot)."""
        root = Node(root_game)
        self._expand(root)
        if not root.children:
            # Terminal root: nothing to search. Caller shouldn't normally hit
            # this, but return a degenerate (all-zero) distribution rather
            # than dividing by zero below.
            return np.zeros(N_ACTIONS, dtype=np.float32)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.cfg.num_sims):
            self._simulate(root)

        counts = np.zeros(N_ACTIONS, dtype=np.float64)
        for a, child in root.children.items():
            counts[a] = child.N

        if temp == 0:
            best = np.flatnonzero(counts == counts.max())
            probs = np.zeros(N_ACTIONS, dtype=np.float32)
            probs[np.random.choice(best)] = 1.0
            return probs

        counts = counts ** (1.0 / temp)
        total = counts.sum()
        if total <= 0:
            # All children unvisited (num_sims == 0): fall back to priors.
            probs = np.array([root.children[a].prior if a in root.children else 0.0
                               for a in range(N_ACTIONS)], dtype=np.float64)
            total = probs.sum()
            return (probs / total).astype(np.float32)
        return (counts / total).astype(np.float32)

    # -- internals --------------------------------------------------------------

    def _expand(self, node: Node) -> dict[str, float]:
        """Evaluate `node.game` with the network, set priors over legal
        actions, and return the absolute value dict. Terminal nodes
        short-circuit to the true outcome and get no children."""
        game = node.game
        node.is_expanded = True
        if game.is_terminal():
            return game.outcome()

        actor = game.current_player()
        node.actor = actor
        obs = game.observation()
        mask = game.action_mask()
        probs, value_rel = self.nnet.predict(obs, mask)

        for a in np.flatnonzero(mask):
            node.children[int(a)] = Node(None, parent=node, prior=float(probs[a]))

        return to_absolute_dict(game.player_ids(), actor, value_rel)

    def _add_dirichlet_noise(self, root: Node) -> None:
        actions = list(root.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(actions))
        eps = self.cfg.dirichlet_eps
        for a, n in zip(actions, noise):
            child = root.children[a]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    def _simulate(self, root: Node) -> None:
        node = root
        path: list[tuple[Node, int]] = []

        # Selection: descend the existing tree until we reach a node with no
        # children — either a freshly-forked, not-yet-expanded node, or an
        # already-expanded terminal one (terminal nodes never get children).
        # That node becomes this simulation's leaf.
        while node.children:
            a = self._select_action(node)
            child = node.children[a]
            if child.game is None:
                child.game = node.game.fork()
                child.game.apply(a)
            path.append((node, a))
            node = child

        value_dict = self._expand(node)
        self._backup(path, value_dict)

    def _select_action(self, node: Node) -> int:
        sqrt_n = math.sqrt(max(1, node.N))
        best_score, best_a = -float("inf"), next(iter(node.children))
        for a, child in node.children.items():
            u = self.cfg.cpuct * child.prior * sqrt_n / (1 + child.N)
            score = child.q() + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _backup(self, path: list[tuple[Node, int]], value_dict: dict[str, float]) -> None:
        for node, a in reversed(path):
            v = value_dict[node.actor]
            child = node.children[a]
            child.N += 1
            child.W += v
            node.N += 1
