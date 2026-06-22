"""AlphaZero training stack for Power Grid.

A from-scratch implementation (structured like alpha-zero-general: Game
adapter / NNet wrapper / MCTS / Coach) that replaces the PettingZoo+PPO
training stack in `python/` with MCTS-guided self-play. Reuses the Rust game
engine and observation/action encoding via `powergrid_py` and
`powergrid_env.constants` — see `alphazero/README.md` for the runbook and
`CLAUDE.md` for how this fits into the wider workspace.
"""
