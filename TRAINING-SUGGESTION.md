Power Grid is a good RL target: it's (nearly) perfect information, but it's 3–6 player, has huge branching in the build phase, and auctions create tricky interactive sub-games. Here's what I'd use:

**Core approach: self-play RL with PPO.** AlphaZero-style MCTS works beautifully for 2-player zero-sum games but gets awkward with 3+ players (the minimax assumption breaks). Model-free self-play with PPO or IMPALA, where each seat is controlled by the current or recent policy, is the standard choice for multiplayer board games. Use a shared network for all players with the observation from that player's perspective.

**Key design decisions:**

- **Factored action space.** Don't flatten everything into one giant action set. Power Grid has distinct phases (auction, resources, building), so use auto-regressive or hierarchical action heads: first pick plant/bid/pass, then bid amount; for building, pick cities one at a time with legality masking. Action masking for illegal moves is essential.
- **State encoding.** Flat vectors work fine for money, resource market, and plant market. The city network is a graph, so a small GNN over the map (nodes = cities with ownership features) helps, though a hand-crafted flat encoding of connection costs is a reasonable v1.
- **Reward.** Win = 1, loss = 0 (or placement-based, e.g. 1/0.3/0 for 1st/2nd/3rd). Sparse rewards work with enough games; if learning stalls, add small shaped rewards (cities powered, income rank) that you anneal away, since shaping can teach it to optimize the wrong thing (Power Grid famously punishes being the early leader).
- **Population-based training.** Pure self-play in multiplayer games can converge to weird equilibria or implicit collusion. Keep a league of past checkpoints plus a couple of scripted heuristic bots (money-hoarder, aggressive builder) and sample opponents from it, AlphaStar-style.

**Practical path:** write 1–2 decent scripted bots first (they're your evaluation baseline and league seeds), get your engine running headless at thousands of games/hour, then train PPO with masking. Expect representation and action-space engineering to be 80% of the work — the RL algorithm itself is the easy part.

