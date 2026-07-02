"""Final-state stats helpers shared by evaluation and placement rewards."""


def learner_stats(state: dict, learner_id: str, winner_id: str | None = None) -> dict:
    """Final-state stats for the learner, plus its placement among all players.

    Placement is ranked by (cities powered last round, cities connected).
    Opponent money is hidden from the view, so the money tiebreak can't be
    reconstructed for opponent-vs-opponent ties; those tied players share the
    better rank. The engine's authoritative winner_id (which does use the
    hidden-money tiebreak) is treated as strictly ahead of everyone else, so
    rank 1 always agrees with the actual win/loss outcome.
    """
    city_owners = state["city_owners"]

    def cities_of(pid: str) -> int:
        return sum(pid in owners for owners in city_owners.values())

    scores = {p["id"]: (p["last_cities_powered"], cities_of(p["id"]))
              for p in state["players"]}
    me = next(p for p in state["players"] if p["id"] == learner_id)
    me_score = scores[learner_id]
    if winner_id == learner_id:
        rank = 1
    else:
        rank = 1 + sum(
            (s > me_score) or (pid == winner_id)
            for pid, s in scores.items() if pid != learner_id
        )
    return {
        "money": me["money"],
        "powered": me["last_cities_powered"],
        "plants": [pl["number"] for pl in me["plants"]],
        "capacity": sum(pl["cities"] for pl in me["plants"]),
        "rank": rank,
        "round": state["round"],
    }
