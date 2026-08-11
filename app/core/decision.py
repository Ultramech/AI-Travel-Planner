"""Deterministic ranking of transport options, and an audit of the LLM's pick.

Why this exists
---------------
This project asks a language model to compare travel options and recommend one.
Over a day of running it, the model:

  * multiplied a group fare by the party size a second time;
  * read a 24 h 30 m train as a thirty-minute hop because the duration was
    missing from its input;
  * recommended a flight from an airport in a different state without
    mentioning the four-hour drive to reach it;
  * asserted "no flights are available" for a route nothing had searched.

Every one of those is an *arithmetic or bookkeeping* error, not a language
error. So the division of labour here is explicit:

    the numbers are computed in this module;
    the model explains the choice in prose.

The model still writes the recommendation, because "an overnight train gets you
there rested and saves a hotel night" is genuinely a language task. But its pick
is now scored against the maths, and when the two disagree that disagreement is
recorded rather than silently trusted. A recommendation you cannot audit is
indistinguishable from one that is wrong.

Scoring
-------
Each candidate carries a real group cost, a real door-to-door time (including
any connecting leg and its check-in buffer) and a set of concrete penalties.
Cost and time are normalised across the candidate set — absolute rupees and
minutes are not comparable, and a fixed exchange rate between them would be a
made-up number. Weights come from the traveller's own stated budget level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: How much cost matters versus time, by the budget level the traveller chose.
#: "low" means they told us money is the constraint; "high" means time is.
_WEIGHTS = {
    "low":    {"cost": 0.70, "time": 0.20, "comfort": 0.10},
    "medium": {"cost": 0.45, "time": 0.40, "comfort": 0.15},
    "high":   {"cost": 0.20, "time": 0.60, "comfort": 0.20},
}

#: Penalties, in "comfort points" (0 = ideal, 1 = as bad as we score).
#: Each is something a traveller would actually complain about.
_PENALTY_SELF_TRANSFER = 0.40   # two bookings; a missed connection is your problem
_PENALTY_NIGHT_START = 0.25     # setting off before 05:00
_PENALTY_OVERNIGHT = 0.15       # arriving on a later calendar day
_PENALTY_ESTIMATED_PRICE = 0.10  # the fare is our estimate, not a quote
_PENALTY_PER_STOP = 0.10


@dataclass
class Candidate:
    """One complete way of getting there, priced and timed door to door."""

    mode: str
    label: str
    total_inr: Optional[float] = None
    total_minutes: Optional[int] = None
    depart_clock: Optional[str] = None
    arrive_date: Optional[str] = None
    arrive_clock: Optional[str] = None
    leg_cost_inr: float = 0.0
    stops: int = 0
    self_transfer: bool = False
    price_is_estimate: bool = False
    starts_at_night: bool = False
    arrives_next_day: bool = False
    notes: list[str] = field(default_factory=list)

    # Filled in by ``rank``.
    score: Optional[float] = None
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def comfort_penalty(self) -> float:
        penalty = 0.0
        if self.self_transfer:
            penalty += _PENALTY_SELF_TRANSFER
        if self.starts_at_night:
            penalty += _PENALTY_NIGHT_START
        if self.arrives_next_day:
            penalty += _PENALTY_OVERNIGHT
        if self.price_is_estimate:
            penalty += _PENALTY_ESTIMATED_PRICE
        penalty += _PENALTY_PER_STOP * max(0, self.stops)
        return min(penalty, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "total_inr": self.total_inr,
            "total_minutes": self.total_minutes,
            "connecting_leg_inr": self.leg_cost_inr or None,
            "departs": self.depart_clock,
            "arrives": (
                f"{self.arrive_date} {self.arrive_clock}"
                if self.arrive_date and self.arrive_clock
                else self.arrive_clock
            ),
            "stops": self.stops,
            "self_transfer": self.self_transfer,
            "price_is_estimate": self.price_is_estimate,
            "score": self.score,
            "score_breakdown": self.breakdown,
            "penalties": self.comfort_penalty,
            "notes": self.notes,
        }


def _normalise(values: list[float]) -> list[float]:
    """Map a list onto 0..1 where 0 is best (lowest). Equal values all score 0.

    Comparing rupees against minutes needs both on a common scale. Normalising
    within the candidate set avoids inventing a rupees-per-minute rate, which
    would be an arbitrary constant dressed up as a decision.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def rank(
    candidates: list[Candidate], budget_level: str = "medium"
) -> list[Candidate]:
    """Score and sort candidates, best first. Lower score is better."""
    usable = [c for c in candidates if c.total_inr is not None]
    if not usable:
        return list(candidates)

    weights = _WEIGHTS.get((budget_level or "medium").lower(), _WEIGHTS["medium"])

    costs = _normalise([c.total_inr for c in usable])            # type: ignore[misc]

    # An unknown duration takes the worst possible normalised time (1.0), not
    # the slowest *known* duration. Substituting the known maximum lets an
    # untimed option tie with the best one whenever it happens to be the only
    # timed candidate — a missing measurement must never flatter an option.
    known = [c.total_minutes for c in usable if c.total_minutes]
    known_norm = dict(zip(known, _normalise([float(m) for m in known])))
    times = [
        known_norm[c.total_minutes] if c.total_minutes else 1.0
        for c in usable
    ]

    for candidate, cost_n, time_n in zip(usable, costs, times):
        comfort = candidate.comfort_penalty
        candidate.breakdown = {
            "cost": round(cost_n * weights["cost"], 4),
            "time": round(time_n * weights["time"], 4),
            "comfort": round(comfort * weights["comfort"], 4),
        }
        candidate.score = round(sum(candidate.breakdown.values()), 4)

    unusable = [c for c in candidates if c.total_inr is None]
    usable.sort(key=lambda c: (c.score if c.score is not None else 1.0))
    return usable + unusable


def compare_with_model(
    ranked: list[Candidate], model_summary: str
) -> dict[str, Any]:
    """Did the model recommend what the arithmetic says it should have?

    Not a correctness check — a cheaper flight is not automatically the right
    answer, and the model may be weighing something real that the score does not
    capture. It is an *audit*: the disagreement is surfaced so a human can judge
    it, instead of the prose quietly overriding the numbers.
    """
    if not ranked or ranked[0].score is None:
        return {"checked": False, "reason": "nothing scorable to compare"}

    best = ranked[0]
    head = (model_summary or "")[:1200].lower()

    def mentioned(candidate: Candidate) -> bool:
        # Match on the identifying token — a train number, a route, a vehicle.
        token = candidate.label.split("·")[0].strip().lower()
        return bool(token) and token in head

    agreed = mentioned(best)
    picked = next((c for c in ranked if mentioned(c)), None)

    result: dict[str, Any] = {
        "checked": True,
        "agrees": agreed,
        "best_by_score": best.label,
        "best_score": best.score,
    }
    if not agreed:
        result["model_appears_to_pick"] = picked.label if picked else "unclear"
        if picked is not None and picked.score is not None:
            result["score_gap"] = round(picked.score - best.score, 4)
            result["cost_gap_inr"] = (
                round((picked.total_inr or 0) - (best.total_inr or 0), 2)
            )
        result["note"] = (
            "The written recommendation does not match the highest-scoring "
            "option. That can be legitimate — the model may be weighing "
            "something the score does not model — but it is worth a look."
        )
    return result
