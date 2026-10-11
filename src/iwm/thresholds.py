"""Provable but *classical* active-threshold identification baseline.

The environment is a deterministic, one-dimensional, monotone event model.
It is not MuJoCo, not an RGB video generator, and not an original ML method.
"""

from dataclasses import dataclass
from typing import Protocol


class Probe(Protocol):
    def __call__(self, action: float) -> bool:
        """Return whether a mode transition occurs for this action."""


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lower <= self.upper <= 1.0):
            raise ValueError("Require 0 <= lower <= upper <= 1.")

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def predict(self, action: float) -> bool | None:
        """Certain event, certain no-event, or unresolved.

        Assumes event(action) = (action >= threshold) for an unknown
        threshold c in [lower, upper]. Endpoints are treated
        conservatively; equality at the boundary is unresolved.
        """
        if not 0.0 <= action <= 1.0:
            raise ValueError("Action outside [0, 1].")
        if action < self.lower:
            return False
        if action >= self.upper:
            return True
        return None


def update(interval: Interval, action: float, event: bool) -> Interval:
    """Update a compatible interval with a physical probe observation.

    event=True gives c <= action; event=False gives c > action.
    This closed interval conservatively retains the excluded endpoint.
    """
    if not interval.lower <= action <= interval.upper:
        raise ValueError("Probe must lie in current uncertainty interval.")
    if event:
        return Interval(interval.lower, action)
    return Interval(action, interval.upper)


def active_bisection(probe: Probe, budget: int, start: Interval = Interval(0, 1)) -> Interval:
    """Minimax-optimal interval halving for deterministic monotone events."""
    if budget < 0:
        raise ValueError("Budget must be nonnegative.")
    interval = start
    for _ in range(budget):
        midpoint = (interval.lower + interval.upper) / 2
        interval = update(interval, midpoint, bool(probe(midpoint)))
    return interval


def true_event(threshold: float, action: float) -> bool:
    if not (0 <= threshold <= 1 and 0 <= action <= 1):
        raise ValueError("Threshold and action must be within [0, 1].")
    return action >= threshold


def demonstrate(threshold: float = 0.57, budget: int = 5) -> None:
    interval = active_bisection(lambda a: true_event(threshold, a), budget)
    print(f"threshold={threshold:.3f}; interval=[{interval.lower:.5f}, "
          f"{interval.upper:.5f}]; width={interval.width:.5f}")


if __name__ == "__main__":
    demonstrate()
