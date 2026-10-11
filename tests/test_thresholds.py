import random

import pytest

from iwm.thresholds import Interval, active_bisection, true_event, update


@pytest.mark.parametrize("budget", [0, 1, 2, 4, 8, 12])
def test_bisection_contains_threshold_and_halves_width(budget):
    rng = random.Random(20261011)
    for _ in range(200):
        threshold = rng.random()
        result = active_bisection(lambda a: true_event(threshold, a), budget)
        assert result.lower <= threshold <= result.upper
        assert abs(result.width - 2 ** (-budget)) < 1e-12


def test_certified_predictions_never_make_a_wrong_claim():
    rng = random.Random(117)
    for _ in range(200):
        threshold = rng.random()
        interval = active_bisection(lambda a: true_event(threshold, a), 4)
        for i in range(101):
            action = i / 100
            answer = interval.predict(action)
            if answer is not None:
                assert answer == true_event(threshold, action)


def test_unobserved_worlds_are_indistinguishable():
    for action in [0.0, 0.1, 0.2, 0.4]:
        assert true_event(0.5, action) == true_event(0.6, action)
    assert true_event(0.5, 0.55) != true_event(0.6, 0.55)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        Interval(-0.1, 0.5)
    with pytest.raises(ValueError):
        active_bisection(lambda _: False, -1)
    with pytest.raises(ValueError):
        update(Interval(0.2, 0.6), 0.8, True)
    with pytest.raises(ValueError):
        Interval(0.2, 0.6).predict(2.0)
