"""Regression checks for trial-boundary and schema safety."""

import pandas as pd
import pytest

from data_conduit.actual.datastructures import TrialSpec, parse_trials, within


def _event_frame(rows: list[tuple[float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Event": [event for _, event in rows]},
        index=pd.Index([time for time, _ in rows], name="Time"),
    )


def test_parse_trials_rejects_non_monotonic_event_times() -> None:
    events = _event_frame([(0.0, "Start"), (2.0, "Close one"), (1.0, "Close two")])

    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        parse_trials(events, TrialSpec(closes_trial="Close", session_start="Start"))


def test_parse_trials_rejects_an_inverted_buffered_window() -> None:
    events = _event_frame([(0.0, "Start"), (2.0, "Close one"), (2.5, "Close two")])

    with pytest.raises(ValueError, match="after its closing event"):
        parse_trials(
            events,
            TrialSpec(closes_trial="Close", session_start="Start", start_buffer=1.0),
        )


def test_contiguous_trials_do_not_share_the_previous_closing_event() -> None:
    events = _event_frame(
        [
            (0.0, "Start"),
            (1.0, "Close one"),
            (1.5, "Middle"),
            (2.0, "Close two"),
        ],
    )
    spec = TrialSpec(
        closes_trial="Close",
        session_start="Start",
        fields=lambda window, _closing: {"window_events": window["Event"].tolist()},
    )

    trials = parse_trials(events, spec)

    assert trials.loc[0, "window_events"] == ["Start", "Close one"]
    assert trials.loc[1, "window_events"] == ["Middle", "Close two"]
    assert trials.loc[1, "start_time"] == 1.0


def test_trial_callbacks_cannot_overwrite_structural_columns() -> None:
    events = _event_frame([(0.0, "Start"), (1.0, "Close")])
    spec = TrialSpec(
        closes_trial="Close",
        session_start="Start",
        fields=lambda _window, _closing: {"start_time": 999.0},
    )

    with pytest.raises(ValueError, match="reserved trial columns"):
        parse_trials(events, spec)


def test_segments_must_reference_existing_columns() -> None:
    events = _event_frame([(0.0, "Start"), (1.0, "Close")])
    spec = TrialSpec(
        closes_trial="Close",
        session_start="Start",
        segments={"broken": ("start_time", "typo_end")},
    )

    with pytest.raises(KeyError, match="typo_end"):
        parse_trials(events, spec)


def test_within_rejects_inverted_bounds() -> None:
    events = _event_frame([(0.0, "Start")])

    with pytest.raises(ValueError, match="start must be less than or equal to end"):
        within(events, 2.0, 1.0)
