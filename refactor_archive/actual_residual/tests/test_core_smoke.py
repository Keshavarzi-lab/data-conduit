"""Focused smoke checks for foundational staged utilities."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data_conduit.actual.core.datetime import name_datetime_reader_for
from data_conduit.actual.core.globaltimes import create_global_clock, index_map_util
from data_conduit.actual.core.segment import get_segment
from data_conduit.actual.core.sync.ttl import TTLSyncModel, build_pulse_table
from data_conduit.actual.core.utils import _concat_split_dataframes, _matches_selector


def test_name_datetime_reader_extracts_a_token_from_a_session_name() -> None:
    """A named convention should compose with regular-expression extraction."""
    reader = name_datetime_reader_for(
        "iso_date",
        pattern=r"(\d{4}-\d{2}-\d{2})",
    )

    assert reader("subject_2026-08-19_session") == datetime(2026, 8, 19)


@pytest.mark.parametrize(
    ("start", "end", "step"),
    [(0.0, 10.0, -1.0), (10.0, 0.0, 1.0)],
)
def test_global_clock_rejects_a_step_pointing_away_from_the_end(
    start: float,
    end: float,
    step: float,
) -> None:
    """An interval with the wrong sign must not produce a two-point clock."""
    with pytest.raises(ValueError, match="must point"):
        create_global_clock(start, end, step)


@pytest.mark.parametrize(
    ("include_end_time", "expected"),
    [(True, [1.0]), (False, [])],
)
def test_global_clock_handles_equal_bounds_without_duplicate_samples(
    include_end_time: bool,
    expected: list[float],
) -> None:
    """Equal bounds contain at most one endpoint sample."""
    result = create_global_clock(1.0, 1.0, 0.25, include_end_time=include_end_time)

    np.testing.assert_array_equal(result, expected)


def test_global_clock_supports_a_descending_timebase() -> None:
    """A negative step remains valid when the end precedes the start."""
    result = create_global_clock(2.0, 0.0, -1.0)

    np.testing.assert_array_equal(result, [2.0, 1.0, 0.0])


def test_index_map_rejects_non_monotonic_stream_times() -> None:
    """Search-based matching must not silently operate on an unsorted stream."""
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        index_map_util(
            np.array([0.0, 1.0, 2.0]),
            np.array([2.0, 1.0, 0.0]),
        )


def test_index_map_accepts_duplicate_stream_times() -> None:
    """Non-decreasing streams may contain repeated timestamps."""
    result = index_map_util(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        match_type="exact",
    )

    np.testing.assert_array_equal(
        result["index_position_array"]["stream_index"],
        [0, 1],
    )


@pytest.mark.parametrize(
    "selector",
    [["keep"], ("keep",), {"keep"}, frozenset({"keep"})],
)
def test_selector_accepts_each_supported_collection(selector: object) -> None:
    """Every collection type in the selector contract should use membership."""
    assert _matches_selector("keep", selector) is True
    assert _matches_selector("drop", selector) is False


@pytest.mark.parametrize("selector", [{"keep": True}, range(3)])
def test_selector_rejects_unsupported_collection_types(selector: object) -> None:
    """Mappings and arbitrary collections should not be silently interpreted."""
    with pytest.raises(ValueError, match="Invalid selector type"):
        _matches_selector("keep", selector)


def test_split_dataframes_keep_filename_order_and_surface_rollbacks() -> None:
    """Cross-file clock resets remain visible instead of being globally sorted."""
    pieces = {
        "part_02": pd.DataFrame({"value": [2]}, index=[0.0]),
        "part_01": pd.DataFrame({"value": [1]}, index=[1.0]),
    }

    with pytest.warns(UserWarning, match="clock appears to have reset"):
        combined = _concat_split_dataframes(pieces)

    np.testing.assert_array_equal(combined["value"], [1, 2])
    np.testing.assert_array_equal(combined.index, [1.0, 0.0])


def test_get_segment_includes_both_window_boundaries() -> None:
    """Segmentation should retain samples exactly at start and end."""
    index_map = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "sample": [10, 11, 12, 13],
        }
    )

    result = get_segment(index_map, 1.0, 2.0, "time", "sample")

    np.testing.assert_array_equal(result, [11, 12])


def test_ttl_model_maps_target_clock_to_reference_clock() -> None:
    """The ported TTL utilities should preserve their basic conversion contract."""
    target = build_pulse_table([0.0, 1.0, 2.0], [0.1, 1.1, 2.1])
    reference = pd.DataFrame(
        {
            "Start": 2.0 * target["Start"] + 5.0,
            "End": 2.0 * target["End"] + 5.0,
        }
    )

    model = TTLSyncModel.fit(reference, target)

    np.testing.assert_allclose(model.transform([3.0]), [11.0])
    assert model.r2 == 1.0
