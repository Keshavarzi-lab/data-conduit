"""Regression checks for corrected staged core edge cases."""

from collections.abc import Callable, Collection
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from data_conduit.actual.core.io import collect_dfs, collect_file_paths
from data_conduit.actual.core.sync.ttl import (
    build_pulse_table,
    fit_linear_timebase,
    plot_conversion_error_tools,
    plot_stacked_pulses,
)
from data_conduit.actual.core.timestamps import (
    collect_timestamps,
    collect_timestamps_nested,
)
from data_conduit.actual.datasources.multisource import MultiSource


def test_build_pulse_table_skips_orphaned_falls_without_losing_later_pulses() -> None:
    """An extra fall between pulses must not shift every later pairing."""
    result = build_pulse_table(rise_times=[1.0, 3.0], fall_times=[2.0, 2.5, 4.0])

    assert result["Start"].tolist() == [1.0, 3.0]
    assert result["End"].tolist() == [2.0, 4.0]


def test_fit_linear_timebase_requires_two_pulses_and_a_valid_edge() -> None:
    """Linear fitting should reject underdetermined data and invalid edge names."""
    one_pulse = pd.DataFrame({"Start": [0.0], "End": [0.1]})

    with pytest.raises(ValueError, match="At least two aligned pulses"):
        fit_linear_timebase(one_pulse, one_pulse)

    two_pulses = pd.DataFrame({"Start": [0.0, 1.0], "End": [0.1, 1.1]})
    with pytest.raises(ValueError, match="either 'start' or 'end'"):
        fit_linear_timebase(two_pulses, two_pulses, use="starts")


@pytest.mark.parametrize(
    "selector",
    [
        lambda key: key == "keep",
        ("keep",),
        {"keep"},
    ],
)
def test_nested_timestamps_accept_callable_and_collection_selectors(
    selector: Callable[[str], bool] | Collection[str],
) -> None:
    """Nested timestamp selectors should retain their supported selector type."""
    frame = pd.DataFrame({"Time": [2.0, 1.0]})
    nested = {"keep": {"events": frame}, "drop": {"events": frame}}

    assert collect_timestamps_nested(nested, l0_selector=selector) == [1.0, 2.0]


@pytest.mark.parametrize("name", [None, "Time"])
def test_collect_timestamps_handles_unnamed_and_time_named_dataarrays(
    name: str | None,
) -> None:
    """DataArray value names must not collide with timestamp extraction."""
    data = xr.DataArray(
        [10.0, 11.0],
        dims="Time",
        coords={"Time": [1.0, 2.0]},
        name=name,
    )

    result = collect_timestamps(data)

    assert result is not None
    assert result.tolist() == [1.0, 2.0]


def test_multisource_lookup_uses_coordinates_without_object_attrs() -> None:
    """Lookup reuse must not place a non-serializable DataArray in attrs."""
    frame = pd.DataFrame(
        {"channel_a": [1.0, 2.0]},
        index=pd.Index([0.0, 1.0], name="Time"),
    )
    source = MultiSource(
        dfs_dict={"device_a": {"stream": frame}},
        virtual_maps={
            "signal": {
                "channel_a": {"device": "device_a", "stream": "stream"},
            }
        },
        global_coord_name="channel",
        virtual_coord_names=["device", "stream"],
    )
    data = source.data_arrays["signal"]

    assert "_lookup_array_ref" not in data.attrs
    assert not any(isinstance(value, xr.DataArray) for value in data.attrs.values())
    assert data.ulookup(device="device_a") == ["channel_a"]


def test_io_rejects_duplicate_file_stems_instead_of_overwriting(tmp_path: Path) -> None:
    """Different extensions sharing one stem must not silently lose a file."""
    (tmp_path / "same.csv").write_text("value\n1\n")
    (tmp_path / "same.json").write_text('{"value": 2}\n')

    with pytest.raises(ValueError, match="duplicate output key 'same'"):
        collect_dfs(
            tmp_path,
            readers={".csv": lambda path: path, ".json": lambda path: path},
        )

    with pytest.raises(ValueError, match="duplicate output key 'same'"):
        collect_file_paths(tmp_path)


def test_empty_conversion_error_plot_returns_no_figure() -> None:
    """An empty aligned comparison should be a clean no-op."""
    empty = pd.DataFrame(columns=["Start", "End", "Duration"])

    assert plot_conversion_error_tools(empty, empty) == (None, None)


def test_stacked_plot_allows_one_empty_pulse_table() -> None:
    """A one-sided pulse plot should not infer zero pulses per row."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = pd.DataFrame({"Start": [0.0], "End": [0.1], "Duration": [0.1]})
    empty = pd.DataFrame(columns=["Start", "End", "Duration"])

    figure, axes = plot_stacked_pulses(reference, empty)

    assert figure is not None
    assert axes is not None
    plt.close(figure)
