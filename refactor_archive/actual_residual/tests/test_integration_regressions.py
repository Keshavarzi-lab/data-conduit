"""Regression tests for optional integration boundaries and alignment safety."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data_conduit.actual.core.io import readers as reader_registry
from data_conduit.actual.integrations.DLC.pose import dlc
from data_conduit.actual.integrations.harp.datasource_presets import multidevice
from data_conduit.actual.integrations.harp.harptools import (
    harptools_core,
    read_harp_bin,
    register_harp_reader,
)


def _pose_table(keypoints: tuple[str, ...], frame_count: int) -> pd.DataFrame:
    """Build an in-memory DLC table with the standard three-column schema."""
    columns = pd.MultiIndex.from_tuples(
        [(keypoint, coordinate) for keypoint in keypoints for coordinate in ("x", "y", "likelihood")],
        names=("bodyparts", "coords"),
    )
    values = np.arange(frame_count * len(columns), dtype=float).reshape(frame_count, len(columns))
    return pd.DataFrame(values, columns=columns)


def _read_mock_pose(
    monkeypatch: pytest.MonkeyPatch,
    tables: list[pd.DataFrame],
) -> dict:
    """Run ``read_dlc_pose`` against monkeypatched in-memory file tables."""
    paths = [Path(f"segment_{index}.csv") for index in range(len(tables))]
    table_by_path = dict(zip(paths, tables, strict=True))
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(dlc, "_dlc_files", lambda *_args: paths)
    monkeypatch.setattr(dlc, "_read_dlc_table", table_by_path.__getitem__)
    return dlc.read_dlc_pose("unused-session")


def test_multidevice_forwards_level_selectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """MultiDevice should pass selector kwargs to its HARP directory loader."""
    captured: dict = {}

    def fake_collect_harp_dfs(**kwargs) -> dict:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(multidevice, "collect_harp_dfs", fake_collect_harp_dfs)
    multidevice.MultiDevice(
        experiment_directory_path="unused",
        l1_selector=("32", "34"),
    )

    assert captured["l1_selector"] == ("32", "34")


def test_harp_level_zero_selector_is_intersected_with_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller l0 selector should narrow, not replace or duplicate, the device filter."""
    captured: dict = {}

    def fake_collect_dfs(**kwargs) -> dict:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(harptools_core, "collect_dfs", fake_collect_dfs)
    harptools_core.collect_harp_dfs(
        "unused",
        "unused.yml",
        device_type="Behavior",
        l0_selector=("Behavior1", "SoundCard"),
    )

    selector = captured["l0_selector"]
    assert selector("Behavior1")
    assert not selector("Behavior0")
    assert not selector("SoundCard")


def test_harp_reader_registration_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated explicit registration of the same HARP reader should be harmless."""
    monkeypatch.delitem(reader_registry._READERS, "harp_bin", raising=False)

    register_harp_reader()
    register_harp_reader()

    assert reader_registry.get_reader("harp_bin") is read_harp_bin


def test_dlc_reader_rejects_inconsistent_file_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different keypoint sets must fail before xarray can outer-join them."""
    with pytest.raises(ValueError, match="does not match the first file's keypoint schema"):
        _read_mock_pose(
            monkeypatch,
            [_pose_table(("nose",), 1), _pose_table(("tail",), 1)],
        )


def test_dlc_alignment_checks_each_file_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equal session totals must not hide different per-file frame counts."""
    pose = _read_mock_pose(
        monkeypatch,
        [_pose_table(("nose",), 1), _pose_table(("nose",), 2)],
    )
    assert pose["position"].attrs[dlc._SEGMENT_FRAME_COUNTS_ATTR] == (1, 2)

    video = SimpleNamespace(
        df={
            "segment_0": pd.DataFrame(index=pd.Index([10.0, 11.0], name="Time")),
            "segment_1": pd.DataFrame(index=pd.Index([12.0], name="Time")),
        }
    )
    with pytest.raises(ValueError, match="per-file frame counts differ"):
        dlc.align_pose_to_video(video, pose)


def test_multifile_dlc_requires_unflattened_video_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flattened video vector cannot prove multi-file positional alignment."""
    pose = _read_mock_pose(
        monkeypatch,
        [_pose_table(("nose",), 1), _pose_table(("nose",), 2)],
    )

    with pytest.raises(ValueError, match="pass the original VideoData source"):
        dlc.align_pose_to_video(np.array([10.0, 11.0, 12.0]), pose)


def test_dlc_alignment_accepts_matching_file_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matching per-file counts should align in filename-first order."""
    pose = _read_mock_pose(
        monkeypatch,
        [_pose_table(("nose",), 1), _pose_table(("nose",), 2)],
    )
    video = SimpleNamespace(
        df={
            "segment_0": pd.DataFrame(index=pd.Index([10.0], name="Time")),
            "segment_1": pd.DataFrame(index=pd.Index([11.0, 12.0], name="Time")),
        }
    )

    aligned = dlc.align_pose_to_video(video, pose)

    np.testing.assert_array_equal(aligned["position"].coords["Time"], [10.0, 11.0, 12.0])
