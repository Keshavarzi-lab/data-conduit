"""End-to-end checks for the Q_C workflow running on the actual package."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_conduit.actual.datastructures import (
    DataStructure,
    StreamCatalog,
    TrialSpec,
    parse_trials,
)
from data_conduit.qc import build_qc_catalog, qc_datastructure, slice_pose_for_trial
from data_conduit.qc.path_plots import diagnose_path_grid
from data_conduit.qc.trial_spec import QC_TRIALS, qc_trial_spec
from data_conduit.qc.trials import parse_events_to_trials


def _events(*, two_trials: bool = False) -> pd.DataFrame:
    rows = [
        (0.0, "Start trial logic"),
        (0.5, "Target zone available (X Y : 740 687 - Radius : 150)"),
        (1.0, "NosePokesLED ON"),
        (2.0, "Target zone triggered"),
        (3.0, "Poke: {Success=True, ChosenPort=3, CorrectPort=3-}"),
    ]
    if two_trials:
        rows.extend(
            [
                (3.2, "Target zone available (X Y : 740 687 - Radius : 100)"),
                (4.0, "Poke: {Success=False, ChosenPort=-1, CorrectPort=4-}"),
            ],
        )
    return pd.DataFrame(
        {"Event": [event for _, event in rows]},
        index=pd.Index([time for time, _ in rows], name="Time"),
    )


def _write_events(session: Path, *, two_trials: bool = False) -> None:
    events_dir = session / "ExperimentEvents"
    events_dir.mkdir(parents=True)
    events = _events(two_trials=two_trials)
    lines = ["Seconds,Value", *(f"{time},{row.Event}" for time, row in events.iterrows())]
    (events_dir / "events.csv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_video_segments(session: Path, counts: tuple[int, ...]) -> None:
    video_dir = session / "VideoData"
    video_dir.mkdir(parents=True)
    next_frame = 0
    for segment, count in enumerate(counts, start=1):
        frames = np.arange(next_frame, next_frame + count)
        pd.DataFrame(
            {
                "Value.ChunkData.FrameID": frames,
                "Value.ChunkData.Timestamp": frames * 10,
            },
            index=pd.Index(frames.astype(float), name="Seconds"),
        ).to_csv(video_dir / f"segment_{segment:02d}.csv")
        next_frame += count


def _write_dlc_segments(session: Path, counts: tuple[int, ...]) -> None:
    dlc_dir = session / "DLC"
    dlc_dir.mkdir(parents=True)
    columns = pd.MultiIndex.from_product(
        [["scorer"], ["nose", "body"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    next_frame = 0
    for segment, count in enumerate(counts, start=1):
        frame_values = np.arange(next_frame, next_frame + count, dtype=float)
        values = np.column_stack(
            [
                frame_values + 10,
                frame_values + 20,
                np.full(count, 0.99),
                frame_values + 30,
                frame_values + 40,
                np.full(count, 0.98),
            ],
        )
        pd.DataFrame(values, columns=columns, index=np.arange(count)).to_csv(
            dlc_dir / f"segment_{segment:02d}.csv",
        )
        next_frame += count


def _write_session_settings(session: Path) -> None:
    """Write two split JSONL sources with distinct metadata and trial rows."""
    settings_dir = session / "SessionSettings"
    settings_dir.mkdir(parents=True)
    for index, protocol in enumerate(("training", "testing"), start=1):
        record = {
            "seconds": float(index),
            "value": {
                "metadata": {"protocol": protocol},
                "trials": [{"maxRuntime": index * 10}],
            },
        }
        (settings_dir / f"settings_{index}.jsonl").write_text(
            json.dumps(record) + "\n",
            encoding="utf-8",
        )


def _session(root: Path, mouse: str, day: str, name: str) -> Path:
    session = root / mouse / day / name
    session.mkdir(parents=True)
    return session


def test_qc_public_types_come_from_actual() -> None:
    catalog = build_qc_catalog(streams=("events", "session_settings", "video", "dlc"))

    assert isinstance(catalog, StreamCatalog)
    assert isinstance(QC_TRIALS, TrialSpec)
    assert catalog._readers["events"].optional is False
    assert catalog._readers["session_settings"].optional is True
    assert catalog._readers["video"].optional is True
    assert catalog._readers["dlc"].optional is True


def test_actual_trial_parser_matches_legacy_qc_parser_for_one_trial() -> None:
    events = _events()
    legacy = parse_events_to_trials(events, nosepoke_count=18, trial_start_buffer=0.0)
    actual = parse_trials(
        events,
        qc_trial_spec(nosepoke_count=18, trial_start_buffer=0.0),
    )

    assert_frame_equal(
        actual,
        legacy[actual.columns],
        check_dtype=False,
    )


def test_actual_parser_fixes_contiguous_trial_boundary_angle() -> None:
    trials = parse_trials(
        _events(two_trials=True),
        qc_trial_spec(nosepoke_count=18, trial_start_buffer=0.0),
    )

    assert trials.loc[0, "angle_offset"] == 0.0
    assert np.isnan(trials.loc[1, "angle_offset"])


def test_qc_actual_pipeline_loads_trials_pose_and_slices(tmp_path: Path) -> None:
    with_pose = _session(tmp_path, "mouse_a", "day_01", "session_a")
    without_pose = _session(tmp_path, "mouse_b", "day_02", "session_b")
    _write_events(with_pose)
    _write_video_segments(with_pose, (2, 2))
    _write_dlc_segments(with_pose, (2, 2))
    _write_events(without_pose)

    data_structure = qc_datastructure(
        tmp_path,
        depth=2,
        level_names=("mouseID", "day"),
        streams=("dlc",),
    )

    assert isinstance(data_structure, DataStructure)
    with pytest.warns(UserWarning, match="optional reader"):
        result = data_structure.load()

    assert set(result) == {"trials", "dlc:position", "dlc:confidence"}
    assert result["trials"]["session"].tolist() == ["session_a", "session_b"]
    assert result["trials"]["mouseID"].tolist() == ["mouse_a", "mouse_b"]

    pose = result["dlc:position"]
    assert pose.dims == ("Time", "keypoints", "space")
    assert pose.coords["session"].values.tolist() == ["session_a"] * 4
    assert pose.coords["mouseID"].values.tolist() == ["mouse_a"] * 4

    first_trial = result["trials"].iloc[0]
    outbound = slice_pose_for_trial(pose, first_trial, segment="outbound")
    assert outbound.coords["Time"].values.tolist() == [0.0, 1.0, 2.0]

    diagnostics = diagnose_path_grid(
        result,
        centroid_points=("nose",),
        row_by="session",
        col_by="segment",
    )
    status = dict(zip(diagnostics["session"], diagnostics["status"], strict=False))
    assert status["session_a"] == "would_plot"
    assert status["session_b"] == "no_pose_for_session"


def test_qc_keeps_video_boundaries_until_dlc_alignment(tmp_path: Path) -> None:
    session = _session(tmp_path, "mouse_a", "day_01", "session_a")
    _write_events(session)
    _write_video_segments(session, (1, 3))
    _write_dlc_segments(session, (2, 2))

    data_structure = qc_datastructure(
        tmp_path,
        depth=2,
        level_names=("mouseID", "day"),
        streams=("dlc",),
    )

    with pytest.raises(ValueError, match="per-file frame counts differ"):
        data_structure.load()


def test_qc_video_only_output_is_a_flat_dataframe(tmp_path: Path) -> None:
    session = _session(tmp_path, "mouse_a", "day_01", "session_a")
    _write_events(session)
    _write_video_segments(session, (2, 2))

    result = qc_datastructure(
        tmp_path,
        depth=2,
        level_names=("mouseID", "day"),
        streams=("video",),
    ).load()

    assert set(result) == {"trials", "video"}
    assert isinstance(result["video"], pd.DataFrame)
    assert result["video"].index.name == "Time"
    assert result["video"]["session"].tolist() == ["session_a"] * 4


def test_qc_keeps_session_settings_schemas_separate(tmp_path: Path) -> None:
    """Multiple settings files should combine by component, never across schemas."""
    session = _session(tmp_path, "mouse_a", "day_01", "session_a")
    _write_events(session)
    _write_session_settings(session)

    result = qc_datastructure(
        tmp_path,
        depth=2,
        level_names=("mouseID", "day"),
        streams=("session_settings",),
    ).load()

    assert set(result) == {
        "trials",
        "session_settings:metadata",
        "session_settings:trials",
    }
    assert result["session_settings:metadata"]["protocol"].tolist() == [
        "training",
        "testing",
    ]
    assert result["session_settings:trials"]["maxRuntime"].tolist() == [10, 20]


def test_qc_events_remain_required(tmp_path: Path) -> None:
    """A selected session without its event log is not a valid Q_C session."""
    _session(tmp_path, "mouse_a", "day_01", "session_a")
    data_structure = qc_datastructure(
        tmp_path,
        depth=2,
        level_names=("mouseID", "day"),
        streams=("events",),
    )

    with pytest.raises(FileNotFoundError, match="ExperimentEvents"):
        data_structure.load()


def test_qc_missing_optional_harp_data_is_skipped(tmp_path: Path) -> None:
    """A session without HARP folders should still contribute its trial table."""
    session = _session(tmp_path, "mouse_a", "day_01", "session_a")
    _write_events(session)

    structure = qc_datastructure(
        tmp_path,
        depth=2,
        level_names=("mouseID", "day"),
        streams=("nosepoke",),
    )
    with pytest.warns(UserWarning, match="optional reader 'nosepoke' absent"):
        result = structure.load()

    assert set(result) == {"trials"}


def test_qc_harp_data_requires_a_real_schema(tmp_path: Path) -> None:
    """Existing HARP binaries must not be silently treated as absent data."""
    session = _session(tmp_path, "mouse_a", "day_01", "session_a")
    _write_events(session)
    behavior = session / "Behavior0"
    behavior.mkdir()
    (behavior / "Behavior0_32.bin").write_bytes(b"not read without a schema")

    structure = qc_datastructure(
        tmp_path,
        depth=2,
        level_names=("mouseID", "day"),
        streams=("nosepoke",),
        device_yaml=tmp_path / "missing-device.yml",
    )

    with pytest.raises(ValueError, match="HARP schema"):
        structure.load()
