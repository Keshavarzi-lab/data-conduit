"""Load one Q_C session and expose its aligned poses to movement 0.17+.

Data-conduit owns file discovery, event parsing and video/pose clock alignment.
Only movement functions perform the optional pose processing in this template.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import Version

from data_conduit.datastructures import StreamCatalog
from data_conduit.integrations.DLC.pose.dlc import pose_to_movement
from data_conduit.revised_qc import (
    LoadedQC,
    QCWorkflowConfig,
    load_qc_workflow,
    preview_qc_sessions,
)

REQUIRED_STREAMS = ("trials", "dlc:position", "dlc:confidence")


@dataclass(frozen=True)
class SessionData:
    """One loaded session, with unprocessed poses and loading diagnostics.

    ``raw_pose`` has position dimensions ``(time, space, keypoint, individual)``
    and confidence dimensions ``(time, keypoint, individual)``. Time is the
    original video-aligned acquisition clock in seconds; position is in pixels.
    ``loaded.session_manifest`` and ``loaded.stream_coverage`` remain available
    for inspection. The dataclass is frozen, but its pandas/xarray values are
    mutable; use ``prepare_pose`` for a separate processed copy.
    """

    loaded: LoadedQC
    trials: pd.DataFrame
    events: pd.DataFrame | None
    raw_pose: xr.Dataset


def build_config(
    root: str | Path,
    *,
    session: str,
    level_names: Sequence[str] = ("mouseID", "day"),
    sources: Sequence[str] = ("trials", "events", "dlc"),
    level_selectors: Mapping[str, Any] | None = None,
    optional_sources: Sequence[str] = (),
    dlc_file_format: str = "auto",
    nosepoke_count: int = 18,
    trial_start_buffer: float = 0.0,
    device_yaml: str | Path = "./device.yml",
    soundcard_yaml: str | Path = "./soundcard.yml",
) -> QCWorkflowConfig:
    """Declare root, single-session scope and contents without reading files.

    The default layout is ``root / mouseID / day / session``. For
    ``root / session`` use ``level_names=()``. ``session`` is the session-folder
    basename; use e.g. ``level_selectors={"l0_selector": "mouse_name"}`` when
    that basename appears beneath multiple animals. Preview rejects ambiguity.

    These figure demos require trials and DLC. Keep or remove raw ``events``
    and add other supported Q_C sources as needed. DLC also reads VideoData
    internally so its timestamps share the acquisition clock with the trials.
    """
    if not isinstance(session, str) or not session.strip():
        raise ValueError("session must be one non-empty session-folder name.")
    if isinstance(level_names, str):
        raise TypeError("level_names must be a sequence of names, e.g. ('mouseID', 'day').")
    if isinstance(sources, str) or not {"trials", "dlc"}.issubset(sources):
        raise ValueError("The figure loading template requires sources trials and dlc.")
    levels = tuple(level_names)
    return QCWorkflowConfig(
        root=root,
        depth=len(levels),
        level_names=levels,
        sources=sources,
        include=(session,),
        level_selectors=level_selectors,
        optional_sources=optional_sources,
        required_streams=REQUIRED_STREAMS,
        dlc_file_format=dlc_file_format,
        nosepoke_count=nosepoke_count,
        trial_start_buffer=trial_start_buffer,
        device_yaml=device_yaml,
        soundcard_yaml=soundcard_yaml,
    )


def preview_session(
    config: QCWorkflowConfig, *, catalog: StreamCatalog | None = None
) -> pd.DataFrame:
    """Inspect directory selection and require exactly one before reading data."""
    manifest = preview_qc_sessions(config, catalog=catalog)
    if len(manifest) != 1:
        raise ValueError(
            f"Expected exactly one session, selected {len(manifest)}. "
            "Check root, session-folder name, level_names and level_selectors."
        )
    return manifest


def _validate_pose(pose: xr.Dataset) -> None:
    """Check the public movement schema and a usable acquisition clock."""
    if Version(version("movement")) < Version("0.17.0"):
        raise RuntimeError("These templates require movement >= 0.17.0.")
    from movement.validators.datasets import ValidPosesInputs

    ValidPosesInputs.validate(pose)
    times = pose.time.to_numpy()
    if (
        len(times) < 2
        or not np.issubdtype(times.dtype, np.number)
        or not np.isfinite(times).all()
        or not (np.diff(times) > 0).all()
    ):
        raise ValueError("Pose time must contain at least two finite, increasing seconds.")
    if pose.attrs.get("time_unit") != "seconds":
        raise ValueError("Pose time_unit must be seconds on the acquisition clock.")


def load_session(
    config: QCWorkflowConfig,
    *,
    individual: str = "individual_0",
    catalog: StreamCatalog | None = None,
) -> SessionData:
    """Load a validated single-session selection and preserve its absolute clock.

    ``catalog`` is an optional custom-reader injection, also used by local
    fixture checks. Physical reader errors propagate; there is no fake-data
    fallback. Inspect coverage even though these demos select only one session.
    """
    preview = preview_session(config, catalog=catalog)
    loaded = load_qc_workflow(config, catalog=catalog)
    if not loaded.session_manifest.equals(preview):
        raise ValueError("Session selection changed between preview and loading.")
    missing = [
        name for name in REQUIRED_STREAMS
        if name not in loaded.stream_coverage
        or not loaded.stream_coverage[name].all()
    ]
    if missing:
        raise ValueError(f"Selected session lacks required stream coverage: {missing}.")

    trials = loaded.streams["trials"]
    events = loaded.streams.get("events")
    if not isinstance(trials, pd.DataFrame) or trials.empty:
        raise ValueError("The trials stream must be a non-empty pandas DataFrame.")
    if events is not None and not isinstance(events, pd.DataFrame):
        raise TypeError("The events stream must be a pandas DataFrame when present.")
    position, confidence = xr.align(
        loaded.streams["dlc:position"],
        loaded.streams["dlc:confidence"],
        join="exact",
    )
    # The existing adapter preserves the aligned clock/provenance but uses
    # plural dimension names. movement 0.17 uses singular names and this order.
    raw_pose = pose_to_movement(
        position, confidence, individual=individual, time_coord=config.dim
    ).rename({"keypoints": "keypoint", "individuals": "individual"})
    raw_pose["position"] = raw_pose.position.transpose(
        "time", "space", "keypoint", "individual"
    )
    raw_pose["confidence"] = raw_pose.confidence.transpose(
        "time", "keypoint", "individual"
    )
    raw_pose.attrs.update(time_unit="seconds", spatial_unit="pixels")
    raw_pose.time.attrs["units"] = "seconds"
    raw_pose.position.attrs["units"] = "pixels"
    _validate_pose(raw_pose)
    return SessionData(
        loaded=loaded,
        trials=trials.copy(deep=True),
        events=events.copy(deep=True) if events is not None else None,
        raw_pose=raw_pose.copy(deep=True),
    )


def prepare_pose(
    raw_pose: xr.Dataset,
    *,
    confidence_threshold: float | None = None,
    max_gap_frames: int | None = None,
    smoothing_window: int | None = None,
) -> xr.Dataset:
    """Return a copy with optional movement filtering, interpolation and smoothing.

    Every operation is off by default. Confidence filtering masks positions,
    preserving measured confidence. ``max_gap_frames`` enables movement's
    linear interpolation for at most that many consecutive NaN samples; it
    counts samples, not elapsed seconds. Its interpolation uses sample order,
    so inspect acquisition-clock gaps before enabling it on irregular data.
    ``smoothing_window`` enables movement's rolling median in samples, with
    all samples in a window required (long missing gaps remain missing).
    Processing is within this session only, and never resets time to zero.
    """
    _validate_pose(raw_pose)
    from movement.filtering import (
        filter_by_confidence,
        interpolate_over_time,
        rolling_filter,
    )

    pose = raw_pose.copy(deep=True)
    position = pose.position
    if confidence_threshold is not None:
        if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("DLC confidence_threshold must lie between 0 and 1.")
        position = filter_by_confidence(
            position, pose.confidence, threshold=confidence_threshold
        )
    if max_gap_frames is not None:
        if type(max_gap_frames) is not int or max_gap_frames < 1:
            raise ValueError("max_gap_frames must be a positive integer or None.")
        position = interpolate_over_time(position, max_gap=max_gap_frames)
    if smoothing_window is not None:
        if (
            type(smoothing_window) is not int
            or smoothing_window < 3
            or smoothing_window % 2 != 1
            or smoothing_window > pose.sizes["time"]
        ):
            raise ValueError("smoothing_window must be odd, >= 3 and <= session samples.")
        position = rolling_filter(
            position, window=smoothing_window, statistic="median"
        )
    pose["position"] = position
    return pose
