"""Load one refactor_qc session and expose its aligned poses to movement 0.17+.

Data-conduit owns file discovery, event parsing and video/pose clock alignment.
Only movement functions perform the optional pose processing in this template.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import Version

from data_conduit.core.utils import _concat_split_dataframes
from data_conduit.datasources.monosource import ExperimentEvents
from data_conduit.datastructures import DataStructure, StreamCatalog, StreamMap
from data_conduit.integrations.DLC.pose.dlc import pose_to_movement
from data_conduit.refactor_qc import QC_STREAMS, build_qc_catalog

REQUIRED_STREAMS = ("trials", "dlc:position", "dlc:confidence")


@dataclass(frozen=True)
class SessionConfig:
    """Notebook settings passed to refactor_qc's catalog and core DataStructure."""

    root: Path
    session: str
    level_names: tuple[str, ...]
    sources: tuple[str, ...]
    level_selectors: Mapping[str, Any]
    device_yaml: Path
    soundcard_yaml: Path
    dlc_file_format: str
    nosepoke_count: int
    trial_start_buffer: float
    dim: str = "Time"
    session_coord: str = "session"


@dataclass(frozen=True)
class LoadedSession:
    """Loaded streams and diagnostics retained for the existing notebook calls."""

    config: SessionConfig
    datastructure: DataStructure
    streams: StreamMap
    session_manifest: pd.DataFrame
    stream_coverage: pd.DataFrame


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

    loaded: LoadedSession
    trials: pd.DataFrame
    events: pd.DataFrame | None
    raw_pose: xr.Dataset


@dataclass(frozen=True)
class FigureData:
    """Explicit shared inputs for the one-session figure functions.

    ``position`` selects one animal and has dimensions ``(time, space,
    keypoint)``; ``point`` selects one landmark and has ``(time, space)``.
    Spatial values are pixels and time is the original acquisition clock in
    seconds. ``raw_pose`` and ``pose`` retain unprocessed and processed copies.
    The returned trial table is sorted by start time with zero-based row labels.
    """

    session_data: SessionData
    raw_pose: xr.Dataset
    pose: xr.Dataset
    position: xr.DataArray
    point: xr.DataArray
    trials: pd.DataFrame
    events: pd.DataFrame | None


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
) -> SessionConfig:
    """Declare root, single-session scope and contents without reading files.

    The default layout is ``root / mouseID / day / session``. For
    ``root / session`` use ``level_names=()``. ``session`` is the session-folder
    basename; use e.g. ``level_selectors={"l0_selector": "mouse_name"}`` when
    that basename appears beneath multiple animals. Preview rejects ambiguity.

    These figure demos require trials and DLC. Keep or remove raw ``events``
    and add other supported Q_C sources as needed. DLC also reads VideoData
    internally so its timestamps share the acquisition clock with the trials.

    The template's ``events`` source means raw plot markers. The refactor_qc
    catalog uses its legacy ``events`` token for parsed trials, so the loader
    adds a separate raw-event reader when requested. Reader optionality follows
    refactor_qc: only DLC may be optional there, and these figures still require
    its final streams. ``optional_sources`` accepts only that existing policy.
    """
    if not isinstance(session, str) or not session.strip():
        raise ValueError("session must be one non-empty session-folder name.")
    if isinstance(level_names, str):
        raise TypeError("level_names must be a sequence of names, e.g. ('mouseID', 'day').")
    if isinstance(sources, str) or not {"trials", "dlc"}.issubset(sources):
        raise ValueError("The figure loading template requires sources trials and dlc.")
    levels = tuple(level_names)
    requested = tuple(sources)
    if any(not isinstance(name, str) or not name for name in levels):
        raise ValueError("level_names must contain non-empty strings.")
    if len(set(levels)) != len(levels) or set(levels).intersection({"path", "session"}):
        raise ValueError("level_names must be unique and cannot use path or session.")
    if len(set(requested)) != len(requested) or set(requested) - {"trials", *QC_STREAMS}:
        raise ValueError(f"sources must be unique names from {('trials', *QC_STREAMS)}.")
    if isinstance(optional_sources, str) or set(optional_sources) - {"dlc"}:
        raise ValueError("refactor_qc only supports optional DLC; the figure still requires pose coverage.")
    selectors = deepcopy(dict(level_selectors or {}))
    if set(selectors) - {f"l{i}_selector" for i in range(len(levels))}:
        raise ValueError("level_selectors must target existing levels with l{n}_selector keys.")
    if dlc_file_format not in {"auto", "csv", "h5"}:
        raise ValueError("dlc_file_format must be auto, csv or h5.")
    if isinstance(nosepoke_count, bool) or not isinstance(nosepoke_count, int) or nosepoke_count < 1:
        raise ValueError("nosepoke_count must be a positive integer.")
    if not np.isfinite(trial_start_buffer) or trial_start_buffer < 0:
        raise ValueError("trial_start_buffer must be finite and non-negative.")
    return SessionConfig(
        root=Path(root),
        session=session,
        level_names=levels,
        sources=requested,
        level_selectors=MappingProxyType(selectors),
        dlc_file_format=dlc_file_format,
        nosepoke_count=nosepoke_count,
        trial_start_buffer=trial_start_buffer,
        device_yaml=Path(device_yaml),
        soundcard_yaml=Path(soundcard_yaml),
    )


def _read_events(path: Path) -> pd.DataFrame:
    """Read raw markers using the same split-file loader as refactor_qc trials."""
    events = _concat_split_dataframes(ExperimentEvents(experiment_directory_path=path).df)
    if not isinstance(events, pd.DataFrame):
        raise TypeError("The raw events reader must return a pandas DataFrame.")
    return events


def _build_catalog(config: SessionConfig) -> StreamCatalog:
    """Keep refactor_qc's trial reader/alignment and add requested raw markers."""
    catalog = build_qc_catalog(
        streams=tuple(name for name in config.sources if name != "trials"),
        device_yaml=config.device_yaml,
        soundcard_yaml=config.soundcard_yaml,
        nosepoke_count=config.nosepoke_count,
        trial_start_buffer=config.trial_start_buffer,
        dlc_file_format=config.dlc_file_format,
    )
    if "events" in config.sources:
        # The existing trial reader owns its event read; this additional read
        # preserves the raw timestamps without replacing the tested parser.
        catalog.add_reader("events", _read_events)
    return catalog


def _build_datastructure(config: SessionConfig, catalog: StreamCatalog) -> DataStructure:
    """Apply notebook directory settings through the core selection API."""
    return DataStructure(
        config.root, catalog, depth=len(config.level_names),
        level_names=config.level_names, include=(config.session,),
        **dict(config.level_selectors),
    )


def _session_manifest(datastructure: DataStructure) -> pd.DataFrame:
    """Describe the selected paths and their hierarchy labels without loading."""
    records = {
        session_id: {"path": Path(session.path), **session.levels, **session.metadata}
        for session_id, session in datastructure.sessions.items()
    }
    return pd.DataFrame.from_dict(records, orient="index").rename_axis("session")


def preview_session(
    config: SessionConfig, *, catalog: StreamCatalog | None = None
) -> pd.DataFrame:
    """Inspect directory selection and require exactly one before reading data."""
    datastructure = _build_datastructure(config, catalog if catalog is not None else StreamCatalog())
    datastructure.select()
    manifest = _session_manifest(datastructure)
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
    config: SessionConfig,
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
    datastructure = _build_datastructure(config, catalog if catalog is not None else _build_catalog(config))
    streams = datastructure.load(dim=config.dim, session_coord=config.session_coord)
    manifest = _session_manifest(datastructure)
    coverage = pd.DataFrame(
        {name: [session_id in container.session_ids for session_id in datastructure.sessions]
         for name, container in datastructure.containers.items()},
        index=manifest.index, dtype=bool,
    )
    loaded = LoadedSession(config, datastructure, streams, manifest, coverage)
    if not loaded.session_manifest.equals(preview):
        raise ValueError("Session selection changed between preview and loading.")
    required = REQUIRED_STREAMS + (("events",) if "events" in config.sources else ())
    missing = [
        name for name in required
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


def load_figure_data(
    config: SessionConfig,
    *,
    individual: str = "individual_0",
    tracking_keypoint: str = "body",
    confidence_threshold: float | None = None,
    max_gap_frames: int | None = None,
    smoothing_window: int | None = None,
) -> FigureData:
    """Load and prepare the explicit data inputs consumed by figure functions.

    Parameters
    ----------
    config
        One-session selection built by ``build_config``. Source discovery,
        trial parsing and DLC clock alignment use the existing loader.
    individual, tracking_keypoint
        Animal and landmark selected from the movement pose schema. All
        landmarks remain available in ``position`` for head/body calculations.
    confidence_threshold, max_gap_frames, smoothing_window
        Passed unchanged to ``prepare_pose``. Processing takes place on the
        continuous recording before any plot window or trial is selected.

    Returns
    -------
    FigureData
        Raw/processed pose, selected arrays, sorted trial rows and raw events.
        This function does not create plots, display tables or reset time.
    """
    #=== 1| Load one session and process the continuous pose ========
    session_data = load_session(config, individual=individual)
    raw_pose = session_data.raw_pose
    pose = prepare_pose(
        raw_pose,
        confidence_threshold=confidence_threshold,
        max_gap_frames=max_gap_frames,
        smoothing_window=smoothing_window,
    )

    #=== 2| Select the animal and tracked point without dropping other landmarks ========
    position = pose.position.sel(individual=individual, drop=True)
    if tracking_keypoint not in position.keypoint:
        raise ValueError(
            f"Unknown tracking_keypoint {tracking_keypoint!r}; "
            f"choose from {position.keypoint.values.tolist()}."
        )
    point = position.sel(keypoint=tracking_keypoint, drop=True).copy(deep=True)
    point.attrs["units"] = "px"

    #=== 3| Return a stable trial table alongside the original timing and pose ========
    trials = session_data.trials.sort_values("start_time").reset_index(drop=True)
    if trials.empty:
        raise ValueError("No parsed trials are available in this session.")
    return FigureData(
        session_data=session_data,
        raw_pose=raw_pose,
        pose=pose,
        position=position,
        point=point,
        trials=trials,
        events=session_data.events,
    )


def read_session_video_frame(
    session_path: str | Path,
    *,
    video_subdir: str = "UndistortedVideoData",
) -> tuple[np.ndarray, Path]:
    """Read the first frame and return it with its video path.

    Parameters
    ----------
    session_path
        Exact selected session directory from the loader's session manifest.
    video_subdir
        Directory whose pixel coordinates match the DLC recording. Use
        ``UndistortedVideoData`` for the undistorted tracking in these examples;
        raw-video tracking should instead use ``VideoData``.

    Returns
    -------
    frame, video_path
        RGB image with shape ``(height, width, 3)`` and the source path. No
        resizing, coordinate transformation or fallback image is applied.
        Split recordings follow filename order, as in the existing DLC loader.
    """
    #=== 1| Resolve the first video chunk within the selected session ========
    import imageio.v3 as iio

    video_dir = Path(session_path) / video_subdir
    video_files = sorted(
        path for path in video_dir.glob("*")
        if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
    )
    if not video_files:
        raise FileNotFoundError(
            f"No video found in {video_dir}; choose the video_subdir used for DLC tracking."
        )

    #=== 2| Read frame zero and retain the original image geometry ========
    video_path = video_files[0]
    frame = iio.imread(video_path, index=0)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4) or min(frame.shape[:2]) == 0:
        raise ValueError(f"Video frame has an unsupported image shape: {frame.shape}.")
    return frame[..., :3], video_path
