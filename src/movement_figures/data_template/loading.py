"""
Build the lab's DataStructure from a root directory and source selection.
------------------------------------------------------------------------

Description:
    Reuse refactor_qc's readers and the generic DataStructure selection API.
    Call build_datastructure(...) to configure one or many recordings, then
    call its .load() method to obtain the StreamMap of pandas/xarray objects.

    The Q_C catalog parses trials and aligns DLC to video timestamps. Here,
    source names describe their outputs: 'trials' is the trial table and 'events'
    is its original event log. Pose conversion stays in data-conduit's existing
    pose_to_movement function; prepare_pose optionally filters its result.
    Video-frame reading remains in movement_figures.video.

Contents:
--------------------------------
- build_datastructure:  Configure the lab's existing DataStructure.
- prepare_pose:         Optionally filter, interpolate and smooth one pose copy.
"""


################################################################################
# Imports
################################################################################

from collections.abc import Sequence
from pathlib import Path
from numbers import Integral, Real
from typing import Any

import numpy as np
import xarray as xr

from data_conduit.datastructures import DataStructure, TrialSpec
from data_conduit.refactor_qc import qc_datastructure

################################################################################


################################################################################
# Public API
################################################################################


# ===============================================================================
# 1| Configure the Lab's DataStructure
# ===============================================================================


def build_datastructure(
    root: str | Path,                               # Directory above the named hierarchy levels.
    *,
    level_names: Sequence[str] = ("mouseID", "day"),  # Lab layout: root / mouseID / day / session.
    streams: Sequence[str] = ("trials", "nosepoke", "soundcard", "session_settings", "video", "dlc"),
                                                    # Explicit output names; raw events are opt-in.
    include: Sequence[str] | None = None,            # Keep these session folder names; None keeps all.
    exclude: Sequence[str] | None = None,            # Alternatively omit these session folder names.
    raw_events: bool = False,                        # Convenience alias for adding "events" to the requested sources.
    device_yaml: str | Path = "./device.yml",        # Lab Nosepoke board register definitions.
    soundcard_yaml: str | Path = "./soundcard.yml",  # Lab SoundCard register definitions.
    nosepoke_count: int = 18,                        # Port count used by the existing Q_C parser.
    trial_start_buffer: float = 0.0,                 # Preserve the existing lab parser setting.
    trial_spec: TrialSpec | None = None,             # Optional experiment rules; None keeps the existing Q_C specification.
    dlc_file_format: str = "auto",                   # Existing reader prefers CSV, otherwise HDF5.
    **level_selectors: Any,                         # For example l0_selector=["mouse_A", "mouse_B"].
) -> DataStructure:
    """
    Configure data loading for the selected mice, days and session folders.

    This constructs the existing Q_C DataStructure. Recording files are read
    when the caller runs .load(), which returns the ordinary StreamMap.

    Parameters
    ----------
    root : str | Path
        Directory containing the experimental hierarchy.
    level_names : Sequence[str]
        Intermediate folder labels; use () when root directly contains sessions.
    streams : Sequence[str]
        Requested sources: trials, events, nosepoke, soundcard, session_settings,
        video and dlc. The default requests all except raw events. Trials are
        only parsed when requested; dlc also needs video times for alignment.
    include, exclude : Sequence[str] | None
        Session folder names to keep or omit. Supply at most one of these lists.
    raw_events : bool
        Add 'events' to streams. Default False; an explicit 'events' source is
        retained regardless. The catalog reads the log once for events/trials.
    device_yaml, soundcard_yaml : str | Path
        Device definitions used when loading the corresponding HARP sources.
    nosepoke_count : int
        Number of arena ports for the Q_C parser. Default 18.
    trial_start_buffer : float
        Existing parser's offset from the previous trial end, in seconds.
    trial_spec : TrialSpec | None
        Trial rules passed through to the catalog. None builds the existing Q_C
        specification using nosepoke_count and trial_start_buffer. Supplying a
        specification makes that object responsible for the trial rules.
    dlc_file_format : str
        'csv', 'h5', or 'auto' to prefer CSV when available.
    **level_selectors : Any
        Existing l0_selector, l1_selector, etc.; each can select multiple folders.

    Returns
    -------
    DataStructure
        Configured object. .load() returns its streams; .sessions holds their
        source directories. Session and hierarchy labels accompany loaded data.
    """

    # === 1| Resolve the Requested Source Names Before Building the Catalog =======

    if isinstance(streams, str):
        raise TypeError('streams must be a sequence, for example ("trials", "dlc").')
    if isinstance(level_names, str):
        raise TypeError('level_names must contain one name per intermediate folder.')

    requested = list(streams)                       # Copy the caller's source list so raw_events does not mutate it.
    if raw_events and 'events' not in requested:
        requested.append('events')                 # Reuse the catalog's shared log instead of installing a second reader.

    # === 2| Reuse the Lab Recipe and Generic Directory Selection =================

    datastructure = qc_datastructure(
        root=root,
        depth=len(level_names),             # One directory level per supplied hierarchy name.
        level_names=level_names,
        streams=requested,
        include=include,
        exclude=exclude,
        device_yaml=device_yaml,
        soundcard_yaml=soundcard_yaml,
        nosepoke_count=nosepoke_count,
        trial_start_buffer=trial_start_buffer,
        trial_spec=trial_spec,
        dlc_file_format=dlc_file_format,
        legacy_events_as_trials=False,      # "events" returns events here; "trials" explicitly requests parsed trials.
        **level_selectors,
    )

    return datastructure


# ===============================================================================


################################################################################
# Optional Pose Processing
################################################################################


# ===============================================================================
# 1| Filter, Interpolate and Smooth a Copy of One Recording's Pose
# ===============================================================================


def prepare_pose(
    raw_pose: xr.Dataset,                          # Full recording converted by data-conduit's pose_to_movement.
    *,
    confidence_threshold: float | None = None,      # Below-threshold or missing likelihood becomes missing position.
    max_gap_frames: int | None = None,              # Longest internal missing run to fill; None preserves every gap.
    smoothing_window: int | None = None,            # Odd median window in frames; None leaves positions unsmoothed.
) -> xr.Dataset:                                   # Returns a processed copy with the raw input and acquired times unchanged.
    """
    Apply explicitly requested movement filters to one full recording.

    The order is confidence filtering, interpolation, then median smoothing.
    Each step is optional and disabled by default. Processing occurs before
    trial selection so trial boundaries do not create artificial filter edges.
    Confidence values remain available; only the copied position is modified.

    Parameters
    ----------
    raw_pose : xr.Dataset
        One recording's movement pose, with position and confidence on acquired
        seconds. Use pose_to_movement on matching data-conduit arrays first.
        Multiple recordings must be processed separately, even if their clocks
        happen to be increasing when concatenated.
    confidence_threshold : float | None
        Keep coordinates whose DLC confidence is at least this value in [0, 1].
        Missing confidence is also masked. None disables this operation; choose
        a threshold after inspecting the tracking rather than assuming one is
        suitable for every recording.
    max_gap_frames : int | None
        Positive maximum number of consecutive missing samples to interpolate.
        movement uses linear interpolation by sample order and fills internal
        gaps bounded by valid samples. Longer gaps and missing edges remain.
        None disables interpolation. This is a frame limit, not a seconds limit;
        it does not detect unrecorded acquisition pauses in an irregular clock.
    smoothing_window : int | None
        Odd median window of at least three frames and no longer than the
        recording. A window requires that many valid samples, so unresolved
        gaps are not filled from partial windows. None disables smoothing.

    Returns
    -------
    xr.Dataset
        Independent copy containing processed position, original confidence,
        original time coordinates and recording labels. With every option None,
        returns an unchanged copy. It does not select trials, convert spatial
        units or correct lens distortion.
    """
    from movement.filtering import filter_by_confidence, interpolate_over_time, rolling_filter
    from movement.validators.datasets import ValidPosesInputs

    # === 1| Check the Pose Schema and Keep Temporal Operations Within a Recording =

    ValidPosesInputs.validate(raw_pose)              # Use movement's own checks for the required pose variables and dimensions.
    if raw_pose.position.dims != ('time', 'space', 'keypoint', 'individual'):
        raise ValueError('position must have dimensions (time, space, keypoint, individual).')
    time_dtype = raw_pose.time.dtype
    if not (np.issubdtype(time_dtype, np.integer) or np.issubdtype(time_dtype, np.floating)):
        raise TypeError('Pose time must contain real numeric acquired seconds.')

    times = raw_pose.time.to_numpy()
    # Compare acquired times without subtracting: unsigned integer subtraction
    # can wrap at a clock rollback and incorrectly report a positive interval.
    if len(times) == 0 or not np.isfinite(times).all() or np.any(times[1:] <= times[:-1]):
        raise ValueError('Pose times must be non-empty, finite and strictly increasing.')
    if raw_pose.attrs.get('time_unit') != 'seconds':
        raise ValueError('Pose time_unit must be seconds; convert the acquired arrays first.')

    if 'session' in raw_pose.coords:
        sessions = np.asarray(raw_pose.session.values).reshape(-1)
        if len(np.unique(sessions)) != 1:
            raise ValueError('Select one recording before filtering or interpolating its pose.')

    # === 2| Check Each Enabled Filter's Numeric Parameters =======================

    if confidence_threshold is not None:
        if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, Real):
            raise TypeError('confidence_threshold must be a number or None.')
        if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError('confidence_threshold must lie between zero and one.')

    for name, value in (('max_gap_frames', max_gap_frames), ('smoothing_window', smoothing_window)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer or None.')

    if smoothing_window is not None:
        if smoothing_window < 3 or smoothing_window % 2 == 0:
            raise ValueError('smoothing_window must be odd and at least three frames.')
        if smoothing_window > len(times):
            raise ValueError('smoothing_window cannot exceed the recording length.')

    # === 3| Copy Raw Pose Before Replacing Low-Confidence Coordinates =============

    pose = raw_pose.copy(deep=True)                  # The loaded tracking remains available for raw/processed comparisons.
    position = pose.position
    if confidence_threshold is not None:
        position = filter_by_confidence(
            position,
            pose.confidence,
            threshold=float(confidence_threshold),   # Mask positions with low or missing DLC confidence.
            print_report=False,                      # The notebook controls which processing summaries it displays.
        )

    # === 4| Fill Only the Requested Short Internal Gaps ===========================

    if max_gap_frames is not None:
        position = interpolate_over_time(
            position,
            method='linear',                         # Join the valid samples on either side using movement's sample-order interpolation.
            max_gap=int(max_gap_frames),             # Preserve missing runs longer than the explicit frame limit.
            print_report=False,
        )

    # === 5| Apply a Median Only Where a Full Valid Window Exists =================

    if smoothing_window is not None:
        position = rolling_filter(
            position,
            window=int(smoothing_window),
            statistic='median',                      # Reduce isolated coordinate spikes without changing the recording's time axis.
            min_periods=int(smoothing_window),       # Do not silently replace unresolved gaps using incomplete windows.
            print_report=False,
        )

    # === 6| Retain Confidence and Verify That the Acquired Clock Was Preserved ====

    if not position.time.equals(raw_pose.time):
        raise ValueError('Pose processing changed the acquired time coordinate.')

    pose['position'] = position                      # Replace only position; confidence and source coordinates remain available.
    return pose


# ===============================================================================




# """
# Build the lab's DataStructure from a root directory and source selection.
# ------------------------------------------------------------------------

# Description:
#     Reuse refactor_qc's readers and the generic DataStructure selection API.
#     Call build_datastructure(...) to configure one or many recordings, then
#     call its .load() method to obtain the StreamMap of pandas/xarray objects.

#     The existing Q_C catalog parses trials and aligns DLC to video timestamps.
#     Its legacy source name 'events' produces 'trials'; raw_events=True also
#     retains the event log as 'events'. Pose filtering belongs in the analysis.

# Contents:
# --------------------------------
# - _read_raw_events:      Adapt the existing event reader to a catalog entry.
# - build_datastructure:  Configure the lab's existing DataStructure.
# """


# ################################################################################
# # Imports
# ################################################################################

# from collections.abc import Sequence
# from pathlib import Path
# from typing import Any

# import pandas as pd

# from data_conduit.core.utils import _concat_split_dataframes
# from data_conduit.datasources.monosource import ExperimentEvents
# from data_conduit.datastructures import DataStructure
# from data_conduit.refactor_qc import QC_STREAMS, qc_datastructure

# ################################################################################


# ################################################################################
# # Source Adapter
# ################################################################################



# # ===============================================================================
# # 1| Read the Session's Event Log
# # ===============================================================================


# def _read_raw_events(session_path: Path) -> pd.DataFrame:
#     """
#     Read a session's event log using the existing data-conduit reader.

#     Parameters
#     ----------
#     session_path : Path
#         Recording directory passed in by the catalog.

#     Returns
#     -------
#     pandas.DataFrame
#         Logged events with their original time index; split files are combined.
#     """
#     events = ExperimentEvents(experiment_directory_path=session_path)
#     return _concat_split_dataframes(events.df)  # Handle single and split recordings alike.


# ################################################################################
# # Public API
# ################################################################################


# # ===============================================================================
# # 1| Configure the Lab's DataStructure
# # ===============================================================================


# def build_datastructure(
#     root: str | Path,                               # Directory above the named hierarchy levels.
#     *,
#     level_names: Sequence[str] = ("mouseID", "day"),  # Lab layout: root / mouseID / day / session.
#     streams: Sequence[str] = QC_STREAMS,             # Source choices supported by refactor_qc.
#     include: Sequence[str] | None = None,            # Keep these session folder names; None keeps all.
#     exclude: Sequence[str] | None = None,            # Alternatively omit these session folder names.
#     raw_events: bool = False,                        # Also retain the log used to derive trials.
#     device_yaml: str | Path = "./device.yml",        # Lab Nosepoke board register definitions.
#     soundcard_yaml: str | Path = "./soundcard.yml",  # Lab SoundCard register definitions.
#     nosepoke_count: int = 18,                        # Port count used by the existing Q_C parser.
#     trial_start_buffer: float = 0.0,                 # Preserve the existing lab parser setting.
#     dlc_file_format: str = "auto",                   # Existing reader prefers CSV, otherwise HDF5.
#     **level_selectors: Any,                         # For example l0_selector=["mouse_A", "mouse_B"].
# ) -> DataStructure:
#     """
#     Configure data loading for the selected mice, days and session folders.

#     This constructs the existing Q_C DataStructure. Recording files are read
#     when the caller runs .load(), which returns the ordinary StreamMap.

#     Parameters
#     ----------
#     root : str | Path
#         Directory containing the experimental hierarchy.
#     level_names : Sequence[str]
#         Intermediate folder labels; use () when root directly contains sessions.
#     streams : Sequence[str]
#         Q_C sources: events, nosepoke, soundcard, session_settings, video, dlc.
#         Defaults to all six. The existing catalog always produces parsed trials;
#         requesting dlc also reads video timestamps for alignment.
#     include, exclude : Sequence[str] | None
#         Session folder names to keep or omit. Supply at most one of these lists.
#     raw_events : bool
#         Add the original event log under the 'events' stream key. Default False.
#     device_yaml, soundcard_yaml : str | Path
#         Device definitions used when loading the corresponding HARP sources.
#     nosepoke_count : int
#         Number of arena ports for the Q_C parser. Default 18.
#     trial_start_buffer : float
#         Existing parser's offset from the previous trial end, in seconds.
#     dlc_file_format : str
#         'csv', 'h5', or 'auto' to prefer CSV when available.
#     **level_selectors : Any
#         Existing l0_selector, l1_selector, etc.; each can select multiple folders.

#     Returns
#     -------
#     DataStructure
#         Configured object. .load() returns its streams; .sessions holds their
#         source directories. Session and hierarchy labels accompany loaded data.
#     """

#     # === 1| Reuse the Lab Recipe and Generic Directory Selection =================

#     datastructure = qc_datastructure(
#         root=root,
#         depth=len(level_names),             # One directory level per supplied hierarchy name.
#         level_names=level_names,
#         streams=streams,
#         include=include,
#         exclude=exclude,
#         device_yaml=device_yaml,
#         soundcard_yaml=soundcard_yaml,
#         nosepoke_count=nosepoke_count,
#         trial_start_buffer=trial_start_buffer,
#         dlc_file_format=dlc_file_format,
#         **level_selectors,
#     )

#     # === 2| Retain Raw Events Only When Requested ================================

#     if raw_events:
#         datastructure.catalog.add_reader("events", _read_raw_events)

#     return datastructure


# # ===============================================================================
