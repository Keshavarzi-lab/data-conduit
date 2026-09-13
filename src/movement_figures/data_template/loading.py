"""
Build the lab's DataStructure from a root directory and source selection.
------------------------------------------------------------------------

Description:
    Reuse refactor_qc's readers and the generic DataStructure selection API.
    Call build_datastructure(...) to configure one or many recordings, then
    call its .load() method to obtain the StreamMap of pandas/xarray objects.

    The existing Q_C catalog parses trials and aligns DLC to video timestamps.
    Its legacy source name 'events' produces 'trials'; raw_events=True also
    retains the event log as 'events'. Pose filtering belongs in the analysis.

Contents:
--------------------------------
- _read_raw_events:      Adapt the existing event reader to a catalog entry.
- build_datastructure:  Configure the lab's existing DataStructure.
"""


################################################################################
# Imports
################################################################################

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from data_conduit.core.utils import _concat_split_dataframes
from data_conduit.datasources.monosource import ExperimentEvents
from data_conduit.datastructures import DataStructure
from data_conduit.refactor_qc import QC_STREAMS, qc_datastructure

################################################################################


################################################################################
# Source Adapter
################################################################################



# ===============================================================================
# 1| Read the Session's Event Log
# ===============================================================================


def _read_raw_events(session_path: Path) -> pd.DataFrame:
    """
    Read a session's event log using the existing data-conduit reader.

    Parameters
    ----------
    session_path : Path
        Recording directory passed in by the catalog.

    Returns
    -------
    pandas.DataFrame
        Logged events with their original time index; split files are combined.
    """
    events = ExperimentEvents(experiment_directory_path=session_path)
    return _concat_split_dataframes(events.df)  # Handle single and split recordings alike.


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
    streams: Sequence[str] = QC_STREAMS,             # Source choices supported by refactor_qc.
    include: Sequence[str] | None = None,            # Keep these session folder names; None keeps all.
    exclude: Sequence[str] | None = None,            # Alternatively omit these session folder names.
    raw_events: bool = False,                        # Also retain the log used to derive trials.
    device_yaml: str | Path = "./device.yml",        # Lab Nosepoke board register definitions.
    soundcard_yaml: str | Path = "./soundcard.yml",  # Lab SoundCard register definitions.
    nosepoke_count: int = 18,                        # Port count used by the existing Q_C parser.
    trial_start_buffer: float = 0.0,                 # Preserve the existing lab parser setting.
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
        Q_C sources: events, nosepoke, soundcard, session_settings, video, dlc.
        Defaults to all six. The existing catalog always produces parsed trials;
        requesting dlc also reads video timestamps for alignment.
    include, exclude : Sequence[str] | None
        Session folder names to keep or omit. Supply at most one of these lists.
    raw_events : bool
        Add the original event log under the 'events' stream key. Default False.
    device_yaml, soundcard_yaml : str | Path
        Device definitions used when loading the corresponding HARP sources.
    nosepoke_count : int
        Number of arena ports for the Q_C parser. Default 18.
    trial_start_buffer : float
        Existing parser's offset from the previous trial end, in seconds.
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

    # === 1| Reuse the Lab Recipe and Generic Directory Selection =================

    datastructure = qc_datastructure(
        root=root,
        depth=len(level_names),             # One directory level per supplied hierarchy name.
        level_names=level_names,
        streams=streams,
        include=include,
        exclude=exclude,
        device_yaml=device_yaml,
        soundcard_yaml=soundcard_yaml,
        nosepoke_count=nosepoke_count,
        trial_start_buffer=trial_start_buffer,
        dlc_file_format=dlc_file_format,
        **level_selectors,
    )

    # === 2| Retain Raw Events Only When Requested ================================

    if raw_events:
        datastructure.catalog.add_reader("events", _read_raw_events)

    return datastructure


# ===============================================================================
