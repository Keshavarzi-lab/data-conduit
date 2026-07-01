'''
Build, from scratch, the catalog of data objects to extract per session.
========================================================================

Description:
    A data-conduit ``DataStructureCatalog`` is a shopping list: it names WHICH data
    objects to pull out of each session folder and HOW to read each one. The loader
    (``build_sessions`` in loading.py) walks this list once per session.

    data-conduit ships a convenience builder, ``default_harp_catalog``, that fills a
    catalog for you. We deliberately do NOT use it here. Instead ``build_catalog``
    below assembles the same thing by hand, one ``DataStructureSpec`` at a time, so
    that a reader can see exactly how a catalog is declared and copy the pattern for
    their own experiment. Each spec is one ``DataStructureSpec(name=..., reader=...)``
    where ``reader`` is a tiny function ``path -> loaded object`` that the loader calls
    with each session's directory.

    What we extract (in order):
      * events            ExperimentEvents CSV. REQUIRED: it carries the master
                          Bonsai/HARP clock that every other stream is aligned to.
      * nosepoke          The 18-port Nosepoke HARP MultiDevice (poke activations).
      * soundcard         The SoundCard HARP device (tone on/off, frequencies).
      * session_settings  SessionSettings (per-session metadata and the trial list).
      * video             VideoData CSV (per-frame camera timestamps).
      * dlc               DeepLabCut pose, loaded FRAME-INDEXED via ``DLCPose``.
                          This reader does NOT attach camera times: it knows
                          nothing about VideoData. To put pose on the session
                          clock, call ``align_pose_to_video`` with the loaded
                          ``video`` stream as an explicit step after loading
                          (see data_conduit.datasources.pose).

    We omit the HARP ``camera`` (Camera0Frames) spec because this rig logs camera
    timing as VideoData CSVs, not a Camera0Frames register folder.

Contents:
--------------------------------
- build_catalog:    Assemble and return the DataStructureCatalog described above.
'''

from __future__ import annotations

from pathlib import Path

# The catalog/spec types and the shared flatten helper are the only data-conduit pieces
# this module needs at import time. The actual reader classes are imported lazily inside
# build_catalog (see there) so that merely importing this module never requires the
# optional harp-python package.
from data_conduit.sessiongroups import DataStructureCatalog, DataStructureSpec
from data_conduit.utils.utils_core import _concat_split_dataframes

# Repo root, used to locate the default HARP device YAMLs that live there by convention.
from pose_trajectory import REPO_ROOT


def build_catalog(
        *,
        device_yaml: str | Path | None = None,
        soundcard_yaml: str | Path | None = None,
        include_harp: bool = True,
        include_video: bool = True,
        include_session_settings: bool = True,
) -> DataStructureCatalog:
    '''
    Assemble the per-session extraction catalog by hand and return it.

    The catalog is editable after the fact: a caller can still
    ``catalog.disable('soundcard')`` or ``catalog.add(...)`` before loading. All
    non-events specs are ``required=False`` so a session missing any one of them
    loads fine (that stream is simply skipped with a warning).

    ----------
    Parameters:
        device_yaml (str | Path | None):
            Path to the HARP device YAML describing the Behavior boards (needed by
            the Nosepoke reader). If None (default), uses ``<repo>/device.yml``.
        soundcard_yaml (str | Path | None):
            Path to the HARP SoundCard YAML (needed by the SoundCard reader). If None
            (default), uses ``<repo>/soundcard.yml``.
        include_harp (bool):
            If True (default), include the ``nosepoke`` and ``soundcard`` HARP specs.
            Set False for a pose-only run with no HARP YAMLs available; you still get
            events, video, session_settings, and dlc.
        include_video (bool):
            If True (default), include the VideoData CSV spec.
        include_session_settings (bool):
            If True (default), include the SessionSettings spec (its two sub-tables
            land as ``session_settings:metadata`` and ``session_settings:trials``).
    Returns:
        DataStructureCatalog:
            The assembled, still-editable catalog.
    '''

    # 1| Import the reader classes lazily, inside the function. The HARP presets pull
    #    in the optional harp-python package, and we do not want importing this module
    #    to fail on a machine that only does pose/events work without HARP installed.
    from data_conduit.datasources.monosource import (
        ExperimentEvents,
        SessionSettings,
        VideoData,
    )
    from data_conduit.datasources.pose import DLCPose

    # 2| Resolve the HARP YAML paths, defaulting to the repo-root copies by convention.
    device_yaml = Path(device_yaml) if device_yaml is not None else REPO_ROOT / 'device.yml'
    soundcard_yaml = Path(soundcard_yaml) if soundcard_yaml is not None else REPO_ROOT / 'soundcard.yml'

    # 3| Start the spec list with events. It is required=True because it provides the
    #    reference clock the rest of the session is aligned to: a session with no events
    #    cannot be aligned, so we want a clear error rather than a silent skip. The
    #    reader returns the ExperimentEvents .df (Time-indexed, with an 'Event' column),
    #    flattened in case the session split its log into multiple files.
    specs: list[DataStructureSpec] = [
        DataStructureSpec(
            name='events',
            reader=lambda p: _concat_split_dataframes(ExperimentEvents(experiment_directory_path=p).df),
            required=True,
        ),
    ]

    # 4| HARP streams (optional). Each reader returns the whole source OBJECT; the
    #    loader then reads its .data_arrays. They need their device YAML, captured here
    #    in the lambda so the reader stays a one-argument path -> object function.
    if include_harp:
        from data_conduit.datasources.presets.harp import Nosepoke, SoundCard

        specs.append(
            DataStructureSpec(
                name='nosepoke',
                reader=lambda p: Nosepoke(experiment_directory_path=p, harp_device_yaml_path=device_yaml),
            )
        )
        specs.append(
            DataStructureSpec(
                name='soundcard',
                reader=lambda p: SoundCard(experiment_directory_path=p, harp_device_yaml_path=soundcard_yaml),
            )
        )

    # 5| SessionSettings (optional). Its reader returns {'metadata': df, 'trials': df},
    #    two sub-tables with different schemas, so we leave them split (no flatten) and
    #    they land as 'session_settings:metadata' / 'session_settings:trials'.
    if include_session_settings:
        specs.append(
            DataStructureSpec(
                name='session_settings',
                reader=lambda p: SessionSettings(experiment_directory_path=p).df,
            )
        )

    # 6| VideoData (optional). Per-frame camera metadata; its 'Seconds' column is the
    #    HARP timestamp of each frame, the same clock DLCPose joins pose onto. Flattened
    #    to one DataFrame in case the recording was split into segments.
    if include_video:
        specs.append(
            DataStructureSpec(
                name='video',
                reader=lambda p: _concat_split_dataframes(VideoData(experiment_directory_path=p).df),
            )
        )

    # 7| DeepLabCut pose. DLCPose reads the DLC output(s) into FRAME-INDEXED arrays:
    #    it does NOT touch VideoData and does NOT attach times. It exposes two arrays
    #    that become the members 'dlc:position' (frame x keypoints x space) and
    #    'dlc:confidence' (frame x keypoints). required=False so non-pose sessions skip.
    #    To put pose on the session clock, call align_pose_to_video with the loaded
    #    'video' stream as an explicit step after loading.
    specs.append(
        DataStructureSpec(
            name='dlc',
            reader=lambda p: DLCPose(experiment_directory_path=p),
        )
    )

    # 8| Hand the ordered specs to a catalog and return it. Order is preserved, so this
    #    is also the order the loader reads streams and the order they appear in repr.
    return DataStructureCatalog(specs)
