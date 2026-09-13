'''
Configured Q_C DataStructure: generic readers, parsing and alignment.
=====================================================================

Description:
    Build a ready-to-use ``DataStructure`` for the Q_C nosepoke / pose task. It
    wires the lab's source readers onto data-conduit's generic Catalog /
    DataStructure front end:

      * Readers (opt-in via ``streams``): trials, events, nosepoke, soundcard,
        session_settings, video, dlc. ``trials`` always means parsed rows. For
        existing refactor_qc notebooks, ``events`` still means trials unless
        ``legacy_events_as_trials=False`` is supplied; then it means raw events.
        Raw events and trials share one event-table read per recording. Stable
        event IDs join their exact row windows even when timestamps tie.
      * Configurator ``align_dlc`` (only when ``dlc`` is requested): stamps the
        VideoData frame times onto the frame-indexed DLC pose, putting pose on the
        same Bonsai Seconds clock as the events / trial times.

    The factory is LAYOUT-AGNOSTIC: you pass ``depth`` and ``level_names`` for the
    directory tree you have. Two layouts are in play:
      * Q_C_Analysis_Workflow/data : <mouse>/<phase>/<day>/<session>  (depth 3,
        level_names=('mouseID', 'phase', 'day')); filter the phase with
        ``l1_selector='Testing'``. This tree currently has NO DLC.
      * datasets/firstdata/BonsaiFiles : <mouse>/<day>/<session>      (depth 2,
        level_names=('mouseID', 'day')); some days (e.g. Day 11) DO have DLC, so
        use this to develop the pose side.

    One ``load()`` returns only requested products with DataStructure's existing
    recording provenance. DLC needs video timestamps but never requires events
    or trials. A video loaded only for DLC alignment is removed before combining.

Contents:
--------------------------------
- build_qc_catalog: Assemble the generic Catalog recipe for the Q_C task.
- qc_datastructure: Wrap that Catalog in a DataStructure for a given root + layout.
'''





################################################################################
# Imports
################################################################################

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from data_conduit.datastructures import DataStructure, StreamCatalog, TrialSpec

################################################################################


def _resolve_harp_yaml_path(path: str | Path) -> Path:
    '''Resolve a HARP YAML path robustly for notebooks/scripts run outside the repo root.'''
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate

    repo_root_candidate = Path(__file__).resolve().parents[3] / candidate
    if repo_root_candidate.exists():
        return repo_root_candidate

    return candidate



# ``trials`` explicitly requests parsed rows. ``events`` requests the raw log
# when legacy_events_as_trials=False; True keeps existing notebook calls valid.
QC_STREAMS = ('trials', 'events', 'nosepoke', 'soundcard', 'session_settings', 'video', 'dlc')




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| build_qc_catalog (Q_C Reader Recipe + Optional DLC Alignment)
#===============================================================================
def build_qc_catalog(
        *,
        streams: Sequence[str] = ('events', 'nosepoke', 'soundcard', 'session_settings', 'video', 'dlc'),
        device_yaml: str | Path = './device.yml',               # Nosepoke-board HARP description.
        soundcard_yaml: str | Path = './soundcard.yml',         # Sound-card HARP description.
        nosepoke_count: int = 18,                               # Arena geometry used by the default Q_C spec.
        trial_start_buffer: float = 0.0,                        # Seconds added after the previous trial ends.
        dlc_file_format: str = 'auto',                          # Existing CSV/HDF5 pose-reader choice.
        trial_spec: TrialSpec | None = None,                    # Optional replacement for the Q_C trial rules.
        legacy_events_as_trials: bool = True,                   # Preserve old events -> trials requests.
) -> StreamCatalog:
    """
    Assemble the Q_C reader recipe using the existing StreamCatalog stages.

    Each requested physical source is read once per recording. When trials are
    requested, a configurator parses the already loaded event table. That same
    table can also be retained as ``events`` without another file read. Stable
    ``event_index`` values connect the raw rows to the parser's included bounds.

    Parameters
    ----------
    streams : Sequence[str]
        Requested products from ``QC_STREAMS``. ``trials`` always requests parsed
        trial rows. By default the old token ``events`` also means trials; set
        ``legacy_events_as_trials=False`` to use ``events`` for the raw log.
        The default requests trials, nosepoke, soundcard, settings, video and DLC.
        Nothing adds trials implicitly. DLC adds video only for frame alignment.
    device_yaml, soundcard_yaml : str | pathlib.Path
        Device descriptions for the corresponding optional HARP readers. Relative
        paths are resolved against the current directory or repository root.
    nosepoke_count : int
        Arena port count passed to qc_trial_spec when trial_spec is None. Default 18.
    trial_start_buffer : float
        Inter-trial seconds passed to qc_trial_spec when trial_spec is None.
        Default 0.0. Supplying trial_spec makes that spec's own buffer authoritative.
    dlc_file_format : str
        DLC export format: 'csv', 'h5', or 'auto' (CSV if available, otherwise HDF5).
    trial_spec : TrialSpec | None
        Optional experiment description, including boundary, inclusion and buffer
        policies. None builds the existing Q_C spec using the two arguments above.
        A supplied spec replaces those rules; nosepoke_count and trial_start_buffer
        are then unused. No spec is constructed when trials are not requested.
    legacy_events_as_trials : bool
        True retains the meaning used by existing refactor_qc notebooks. False
        makes 'events' mean raw events, so request ('events', 'trials') for both.
        An explicit 'trials' token works in either mode. Default True.

    Returns
    -------
    StreamCatalog
        Existing reader/configurator recipe ready for DataStructure. An absent
        optional DLC source contributes no pose. Present DLC without video timing
        raises rather than returning pose with unaligned frame numbers.
    """

    # === 1| Resolve Existing HARP Paths and Import the Requested Pipeline =========
    # Source-reader imports remain local: constructing the recipe should not
    # import HARP or DLC dependencies unless those sources are requested.

    device_yaml = _resolve_harp_yaml_path(device_yaml)
    soundcard_yaml = _resolve_harp_yaml_path(soundcard_yaml)

    from data_conduit.core.utils import _concat_split_dataframes
    from data_conduit.datasources.monosource import SessionSettings, VideoData
    from data_conduit.datastructures import StreamCatalog as Catalog, parse_trials
    from data_conduit.refactor_qc.trial_spec import qc_trial_spec, read_qc_events

    # === 2| Resolve Product Names Without Adding Unrequested Trials ==============
    # Translate only the historical events token. A caller choosing raw events
    # opts out explicitly; old refactor_qc notebooks retain their trials output.

    if isinstance(streams, str):
        raise TypeError("streams must be a sequence of names, for example ('trials',).")

    requested = list(streams)                                   # Copy before adding a video dependency.
    video_explicit = 'video' in requested                       # Remember whether to return video itself.
    unknown = [name for name in requested if name not in QC_STREAMS]
    if unknown:
        raise ValueError(f'unknown streams {unknown}; valid: {list(QC_STREAMS)}.')

    if legacy_events_as_trials and 'events' in requested:
        requested.remove('events')                              # Old token names the parsed product.
        requested.append('trials')                              # Keep the existing returned stream name.

    events_explicit = 'events' in requested                     # Raw events survive configuration only if asked for.
    if 'dlc' in requested and 'video' not in requested:
        requested.append('video')                               # Pose alignment needs camera timestamps.

    catalog = Catalog()                                         # Reuse the existing per-recording recipe.

    # === 3| Register the Existing Physical Readers ===============================
    # These readers keep their established return types. Video is optional only
    # when pulled in for optional DLC; explicitly requested video stays required.

    if 'nosepoke' in requested:
        from data_conduit.integrations.harp.datasource_presets import Nosepoke
        catalog.add_reader(
            'nosepoke',
            lambda p: Nosepoke(experiment_directory_path=p, harp_device_yaml_path=device_yaml),
        )
    if 'soundcard' in requested:
        from data_conduit.integrations.harp.datasource_presets import SoundCard
        catalog.add_reader(
            'soundcard',
            lambda p: SoundCard(experiment_directory_path=p, harp_device_yaml_path=soundcard_yaml),
        )
    if 'session_settings' in requested:
        catalog.add_reader(
            'session_settings',
            lambda p: SessionSettings(experiment_directory_path=p).df,
        )
    if 'video' in requested:
        # Return the flattened video table (one DataFrame), not the VideoData
        # object: a multi-file session's object nests into split members, whereas
        # the flattened table is one clean 'video' member, and the aligner reads
        # its Time index for the frame times all the same.
        def _read_video(
                path: Path,                                     # One recording selected by DataStructure.
        ) -> pd.DataFrame:                                      # Flattened video table for the existing aligner.
            """
            Read frame timestamps without changing their acquisition order.

            The pose reader concatenates its rows in filename/frame order. Camera
            rows must keep that same order before their times are assigned to pose;
            sorting the camera clock alone would pair different frames. A missing
            source is optional when only DLC requested it, but an invalid acquired
            clock raises even when every requested file is present.

            Parameters
            ----------
            path : pathlib.Path
                Recording directory searched by the existing VideoData reader.

            Returns
            -------
            pandas.DataFrame
                One flattened camera timestamp table on a finite, strictly
                increasing Time index, with each file's original row order intact.

            Raises
            ------
            FileNotFoundError
                If VideoData found no files. Other reading/format errors propagate.
            TypeError, ValueError
                If the acquired camera times are not real numeric values, are
                nonfinite, repeat, decrease, or contain no frames.
            """

            # === 1| Translate Only a Truly Absent Video Source into FileNotFound ===
            # VideoData returns {} when no files match. The catalog's optional
            # reader mechanism only handles FileNotFoundError, not malformed data.

            frames = VideoData(experiment_directory_path=path).df
            if isinstance(frames, dict) and not frames:
                raise FileNotFoundError(f'no VideoData timestamps found in {path}.')

            # === 2| Join Camera Files Without Reordering Their Frame Rows =========
            # The shared helper normally sorts event rows by time. Disable that
            # sorting for video, whose row positions must still match DLC frames.

            frames = _concat_split_dataframes(
                frames,
                sort_by_time=False,                            # Preserve each camera file's acquired frame order.
                on_rollback='ignore',                          # Validate the complete clock once below, including single-file input.
            )

            # === 3| Reject an Invalid Clock Before Assigning Any Times to Pose =====
            # Compare adjacent values directly: subtracting unsigned timestamps
            # can wrap at a rollback and incorrectly make the gap look positive.

            times = frames.index.to_numpy()
            if not (np.issubdtype(times.dtype, np.integer) or np.issubdtype(times.dtype, np.floating)):
                raise TypeError('VideoData times must contain real numeric acquired seconds.')
            if len(times) == 0 or not np.isfinite(times).all() or np.any(times[1:] <= times[:-1]):
                raise ValueError('VideoData times must be non-empty, finite and strictly increasing in acquired frame order.')

            return frames                                      # Valid camera times still refer to their original frame rows.

        catalog.add_reader(
            'video',
            _read_video,                                        # Handle real absence without hiding parse failures.
            optional=not video_explicit,                        # Explicit video remains a required source.
        )
    if 'dlc' in requested:
        from data_conduit.integrations.DLC.pose import DLCPose

        # Selected sessions without generated pose remain in the load; they simply
        # contribute no DLC streams, as the original Q_C notebooks expect.
        catalog.add_reader(
            'dlc',
            lambda p: DLCPose(experiment_directory_path=p, file_format=dlc_file_format),
            optional=True,
        )

    # === 4| Read Events Once, Then Derive the Requested Trial Table ================
    # Both products need the same stable ordering and event IDs. Register one
    # raw reader; the configurator decides whether to return events, trials or both.

    if events_explicit or 'trials' in requested:
        catalog.add_reader('events', read_qc_events)            # One disk read for each selected recording.

    if 'trials' in requested:
        spec = trial_spec
        if spec is None:
            spec = qc_trial_spec(
                nosepoke_count=nosepoke_count,                  # Existing task geometry for angular offsets.
                trial_start_buffer=trial_start_buffer,          # Existing zero-buffer catalog convention.
            )

        def _parse_trials(
                objects: dict[str, object],                     # Reader outputs for one recording.
        ) -> dict[str, object]:                                 # Same outputs plus the parsed trials table.
            """
            Parse the loaded events, retaining the raw table only if requested.

            Parameters
            ----------
            objects : dict[str, object]
                One recording's reader outputs, including required 'events'.

            Returns
            -------
            dict[str, object]
                Copy containing 'trials' and all requested physical streams.
                'events' remains only when the caller requested the raw log.
            """

            # === 1| Parse the Same Event Table That Supplies Raw Event Output =====

            result = dict(objects)                              # Preserve the reader-output mapping.
            result['trials'] = parse_trials(result['events'], spec)

            # === 2| Remove the Internal Input Unless Raw Events Were Requested ====

            if not events_explicit:
                result.pop('events')                            # Trials-only output carries no duplicate log.
            return result

        catalog.add_configurator('parse_trials', _parse_trials)

    # === 5| Align Present DLC to Camera Time, Then Remove Internal Video ==========
    # This stage is independent of trials. An optional absent pose is fine; a
    # present pose without camera timing cannot be returned as aligned data.

    if 'dlc' in requested:
        from data_conduit.integrations.DLC.pose import align_pose_to_video

        def _align_dlc(
                objects: dict[str, object],                     # This recording's loaded/configured objects.
        ) -> dict[str, object]:                                 # Pose aligned; dependency-only video removed.
            """
            Assign camera timestamps to available pose using the existing aligner.

            Parameters
            ----------
            objects : dict[str, object]
                Per-recording mapping that may contain optional DLC and video.

            Returns
            -------
            dict[str, object]
                Copy with aligned DLC when present. Dependency-only video is
                removed even when the recording contains no pose.

            Raises
            ------
            FileNotFoundError
                If DLC was loaded but no video timestamps were available.
            """

            # === 1| Require Camera Times Only When Pose Was Actually Loaded =======

            result = dict(objects)
            if 'dlc' in result:
                if 'video' not in result:
                    raise FileNotFoundError('DLC pose requires VideoData timestamps for alignment.')
                result['dlc'] = align_pose_to_video(result['video'], result['dlc'])

            # === 2| Return Video Only When It Was an Explicitly Requested Product ==

            if not video_explicit:
                result.pop('video', None)                       # No dependency-only video leaks from DLC-less recordings.
            return result

        catalog.add_configurator('align_dlc', _align_dlc)

    return catalog

#===============================================================================



#===============================================================================
# 2| qc_datastructure (Catalog + Root + Layout -> DataStructure)
#===============================================================================
def qc_datastructure(
        root: str | Path,                                       # Root containing the recording directory tree.
        *,
        depth: int,                                             # Number of intermediate levels above recordings.
        level_names: Sequence[str],                             # Directory levels retained as provenance.
        streams: Sequence[str] = ('events', 'nosepoke', 'soundcard', 'session_settings', 'video', 'dlc'),
        include: Sequence[str] | None = None,                   # Optional recording-name allowlist.
        exclude: Sequence[str] | None = None,                   # Optional recording-name denylist.
        device_yaml: str | Path = './device.yml',               # Nosepoke-board HARP description.
        soundcard_yaml: str | Path = './soundcard.yml',         # Sound-card HARP description.
        nosepoke_count: int = 18,                               # Arena geometry used by the default Q_C spec.
        trial_start_buffer: float = 0.0,                        # Seconds added after the previous trial ends.
        dlc_file_format: str = 'auto',
        trial_spec: TrialSpec | None = None,                    # Forward custom rules without another wrapper class.
        legacy_events_as_trials: bool = True,                   # Same name policy as build_qc_catalog.
        **level_selectors,
) -> DataStructure:
    '''
    Build a configured Q_C DataStructure for a given root directory and layout.

    Convenience wrapper: assembles the Q_C catalog (via ``build_qc_catalog``) and
    points a DataStructure at ``root`` with the directory layout and selectors you
    supply. Call ``.load()`` on the result to get the requested streams combined
    across selected recordings, including trials only when requested.

    ----------
    Parameters:
        root (str | Path):
            The data directory to search.
        depth (int):
            Levels below ``root`` where the session folders sit (e.g. 3 for
            Q_C_Analysis_Workflow mouse/phase/day/session, 2 for firstdata
            mouse/day/session).
        level_names (Sequence[str]):
            Names for the intermediate directory levels, retained as provenance
            columns (e.g. ``('mouseID', 'phase', 'day')`` or ``('mouseID', 'day')``).
        streams (Sequence[str]):
            Requested products; see ``build_qc_catalog``. The default requests
            trials and all physical sources. No trials are added to DLC-only calls.
        include (Sequence[str] | None):
            INCLUDE-mode session-name allowlist.
        exclude (Sequence[str] | None):
            EXCLUDE-mode session-name denylist.
        device_yaml, soundcard_yaml (str | Path):
            HARP YAMLs (only needed when nosepoke / soundcard are in ``streams``).
        nosepoke_count (int):
            Arena port count for the trial parser. Default 18.
        trial_start_buffer (float):
            Inter-trial buffer for the trial parser. Default 0.0.
        dlc_file_format (str):
            DLC export the pose reader loads: ``'csv'``, ``'h5'``, or ``'auto'``
            (csv if present, else h5). Default ``'auto'``.
        trial_spec (TrialSpec | None):
            Optional trial rules forwarded unchanged. When supplied, its own
            buffer and fields replace nosepoke_count/trial_start_buffer defaults.
        legacy_events_as_trials (bool):
            True preserves old events -> trials requests. False makes events
            return the raw log; request trials explicitly when needed.
        **level_selectors:
            ``l{n}_selector`` filters for the intermediate levels (e.g.
            ``l1_selector='Testing'`` to keep only the Testing phase).
    Returns:
        DataStructure:
            The configured DataStructure; call ``.load()`` to run it.
    '''
    from data_conduit.datastructures import DataStructure

    # === 1| Build the Per-Recording Recipe Using the Requested Rules ==============

    catalog = build_qc_catalog(
        streams=streams,
        device_yaml=device_yaml,
        soundcard_yaml=soundcard_yaml,
        nosepoke_count=nosepoke_count,
        trial_start_buffer=trial_start_buffer,
        dlc_file_format=dlc_file_format,
        trial_spec=trial_spec,                                  # Carry the exact supplied rule object into the catalog.
        legacy_events_as_trials=legacy_events_as_trials,        # Keep stream naming identical through this wrapper.
    )

    # === 2| Reuse DataStructure for Discovery, Selection and Stream Combination ===

    return DataStructure(
        root,
        catalog,
        depth=depth,
        level_names=level_names,
        include=include,
        exclude=exclude,
        **level_selectors,
    )

#===============================================================================



################################################################################



# '''
# Configured Q_C DataStructure: generic readers, parsing and alignment.
# =====================================================================

# Description:
#     Build a ready-to-use ``DataStructure`` for the Q_C nosepoke / pose task. It
#     wires the lab's source readers onto data-conduit's generic Catalog /
#     DataStructure front end:

#       * Readers (opt-in via ``streams``): events, nosepoke, soundcard,
#         session_settings, video, dlc. The legacy ``events`` request selects a
#         ``trials`` reader that loads the event log and applies the Q_C
#         ``TrialSpec`` with the generic parser. Raw events remain an internal
#         input and are not concatenated across sessions.
#       * Configurator ``align_dlc`` (only when ``dlc`` is requested): stamps the
#         VideoData frame times onto the frame-indexed DLC pose, putting pose on the
#         same Bonsai Seconds clock as the events / trial times.

#     The factory is LAYOUT-AGNOSTIC: you pass ``depth`` and ``level_names`` for the
#     directory tree you have. Two layouts are in play:
#       * Q_C_Analysis_Workflow/data : <mouse>/<phase>/<day>/<session>  (depth 3,
#         level_names=('mouseID', 'phase', 'day')); filter the phase with
#         ``l1_selector='Testing'``. This tree currently has NO DLC.
#       * datasets/firstdata/BonsaiFiles : <mouse>/<day>/<session>      (depth 2,
#         level_names=('mouseID', 'day')); some days (e.g. Day 11) DO have DLC, so
#         use this to develop the pose side.

#     One ``load()`` then returns the cross-session trial table (with mouseID / day /
#     session provenance columns) alongside any other requested streams.

# Contents:
# --------------------------------
# - build_qc_catalog: Assemble the generic Catalog recipe for the Q_C task.
# - qc_datastructure: Wrap that Catalog in a DataStructure for a given root + layout.
# '''





# ################################################################################
# # Imports
# ################################################################################

# from pathlib import Path

# ################################################################################


# def _resolve_harp_yaml_path(path: str | Path) -> Path:
#     '''Resolve a HARP YAML path robustly for notebooks/scripts run outside the repo root.'''
#     candidate = Path(path)
#     if candidate.is_absolute() or candidate.exists():
#         return candidate

#     repo_root_candidate = Path(__file__).resolve().parents[3] / candidate
#     if repo_root_candidate.exists():
#         return repo_root_candidate

#     return candidate



# # The established notebook request names. ``'events'`` is retained as the legacy
# # request for the parsed trials product; the remaining physical sources are opt-in.
# QC_STREAMS = ('events', 'nosepoke', 'soundcard', 'session_settings', 'video', 'dlc')




# ################################################################################
# # Public API
# ################################################################################



# #===============================================================================
# # 1| build_qc_catalog (Q_C Reader Recipe + Optional DLC Alignment)
# #===============================================================================
# def build_qc_catalog(
#         *,
#         streams=('events', 'nosepoke', 'soundcard', 'session_settings', 'video', 'dlc'),
#         device_yaml: str | Path = './device.yml',
#         soundcard_yaml: str | Path = './soundcard.yml',
#         nosepoke_count: int = 18,
#         trial_start_buffer: float = 0.0,
#         dlc_file_format: str = 'auto',
# ):
#     '''
#     Assemble the Q_C-specific reader recipe on a generic ``StreamCatalog``.

#     ----------
#     Parameters:
#         streams (Sequence[str]):
#             Which sources to load, using the established ``QC_STREAMS`` request
#             names. ``'events'`` is added automatically and produces the parsed
#             ``'trials'`` stream through the generic trial parser; requesting
#             ``'dlc'`` also adds ``'video'`` for alignment. Default ``('events',)``
#             (lightest: just the trial table).
#         device_yaml (str | Path):
#             HARP device YAML for the Nosepoke boards. Default ``'./device.yml'``.
#         soundcard_yaml (str | Path):
#             HARP device YAML for the SoundCard. Default ``'./soundcard.yml'``.
#         nosepoke_count (int):
#             Number of arena ports, forwarded to the trial parser. Default 18.
#         trial_start_buffer (float):
#             Seconds between one trial's end and the next trial's start, forwarded
#             to the trial parser. Default 0.0 (no buffer, unlike Q_C_Analysis_Workflow).
#         dlc_file_format (str):
#             Which DLC export the pose reader loads from each session's ``DLC``
#             folder: ``'csv'``, ``'h5'``, or ``'auto'`` (csv if any present, else
#             h5). Default ``'auto'`` so a session holding only ``.h5`` still loads.
#     Returns:
#         Catalog:
#             The assembled catalog, ready to hand to a DataStructure.
#     '''

#     device_yaml = _resolve_harp_yaml_path(device_yaml)
#     soundcard_yaml = _resolve_harp_yaml_path(soundcard_yaml)

#     # 1| Import lazily: the HARP presets pull in optional packages, and keeping the
#     #    imports inside the function keeps ``import data_conduit.refactor_qc`` light.
#     from data_conduit.core.utils import _concat_split_dataframes
#     from data_conduit.datasources.monosource import SessionSettings, VideoData
#     from data_conduit.datastructures import StreamCatalog as Catalog
#     from data_conduit.refactor_qc.trial_spec import qc_trial_spec, qc_trials_reader

#     # 2| Normalise the request: the legacy events token guarantees a trials reader;
#     #    DLC implies video because the aligner needs the camera frame times. Remember
#     #    whether video was asked for in its own right, so a video pulled in only to
#     #    align DLC can be dropped from the output rather than cluttering it.
#     requested = list(streams)
#     video_explicit = 'video' in requested
#     unknown = [s for s in requested if s not in QC_STREAMS]
#     if unknown:
#         raise ValueError(f'unknown streams {unknown}; valid: {list(QC_STREAMS)}.')
#     if 'events' not in requested:
#         requested = ['events', *requested]
#     if 'dlc' in requested and 'video' not in requested:
#         requested = [*requested, 'video']

#     catalog = Catalog()
#     trial_reader = qc_trials_reader(
#         qc_trial_spec(
#             nosepoke_count=nosepoke_count,
#             trial_start_buffer=trial_start_buffer,
#         )
#     )

#     # 3| Readers. Each is a one-argument ``path -> object`` callable. The Q_C trial
#     #    reader owns event loading plus generic TrialSpec parsing, so raw events
#     #    never enter the cross-session output. HARP / pose readers return their
#     #    normal source objects.
#     if 'nosepoke' in requested:
#         from data_conduit.integrations.harp.datasource_presets import Nosepoke
#         catalog.add_reader(
#             'nosepoke',
#             lambda p: Nosepoke(experiment_directory_path=p, harp_device_yaml_path=device_yaml),
#         )
#     if 'soundcard' in requested:
#         from data_conduit.integrations.harp.datasource_presets import SoundCard
#         catalog.add_reader(
#             'soundcard',
#             lambda p: SoundCard(experiment_directory_path=p, harp_device_yaml_path=soundcard_yaml),
#         )
#     if 'session_settings' in requested:
#         catalog.add_reader(
#             'session_settings',
#             lambda p: SessionSettings(experiment_directory_path=p).df,
#         )
#     if 'video' in requested:
#         # Return the flattened video table (one DataFrame), not the VideoData
#         # object: a multi-file session's object nests into split members, whereas
#         # the flattened table is one clean 'video' member, and the aligner reads
#         # its Time index for the frame times all the same.
#         catalog.add_reader(
#             'video',
#             lambda p: _concat_split_dataframes(VideoData(experiment_directory_path=p).df),
#         )
#     if 'dlc' in requested:
#         from data_conduit.integrations.DLC.pose import DLCPose

#         # Selected sessions without generated pose remain in the load; they simply
#         # contribute no DLC streams, as the original Q_C notebooks expect.
#         catalog.add_reader(
#             'dlc',
#             lambda p: DLCPose(experiment_directory_path=p, file_format=dlc_file_format),
#             optional=True,
#         )

#     # Register trials last, matching the previous final stream order after the raw
#     # events intermediate had been replaced by the parsed table.
#     catalog.add_reader('trials', trial_reader)

#     # 4| Configurator: put DLC pose on the session clock using the video frame times.
#     #    Only added when DLC was requested (it needs the video object too).
#     if 'dlc' in requested:
#         from data_conduit.integrations.DLC.pose import align_pose_to_video

#         def _align_dlc(objects: dict) -> dict:
#             result = dict(objects)
#             # Align only when both pose and video are present (either may have been
#             # skipped, e.g. a session with no DLC folder).
#             if 'dlc' in result and 'video' in result:
#                 result['dlc'] = align_pose_to_video(result['video'], result['dlc'])
#             # Drop the video table if it was only pulled in to align DLC (not asked
#             # for in its own right), so the output stays focused on dlc + trials.
#             # Done in every session (even DLC-less ones) so no stray video leaks in.
#             if not video_explicit:
#                 result.pop('video', None)
#             return result

#         catalog.add_configurator('align_dlc', _align_dlc)

#     return catalog

# #===============================================================================



# #===============================================================================
# # 2| qc_datastructure (Catalog + Root + Layout -> DataStructure)
# #===============================================================================
# def qc_datastructure(
#         root: str | Path,
#         *,
#         depth: int,
#         level_names,
#         streams=('events', 'nosepoke', 'soundcard', 'session_settings', 'video', 'dlc'),
#         include=None,
#         exclude=None,
#         device_yaml: str | Path = './device.yml',
#         soundcard_yaml: str | Path = './soundcard.yml',
#         nosepoke_count: int = 18,
#         trial_start_buffer: float = 0.0,
#         dlc_file_format: str = 'auto',
#         **level_selectors,
# ):
#     '''
#     Build a configured Q_C DataStructure for a given root directory and layout.

#     Convenience wrapper: assembles the Q_C catalog (via ``build_qc_catalog``) and
#     points a DataStructure at ``root`` with the directory layout and selectors you
#     supply. Call ``.load()`` on the result to get the cross-session trial table
#     (plus any other requested streams).

#     ----------
#     Parameters:
#         root (str | Path):
#             The data directory to search.
#         depth (int):
#             Levels below ``root`` where the session folders sit (e.g. 3 for
#             Q_C_Analysis_Workflow mouse/phase/day/session, 2 for firstdata
#             mouse/day/session).
#         level_names (Sequence[str]):
#             Names for the intermediate directory levels, retained as provenance
#             columns (e.g. ``('mouseID', 'phase', 'day')`` or ``('mouseID', 'day')``).
#         streams (Sequence[str]):
#             Which streams to load; see ``build_qc_catalog``. Default ``('events',)``.
#         include (Sequence[str] | None):
#             INCLUDE-mode session-name allowlist.
#         exclude (Sequence[str] | None):
#             EXCLUDE-mode session-name denylist.
#         device_yaml, soundcard_yaml (str | Path):
#             HARP YAMLs (only needed when nosepoke / soundcard are in ``streams``).
#         nosepoke_count (int):
#             Arena port count for the trial parser. Default 18.
#         trial_start_buffer (float):
#             Inter-trial buffer for the trial parser. Default 0.0.
#         dlc_file_format (str):
#             DLC export the pose reader loads: ``'csv'``, ``'h5'``, or ``'auto'``
#             (csv if present, else h5). Default ``'auto'``.
#         **level_selectors:
#             ``l{n}_selector`` filters for the intermediate levels (e.g.
#             ``l1_selector='Testing'`` to keep only the Testing phase).
#     Returns:
#         DataStructure:
#             The configured DataStructure; call ``.load()`` to run it.
#     '''
#     from data_conduit.datastructures import DataStructure

#     catalog = build_qc_catalog(
#         streams=streams,
#         device_yaml=device_yaml,
#         soundcard_yaml=soundcard_yaml,
#         nosepoke_count=nosepoke_count,
#         trial_start_buffer=trial_start_buffer,
#         dlc_file_format=dlc_file_format,
#     )
#     return DataStructure(
#         root,
#         catalog,
#         depth=depth,
#         level_names=level_names,
#         include=include,
#         exclude=exclude,
#         **level_selectors,
#     )

# #===============================================================================



# ################################################################################
