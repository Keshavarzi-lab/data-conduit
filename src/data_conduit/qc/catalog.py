'''
Configured Q_C DataStructure: readers + trial-table / DLC configurators.
=======================================================================

Description:
    Build a ready-to-use ``DataStructure`` for the Q_C nosepoke / pose task. It
    wires the lab's source readers and two configurators onto data-conduit's
    generic Catalog / DataStructure front end:

      * Readers (opt-in via ``streams``): events, nosepoke, soundcard,
        session_settings, video, dlc. ``events`` is always included because the
        trial table is built from it.
      * Configurator ``trials``: runs ``parse_events_to_trials`` on the events
        log and replaces the raw events with the distilled trial table (so the
        huge per-session event log is not concatenated across sessions).
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

    One ``load()`` then returns the cross-session trial table (with mouseID / day /
    session provenance columns) alongside any other requested streams.

Contents:
--------------------------------
- build_qc_catalog: Assemble the Catalog (readers + configurators) for the Q_C task.
- qc_datastructure: Wrap that Catalog in a DataStructure for a given root + layout.
'''





################################################################################
# Imports
################################################################################

from pathlib import Path

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



# The full set of streams the Q_C catalog knows how to read. 'events' is always
# loaded (the trial table needs it); the rest are opt-in via the ``streams`` arg.
QC_STREAMS = ('events', 'nosepoke', 'soundcard', 'session_settings', 'video', 'dlc')




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| build_qc_catalog (Readers + Trial / DLC Configurators)
#===============================================================================
def build_qc_catalog(
        *,
        streams=('events', 'nosepoke', 'soundcard', 'session_settings', 'video', 'dlc'),
        device_yaml: str | Path = './device.yml',
        soundcard_yaml: str | Path = './soundcard.yml',
        nosepoke_count: int = 18,
        trial_start_buffer: float = 0.0,
        dlc_file_format: str = 'auto',
):
    '''
    Assemble the Q_C Catalog: the requested readers plus the trial / DLC configurators.

    ----------
    Parameters:
        streams (Sequence[str]):
            Which streams to load, any of ``QC_STREAMS``. ``'events'`` is added
            automatically (the trial table needs it); requesting ``'dlc'`` also
            adds ``'video'`` (the aligner needs the camera frame times). Default
            ``('events',)`` (lightest: just the trial table).
        device_yaml (str | Path):
            HARP device YAML for the Nosepoke boards. Default ``'./device.yml'``.
        soundcard_yaml (str | Path):
            HARP device YAML for the SoundCard. Default ``'./soundcard.yml'``.
        nosepoke_count (int):
            Number of arena ports, forwarded to the trial parser. Default 18.
        trial_start_buffer (float):
            Seconds between one trial's end and the next trial's start, forwarded
            to the trial parser. Default 0.0 (no buffer, unlike Q_C_Analysis_Workflow).
        dlc_file_format (str):
            Which DLC export the pose reader loads from each session's ``DLC``
            folder: ``'csv'``, ``'h5'``, or ``'auto'`` (csv if any present, else
            h5). Default ``'auto'`` so a session holding only ``.h5`` still loads.
    Returns:
        Catalog:
            The assembled catalog, ready to hand to a DataStructure.
    '''

    device_yaml = _resolve_harp_yaml_path(device_yaml)
    soundcard_yaml = _resolve_harp_yaml_path(soundcard_yaml)

    # 1| Import lazily: the HARP presets pull in optional packages, and keeping the
    #    imports inside the function keeps ``import data_conduit.qc`` light.
    from data_conduit.datastructures import StreamCatalog as Catalog
    from data_conduit.datasources.monosource import ExperimentEvents, SessionSettings, VideoData
    from data_conduit.qc.trials import parse_events_to_trials
    from data_conduit.core.utils import _concat_split_dataframes

    # 2| Normalise the requested stream set: events is mandatory; dlc implies video
    #    (the aligner needs the camera frame times). Remember whether video was asked
    #    for in its own right, so a video pulled in only to align DLC can be dropped
    #    from the output afterwards rather than cluttering it.
    requested = list(streams)
    video_explicit = 'video' in requested
    unknown = [s for s in requested if s not in QC_STREAMS]
    if unknown:
        raise ValueError(f'unknown streams {unknown}; valid: {list(QC_STREAMS)}.')
    if 'events' not in requested:
        requested = ['events', *requested]
    if 'dlc' in requested and 'video' not in requested:
        requested = [*requested, 'video']

    catalog = Catalog()

    # 3| Readers. Each is a one-argument ``path -> object`` callable. The events
    #    reader flattens its (possibly multi-file) log into one DataFrame with the
    #    shared filename-first policy; the HARP / pose readers return source objects.
    catalog.add_reader(
        'events',
        lambda p: _concat_split_dataframes(ExperimentEvents(experiment_directory_path=p).df),
    )
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
        catalog.add_reader(
            'video',
            lambda p: _concat_split_dataframes(VideoData(experiment_directory_path=p).df),
        )
    if 'dlc' in requested:
        from data_conduit.integrations.DLC.pose import DLCPose
        catalog.add_reader(
            'dlc',
            lambda p: DLCPose(experiment_directory_path=p, file_format=dlc_file_format),
        )

    # 4| Configurator: build the trial table and DROP the raw events. The distilled
    #    table is what we want concatenated across sessions; the full event log is
    #    huge and only an intermediate, so it is not carried into the output.
    def _build_trials(objects: dict) -> dict:
        # The events reader can be skipped for a session with no log; then there is
        # no trial table to build, so leave the objects untouched.
        if 'events' not in objects:
            return objects
        # Drop the raw events IN PLACE before parsing. The full event log is huge
        # and only an intermediate, so it never belongs in the output. Popping it
        # up front also means that if parsing raises (e.g. a truncated log with no
        # "Start trial logic" marker), read_session skips this configurator with a
        # warning and the raw events cannot leak into the combined result.
        events = objects.pop('events')
        trials = parse_events_to_trials(
            events,
            nosepoke_count=nosepoke_count,
            trial_start_buffer=trial_start_buffer,
        )
        return {**objects, 'trials': trials}

    catalog.add_configurator('trials', _build_trials)

    # 5| Configurator: put DLC pose on the session clock using the video frame times.
    #    Only added when DLC was requested (it needs the video object too).
    if 'dlc' in requested:
        from data_conduit.integrations.DLC.pose import align_pose_to_video

        def _align_dlc(objects: dict) -> dict:
            result = dict(objects)
            # Align only when both pose and video are present (either may have been
            # skipped, e.g. a session with no DLC folder).
            if 'dlc' in result and 'video' in result:
                result['dlc'] = align_pose_to_video(result['video'], result['dlc'])
            # Drop the video table if it was only pulled in to align DLC (not asked
            # for in its own right), so the output stays focused on dlc + trials.
            # Done in every session (even DLC-less ones) so no stray video leaks in.
            if not video_explicit:
                result.pop('video', None)
            return result

        catalog.add_configurator('align_dlc', _align_dlc)

    return catalog

#===============================================================================



#===============================================================================
# 2| qc_datastructure (Catalog + Root + Layout -> DataStructure)
#===============================================================================
def qc_datastructure(
        root: str | Path,
        *,
        depth: int,
        level_names,
        streams=('events', 'nosepoke', 'soundcard', 'session_settings', 'video', 'dlc'),
        include=None,
        exclude=None,
        device_yaml: str | Path = './device.yml',
        soundcard_yaml: str | Path = './soundcard.yml',
        nosepoke_count: int = 18,
        trial_start_buffer: float = 0.0,
        dlc_file_format: str = 'auto',
        **level_selectors,
):
    '''
    Build a configured Q_C DataStructure for a given root directory and layout.

    Convenience wrapper: assembles the Q_C catalog (via ``build_qc_catalog``) and
    points a DataStructure at ``root`` with the directory layout and selectors you
    supply. Call ``.load()`` on the result to get the cross-session trial table
    (plus any other requested streams).

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
            Which streams to load; see ``build_qc_catalog``. Default ``('events',)``.
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
        **level_selectors:
            ``l{n}_selector`` filters for the intermediate levels (e.g.
            ``l1_selector='Testing'`` to keep only the Testing phase).
    Returns:
        DataStructure:
            The configured DataStructure; call ``.load()`` to run it.
    '''
    from data_conduit.datastructures import DataStructure

    catalog = build_qc_catalog(
        streams=streams,
        device_yaml=device_yaml,
        soundcard_yaml=soundcard_yaml,
        nosepoke_count=nosepoke_count,
        trial_start_buffer=trial_start_buffer,
        dlc_file_format=dlc_file_format,
    )
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
