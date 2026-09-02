'''
Select sessions and load their data into movement-ready shapes.
===============================================================

Description:
    This module is the bridge from "a mouse's folder on disk" to "objects the
    notebooks can hand to movement". It does four things, all pure data-conduit
    plumbing (no movement calls):

      1. list_sessions          find the day/session folders under one mouse and
                                keep the ones that actually contain DLC pose.
      2. build_session_group    run the data-conduit loader over those sessions,
                                producing one Session (an object holding every data
                                object: events, nosepoke, dlc, ...) per session.
      3. combine_all            stack every data object across sessions into one
                                container, each tagged with a 'label' (session)
                                coordinate / column. This is the "all sessions, all
                                data objects, plus a session column" object.
      4. session_movement_dataset
                                read ONE session's DLC pose and reshape it into a
                                movement-style xr.Dataset, ready for movement's
                                kinematics. movement needs a single monotonic clock,
                                so kinematics are always done per session, never on
                                the cross-session stack.

    The expected on-disk layout is one mouse folder containing day folders, each
    containing session folders:
        <mouse_dir>/<day>/<session>/{ExperimentEvents, VideoData, DLC, ...}
    so ``list_sessions(mouse_dir, depth=1, level_names=('day',))`` finds the sessions.

Contents:
--------------------------------
- list_sessions:            Select pose-bearing sessions under a mouse folder.
- build_session_group:      Load all selected sessions into an ordered SessionGroup.
- combine_all:              Stack every member across sessions into one Session.
- session_movement_dataset: Build one session's movement-style pose Dataset.
'''

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from data_conduit.datasources.monosource import VideoData
from data_conduit.datasources.pose import align_pose_to_video, pose_to_movement, read_dlc_pose
from data_conduit.sessiongroups import (
    build_sessions,
    combine_sessions,
    select_sessions,
)
from data_conduit.utils.utils_core import _matches_selector


def list_sessions(
        mouse_dir: str | Path,
        *,
        depth: int = 1,
        level_names=('day',),
        name_pattern: str | None = None,
        regex: bool = False,
        require_dlc: bool = True,
) -> dict[str, dict]:
    '''
    Select session folders under a mouse directory, keeping pose-bearing ones.

    Wraps ``data_conduit.sessiongroups.select_sessions`` (which walks the tree and
    labels each session with its folder name under ``'label'`` and its intermediate
    levels under ``level_names``) and then applies two convenience filters: an
    optional name match and a "must contain a DLC/ folder" gate.

    The DLC gate matters before a whole-session combine: ``combine_all`` keeps only
    members present in EVERY session, so one DLC-less session would silently drop
    pose from the combined result. Filtering to DLC-bearing sessions here avoids that.

    ----------
    Parameters:
        mouse_dir (str | Path):
            The mouse folder, e.g. ``.../BonsaiFiles/FbR_M01569522`` or
            ``<DATA_ROOT>/<PHASE>/<MOUSE_ID>``. Day/session folders sit below it.
        depth (int):
            Levels below ``mouse_dir`` to the session folders. Default 1 (one
            intermediate ``day`` level), matching the layout above.
        level_names (sequence of str):
            Names for the intermediate levels, attached as session metadata.
            Default ``('day',)``.
        name_pattern (str | None):
            If given, keep only sessions whose ``"{day}/{label}"`` string matches.
            ``None`` (default) keeps all (subject to the DLC gate).
        regex (bool):
            If True, ``name_pattern`` is a regular expression (``re.search``); if
            False (default), it is a plain case-insensitive substring test.
        require_dlc (bool):
            If True (default), drop any session whose folder lacks a ``DLC/``
            subdirectory.
    Returns:
        dict[str, dict]:
            ``{session_key: {'path': Path, 'day': ..., 'label': ...}}`` for the kept
            sessions, in selection order.
    '''
    # 1| Walk the tree to the session folders, labelling day + folder name.
    sessions = select_sessions(Path(mouse_dir), depth=depth, level_names=level_names)

    # 2| Pre-compile the optional name matcher: a regex as-is, or a plain substring
    #    escaped into a regex, both case-insensitive.
    matcher = None
    if name_pattern is not None:
        matcher = re.compile(name_pattern if regex else re.escape(name_pattern), re.IGNORECASE)

    # 3| Keep the sessions that pass both gates, preserving order.
    kept: dict[str, dict] = {}
    for key, entry in sessions.items():
        path = Path(entry['path'])

        # 3a| DLC gate: must have a DLC/ folder when require_dlc is on.
        if require_dlc and not (path / 'DLC').is_dir():
            continue

        # 3b| Name gate: match against the flattened "{day}/{label}" string.
        if matcher is not None:
            day = entry.get('day', '')
            label = entry.get('label', key)
            if matcher.search(f'{day}/{label}') is None:
                continue

        kept[key] = entry

    return kept


def select_pose_sessions(
        data_root: str | Path,
        *,
        mice=None,
        days=None,
        sessions=None,
        require_dlc: bool = True,
) -> dict[str, dict]:
    '''
    Select sessions from a whole data tree using include / exclude / all at each level.

    This is the multi-mouse entry point: point it at the top BonsaiFiles directory
    (the folder that holds one subfolder per mouse, with day folders below and session
    folders below those) and choose which mice, days, and sessions to keep. Each of the
    three levels takes the same three forms:

      * ``None``            -> keep ALL folders at that level.
      * a list of names     -> INCLUDE only those (matched by exact folder name).
      * ``exclude(...)``    -> keep everything EXCEPT those named (the complement of a list;
                               import ``exclude`` from ``data_conduit.utils.utils_core``).

    Wraps ``data_conduit.sessiongroups.select_sessions`` (which walks the tree and labels
    each session with its ``mouse`` and ``day``), routing the mouse choice to the first
    intermediate level and the day choice to the second, then filtering sessions and
    applying the DLC gate here.

    ----------
    Parameters:
        data_root (str | Path):
            The directory holding one subfolder per mouse, e.g.
            ``.../datasets/firstdata/BonsaiFiles``. Layout assumed below it is
            ``<mouse>/<day>/<session>/``.
        mice (list[str] | Callable | None):
            Mouse-level filter. ``None`` keeps all mice; a list includes only those mouse
            folders; an ``exclude(...)`` callable keeps all but those.
        days (list[str] | Callable | None):
            Day-level filter, same three forms. Note the day folders on disk are not
            uniformly named (``'Day 1'`` vs ``'Day 01'`` vs ``'Day_11'``), so include lists
            must use the exact folder names.
        sessions (list[str] | Callable | None):
            Session-level filter, same three forms, matched against the session folder name
            (the timestamp). ``None`` keeps every session.
        require_dlc (bool):
            If True (default), drop any selected session whose folder lacks a ``DLC/``
            subdirectory. This matters before a combine, which keeps only members present in
            every session, so a DLC-less session would otherwise silently drop pose.
    Returns:
        dict[str, dict]:
            ``{session_key: {'path': Path, 'mouse': ..., 'day': ..., 'label': ...}}`` for the
            kept sessions, in selection order.
    '''
    # 1| Walk the tree to the session folders (two intermediate levels: mouse then day),
    #    applying the mouse and day include/exclude/all filters as we descend.
    selection = select_sessions(
        Path(data_root),
        depth=2,
        level_names=('mouse', 'day'),
        l0_selector=mice,
        l1_selector=days,
    )

    # 2| Apply the session-level filter and the DLC gate, preserving selection order.
    kept: dict[str, dict] = {}
    for key, entry in selection.items():
        # 2a| Session include/exclude/all: None keeps all, a list includes those keys, an
        #     exclude(...) callable keeps all but those. _matches_selector handles all three.
        if not _matches_selector(key, sessions):
            continue

        # 2b| DLC gate: must have a DLC/ folder when require_dlc is on.
        if require_dlc and not (Path(entry['path']) / 'DLC').is_dir():
            continue

        kept[key] = entry

    return kept


def build_session_group(
        selection: dict[str, dict],
        catalog,
        *,
        label: str | None = None,
        sort_by=('day', 'label'),
        normalise: bool = False,
        verbose: bool = False,
):
    '''
    Load every selected session into an ordered SessionGroup.

    Thin wrapper over ``data_conduit.sessiongroups.build_sessions``. Each returned
    Session holds all the catalog's data objects as members (``session.data`` is the
    ``{member_name: object}`` dict; ``session.metadata`` carries ``day`` and
    ``label``). The group is ordered by ``sort_by`` so a later combine stacks
    sessions chronologically.

    ----------
    Parameters:
        selection (dict[str, dict]):
            Output of ``list_sessions``.
        catalog (DataStructureCatalog):
            What to extract per session (from ``build_catalog``).
        label (str | None):
            A label for the group as a whole, e.g. the mouse ID. Optional.
        sort_by (sequence of str):
            Session metadata keys to order by. Default ``('day', 'label')`` so
            sessions sort by day then by session start time.
        normalise (bool):
            If True, shift each session to start at t=0. Default False (every stream
            keeps its shared original session clock, the intended mode).
        verbose (bool):
            If True, print per-structure load/skip lines from the loader.
    Returns:
        SessionGroup:
            The loaded, ordered sessions.
    '''
    # Defer entirely to data-conduit's loader; we only fix sensible defaults.
    return build_sessions(
        selection,
        catalog,
        label=label,
        sort_by=sort_by,
        normalise=normalise,
        verbose=verbose,
    )


def combine_all(group):
    '''
    Stack every data object across sessions into one Session, tagged by session.

    Wraps ``combine_sessions(group, type_name=None)``. The returned Session's
    ``.data`` is the cross-session container: one entry per member (``events``,
    ``dlc:position``, ``nosepoke:Activations``, ...), each concatenated across all
    sessions and carrying a ``'label'`` coordinate (xarray) or column (DataFrame)
    naming each row/sample's source session.

    Only members present in EVERY session survive (the default intersection combine),
    which is why ``list_sessions`` filters to DLC-bearing sessions first.

    ----------
    Parameters:
        group (SessionGroup):
            The sessions to combine (from ``build_session_group``).
    Returns:
        Session:
            The combined Session; read its ``.data`` for the container dict.
    '''
    # type_name=None means "combine every member" and return a Session.
    return combine_sessions(group, type_name=None)


def session_movement_dataset(
        session_dir: str | Path,
        *,
        fps: float | None = None,
        on_rollback: str = 'warn',
):
    '''
    Read one session's DLC pose and reshape it into a movement-style Dataset.

    This is the formatting bridge into movement. It reads the session's DLC output
    FRAME-INDEXED with data-conduit's ``read_dlc_pose``, reads the session's
    ``VideoData`` for the per-frame camera times, aligns the two with
    ``align_pose_to_video`` (which checks the frame counts match and stamps each
    frame with its VideoData timestamp, putting pose on the session clock), then
    repackages the position + confidence pair into a movement-style ``xr.Dataset``
    with dims ``(time, individuals, keypoints, space)`` via ``pose_to_movement``.

    No confidence filtering or kinematics happen here: those are movement operations
    and belong in the notebook. This function only produces the dataset to feed in.

    ----------
    Parameters:
        session_dir (str | Path):
            One session folder containing ``DLC/`` and ``VideoData/``.
        fps (float | None):
            Frame rate stamped onto the movement dataset. If None (default), it is
            estimated from the median spacing of the pose ``Time`` axis.
        on_rollback (str):
            Forwarded to ``align_pose_to_video``: ``'warn'`` (default), ``'error'``,
            or ``'ignore'`` when the aligned ``Time`` axis steps backwards (the
            camera clock reset between VideoData files). A frame-count mismatch
            between pose and VideoData always raises (the two cannot be aligned).
    Returns:
        xr.Dataset:
            movement-style poses dataset (data variables ``position`` and
            ``confidence``; dims ``time, individuals, keypoints, space``).
    '''
    # 1| Read FRAME-INDEXED pose and the session's VideoData (per-frame camera
    #    times), then align: align_pose_to_video checks pose and video carry the
    #    same number of frames and stamps each frame with its VideoData timestamp,
    #    returning {'position', 'confidence'} on the session clock ('Time' axis).
    pose = read_dlc_pose(Path(session_dir))
    video = VideoData(experiment_directory_path=Path(session_dir))
    aligned = align_pose_to_video(video, pose, on_rollback=on_rollback)
    position, confidence = aligned['position'], aligned['confidence']

    # 2| Estimate the frame rate from the median inter-frame interval when not given,
    #    so movement's time axis carries a sensible fps for derivative units.
    if fps is None:
        times = position['Time'].values
        fps = 1.0 / float(np.median(np.diff(times)))

    # 3| Repackage into the movement schema (renames 'Time' -> 'time', adds an
    #    individual dim).
    dataset = pose_to_movement(position, confidence, fps=fps)

    # 4| Match the installed movement's canonical dimension names. movement 0.17 uses the
    #    SINGULAR 'keypoint' / 'individual' (its orientation functions validate those
    #    names), whereas pose_to_movement still emits the older plural forms. Rename so
    #    every movement function (path metrics AND head-direction) accepts the dataset.
    renames = {old: new for old, new in (('keypoints', 'keypoint'), ('individuals', 'individual'))
               if old in dataset.dims}
    return dataset.rename(renames) if renames else dataset
