'''
Grid plots of DLC centroid path traces per trial.
==================================================

Description:
    Plot the animal's centroid path (from DLC pose) for each trial, laid out on a
    grid, over the arena's video frame. One flexible function, ``plot_path_grid``,
    covers the requested figures.

    Rows and columns are each driven by one OR MORE trials-table columns (or the
    special axis ``'segment'`` = the outbound|inbound pair). Passing a LIST nests
    the axes: ``col_by=['mouseID', 'segment']`` gives one column block per mouse,
    each split into adjacent outbound and inbound subplots.

      * Outbound vs inbound per mouse:
            plot_path_grid(result, root, mice=['FbR_...'],
                           row_by='session', col_by='segment')

      * All mice as columns, each with outbound + inbound, one row per day:
            plot_path_grid(result, root,
                           row_by='day', col_by=['mouseID', 'segment'])

      * Inbound only, grouped:
            plot_path_grid(result, root, groups={'g1': {...}, 'g2': {...}},
                           segment='inbound', row_by='day', col_by='group')

    Grouping: ``groups={group: {mice}}`` adds a ``group`` column (reverse of the
    mapping), after which ``group`` is just another axis for ``row_by`` / ``col_by``.

    Colouring: by default each mouse has its own colourmap (so overlaid mice stay
    distinguishable). When ``groups=...`` is supplied, colouring switches to the
    ``group`` column unless ``color_by`` overrides it. Within each ROW GROUP (per
    active colour key), trials ramp across ``cmap_range`` (0.2 -> 1.0) in
    chronological order (session, then trial), so earlier trials are lighter. Set
    ``row_by='session'`` for a per-session ramp, ``row_by='day'`` for a per-day
    ramp across that day's sessions.

    Each cell draws that session's first video frame as a static background and
    fixes the axes to the frame's pixel size, so the whole arena is visible and
    every cell shares the same extent. A session with no DLC pose renders as the
    bare frame (no traces), which makes missing pose visible rather than hidden.

Contents:
--------------------------------
- DEFAULT_CENTROID_POINTS: the DLC keypoints averaged into the centroid by default.
- add_group_column:        add a ``group`` column from a {group: {mice}} mapping.
- diagnose_path_grid:      report whether each selected trial would produce a path.
- plot_path_grid:          the grid plotter.
'''





################################################################################
# Imports
################################################################################

from itertools import product
from pathlib import Path

import imageio.v3 as iio
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_conduit.qc.slicing import slice_pose_for_trial

################################################################################




# The DLC keypoints in this rig, hardcoded from the actual pose data. The centroid
# is their mean by default; pass a subset via ``centroid_points`` to change it.
DEFAULT_CENTROID_POINTS = ('nose', 'lear', 'rear', 'body', 'tailbase')

# Sequential colourmaps assigned to colour groups (mice or groups) in turn when
# the caller gives none. Each ramps light -> dark, which is what the "earlier
# trials lighter" rule wants.
_CMAP_ROTATION = ('Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Greys', 'YlOrBr', 'PuBuGn')

# Keep this local to the plotter so failed path slices can report the exact trial
# window that did not produce DLC points.
_SEGMENT_TIME_COLUMNS = {
    'outbound': ('outbound_start_time', 'outbound_end_time'),
    'inbound': ('inbound_start_time', 'inbound_end_time'),
    'trial': ('start_time', 'end_time'),
}




################################################################################
# Private Helpers
################################################################################



#===============================================================================
# 0| Lazy Movement Plotter Import
#===============================================================================
def _plot_centroid_trajectory(*args, **kwargs):
    '''
    Call movement's centroid trajectory plotter, importing it only when needed.

    ``movement`` is an optional dependency of data-conduit; keeping this import
    lazy lets the rest of ``data_conduit.qc`` import cleanly in non-plotting
    environments.
    '''
    try:
        from movement.plots import plot_centroid_trajectory
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plot_path_grid requires the optional 'movement' dependency. Install "
            "it with `pip install data-conduit[movement]` or install `movement` "
            "in this environment."
        ) from exc
    return plot_centroid_trajectory(*args, **kwargs)

#===============================================================================



#===============================================================================
# 1| Resolve Labels -> Colourmap Mapping
#===============================================================================
def _resolve_color_cmaps(color_cmaps, labels):
    '''
    Return a ``{label: Colormap}`` mapping, auto-filling any label not supplied.

    ----------
    Parameters:
        color_cmaps (dict | None):
            Caller-supplied ``{label: colourmap name or Colormap}``, possibly partial.
        labels (sequence):
            The colour-group labels that need a colourmap.
    Returns:
        dict:
            ``{label: matplotlib.colors.Colormap}`` for every label in ``labels``.
    '''
    resolved = {
        label: mpl.colormaps[_CMAP_ROTATION[i % len(_CMAP_ROTATION)]]
        for i, label in enumerate(labels)
    }
    if color_cmaps:
        for label, cmap in color_cmaps.items():
            resolved[label] = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    return resolved


def _resolve_color_column(
        color_by: str | None,
        groups,
        mouse_column: str,
        *,
        group_column: str = 'group',
) -> str:
    '''
    Return the trials-table column that drives colour assignment and ramping.

    Grouped plots default to colouring by ``group`` so every mouse in the same
    group shares a colour family. Callers can override this with ``color_by``.
    '''
    if color_by is not None:
        return color_by
    return group_column if groups is not None else mouse_column


def _truncate_cmap(cmap, cmap_range, n: int = 128):
    '''
    Return the ``[lo, hi]`` sub-range of a colourmap as its own colourmap.

    Used for the per-subpanel legend so the colourbar shows exactly the band the
    trial ramp uses (``cmap_range``), not the whole colourmap.

    ----------
    Parameters:
        cmap (matplotlib.colors.Colormap):
            The source colourmap.
        cmap_range (tuple):
            ``(lo, hi)`` fractions to keep.
        n (int):
            Number of samples in the truncated map.
    Returns:
        matplotlib.colors.Colormap:
            The sub-range colourmap.
    '''
    lo, hi = cmap_range
    return mcolors.LinearSegmentedColormap.from_list(f'{cmap.name}_sub', cmap(np.linspace(lo, hi, n)))

#===============================================================================



#===============================================================================
# 2| First Video Frame For a Session (cached)
#===============================================================================
def _first_frame_for_session(root, session_id, video_subdir, cache: dict):
    '''
    Return the first frame of a session's VideoData ``.avi``, or None if absent.

    Located by globbing ``<root>/**/<session_id>/<video_subdir>/*.avi`` so the exact
    layout need not be known. Frames are cached per session id.

    ----------
    Parameters:
        root (str | Path):
            The data root the DataStructure was loaded from.
        session_id (str):
            The session folder name (the ``session`` tag on the trials table).
        video_subdir (str):
            The per-session video subfolder. Usually ``'VideoData'``.
        cache (dict):
            A dict reused across calls to memoise frames by session id.
    Returns:
        np.ndarray | None:
            The first frame (H x W x 3), or None if no ``.avi`` was found.
    '''
    if session_id in cache:
        return cache[session_id]
    # Duplicate day folders ('Day 11' / 'Day_11') can yield two matches for one
    # session; they hold the same recording, so the first sorted match is fine.
    matches = sorted(Path(root).glob(f'**/{session_id}/{video_subdir}/*.avi'))
    frame = iio.imread(str(matches[0]), index=0) if matches else None
    cache[session_id] = frame
    return frame

#===============================================================================



#===============================================================================
# 3| Resolve One Grid Axis Into Ordered (key, value) Combinations
#===============================================================================
def _axis_combos(axis, trials: pd.DataFrame, segments):
    '''
    Turn a row/column axis spec into its ordered list of cell combinations.

    An axis is a single spec or a list of specs; each spec is either a trials-table
    column name or the literal ``'segment'``. The returned combinations are the
    Cartesian product of each spec's values (nested axes), each a tuple of
    ``(key, value)`` pairs.

    ----------
    Parameters:
        axis (str | list[str]):
            One spec, or a list of specs to nest (outer first).
        trials (pd.DataFrame):
            The working trials table (values are read from its columns).
        segments (sequence):
            The segment names used when a spec is ``'segment'``.
    Returns:
        tuple[list[str], list[tuple]]:
            The normalised list of specs, and the ordered list of combinations
            (each a tuple of ``(key, value)`` pairs).
    '''
    specs = [axis] if isinstance(axis, str) else list(axis)
    per_spec = []
    for spec in specs:
        if spec == 'segment':
            per_spec.append([('segment', s) for s in segments])
        else:
            # pd.unique preserves first-appearance (walk = chronological) order.
            per_spec.append([(spec, value) for value in pd.unique(trials[spec])])
    combos = [tuple(combo) for combo in product(*per_spec)]
    return specs, combos

#===============================================================================



#===============================================================================
# 4| Trial / Pose Validation Helpers
#===============================================================================
def _trial_time_window(row: pd.Series, segment: str):
    '''Return the start and end time columns for one plotted path segment.'''
    if segment not in _SEGMENT_TIME_COLUMNS:
        raise ValueError(f"segment must be one of {sorted(_SEGMENT_TIME_COLUMNS)}, got {segment!r}.")
    start_col, end_col = _SEGMENT_TIME_COLUMNS[segment]
    return row[start_col], row[end_col]


def _pose_session_values(pose, session_coord: str) -> set:
    '''Return the session ids present in the combined pose array.'''
    if session_coord not in pose.coords:
        raise KeyError(f'pose array has no {session_coord!r} coordinate.')
    return set(pose.coords[session_coord].values)


def _pose_time_summary(pose, *, session, session_coord: str, time_coord: str) -> str:
    '''Return a compact per-session pose time summary for error messages.'''
    sessions = pose.coords[session_coord].values
    times = pose.coords[time_coord].values
    mask = sessions == session
    if not np.any(mask):
        return f'pose has no entries tagged {session_coord}={session!r}'
    session_times = times[mask]
    return (
        f'pose {time_coord} range for {session_coord}={session!r}: '
        f'{float(np.nanmin(session_times))} to {float(np.nanmax(session_times))} '
        f'({int(mask.sum())} samples)'
    )


def _has_finite_centroid_points(sliced, centroid_points) -> bool:
    '''Check whether a sliced trial has any finite x/y centroid source values.'''
    requested = list(centroid_points)
    available_keypoints = set(sliced.coords['keypoints'].values)
    missing = [point for point in requested if point not in available_keypoints]
    if missing:
        raise KeyError(f'centroid_points missing from pose keypoints: {missing}.')

    subset = sliced.sel(keypoints=requested)
    if 'space' in subset.coords:
        spaces = [space for space in ('x', 'y') if space in set(subset.coords['space'].values)]
        if spaces:
            subset = subset.sel(space=spaces)
    return bool(np.isfinite(subset.to_numpy()).any())

#===============================================================================



#===============================================================================
# 5| Overlay One Cell's Trial Paths Onto an Axis
#===============================================================================
def _plot_paths_on_ax(
        ax,
        pose,
        cell_trials: pd.DataFrame,
        *,
        segment: str,
        centroid_points,
    color_cmaps: dict,
        cmap_range,
    color_column: str,
        session_column: str,
        session_coord: str,
        ramp_column: str,
        time_coord: str,
        marker_size: float,
        require_pose_for_trials: bool,
):
    '''
    Draw every trial in ``cell_trials`` as a centroid path on ``ax``.

    Each trial is sliced to its ``segment`` window and drawn with ``movement``'s
    ``plot_centroid_trajectory`` (the centroid of ``centroid_points``) in a single
    colour: its active colour-group colourmap sampled at the trial's precomputed
    ramp position (``ramp_column``).

    ----------
    Parameters:
        ax (matplotlib.axes.Axes):
            The cell axis (a background frame is drawn by the caller).
        pose (xr.DataArray):
            The combined ``dlc:position`` stream.
        cell_trials (pd.DataFrame):
            The trials whose paths belong in this cell.
        segment (str):
            ``'outbound'``, ``'inbound'``, or ``'trial'``.
        centroid_points (sequence):
            Keypoints averaged into the centroid.
        color_cmaps (dict):
            ``{label: Colormap}`` for the active colour basis.
        cmap_range (tuple):
            ``(lo, hi)`` fractions of the colourmap the trial ramp spans.
        color_column (str):
            Trial-table column naming the active colour basis.
        session_column (str):
            Trial-table column naming the session.
        session_coord (str):
            Pose coordinate naming the session.
        ramp_column (str):
            Column holding each trial's 0..1 ramp position within its row group.
        time_coord (str):
            Pose time dimension (``'Time'``).
        marker_size (float):
            Scatter marker size passed to ``movement``.
        require_pose_for_trials (bool):
            If True, raise when a selected trial from a session present in the
            pose array does not yield drawable DLC points for its path window.
    Returns:
        pd.DataFrame:
            The subset of ``cell_trials`` that was actually sent to movement for
            plotting.
    '''
    lo, hi = cmap_range
    keypoints = list(centroid_points)
    pose_sessions = _pose_session_values(pose, session_coord)
    plotted_indices = []

    for trial_idx, row in cell_trials.iterrows():
        session = row[session_column]
        session_has_pose = session in pose_sessions
        start, end = _trial_time_window(row, segment)

        # A selected session with no DLC in the combined pose array is still left
        # blank. A selected trial in a session that DOES have DLC must yield points.
        if not session_has_pose:
            continue

        if pd.isna(start) or pd.isna(end) or end < start:
            if require_pose_for_trials:
                raise ValueError(
                    f'cannot plot {segment!r} path for session={session!r}, '
                    f'trial_index={row.get("trial_index", trial_idx)!r}: invalid '
                    f'time window start={start!r}, end={end!r}.'
                )
            continue

        sliced = slice_pose_for_trial(
            pose,
            row,
            segment=segment,
            session_column=session_column,
            session_coord=session_coord,
            time_coord=time_coord,
        )
        if sliced.sizes.get(time_coord, 0) == 0:
            if require_pose_for_trials:
                raise ValueError(
                    f'cannot plot {segment!r} path for session={session!r}, '
                    f'trial_index={row.get("trial_index", trial_idx)!r}: DLC slice is '
                    f'empty for window {start!r} to {end!r}. '
                    f'{_pose_time_summary(pose, session=session, session_coord=session_coord, time_coord=time_coord)}.'
                )
            continue

        if not _has_finite_centroid_points(sliced, keypoints):
            if require_pose_for_trials:
                raise ValueError(
                    f'cannot plot {segment!r} path for session={session!r}, '
                    f'trial_index={row.get("trial_index", trial_idx)!r}: DLC slice has '
                    f'{sliced.sizes.get(time_coord, 0)} samples but no finite x/y '
                    f'values for centroid_points={tuple(keypoints)!r}.'
                )
            continue

        # Colour: the trial's active colour-group colourmap at its ramp position
        # (earlier -> lo).
        frac = lo + (hi - lo) * row[ramp_column]
        colour = mcolors.to_hex(color_cmaps[row[color_column]](frac))

        # movement's scatter: recast to its schema; passing ``c`` overrides its
        # default time-gradient with our solid per-trial colour.
        movement_pose = sliced.rename({time_coord: 'time', 'keypoints': 'keypoint'}).expand_dims({'individual': ['ind']})
        _plot_centroid_trajectory(movement_pose, keypoints=keypoints, ax=ax, c=colour, s=marker_size)
        plotted_indices.append(trial_idx)

    # movement writes a title / axis labels on every call; clear them so the grid
    # can set its own row and column labels.
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    return cell_trials.loc[plotted_indices]

#===============================================================================



################################################################################




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| add_group_column (mouse -> group)
#===============================================================================
def add_group_column(
        trials: pd.DataFrame,
        groups: dict,
        *,
        mouse_column: str = 'mouseID',
        group_column: str = 'group',
) -> pd.DataFrame:
    '''
    Return a copy of ``trials`` with a ``group`` column from a {group: {mice}} map.

    Reverses the mapping to ``mouse -> group`` and labels each trial; trials whose
    mouse is in no group get NaN. Once present, ``group`` behaves like any other
    axis in ``plot_path_grid``.

    ----------
    Parameters:
        trials (pd.DataFrame):
            A trials table carrying ``mouse_column``.
        groups (dict):
            ``{group_name: iterable_of_mice}``.
        mouse_column (str):
            Column holding the mouse id. Default ``'mouseID'``.
        group_column (str):
            Name of the column to add. Default ``'group'``.
    Returns:
        pd.DataFrame:
            A copy with ``group_column`` added.
    '''
    mouse_to_group = {mouse: group for group, mice in groups.items() for mouse in mice}
    out = trials.copy()
    out[group_column] = out[mouse_column].map(mouse_to_group)
    return out

#===============================================================================



#===============================================================================
# 2| diagnose_path_grid
#===============================================================================
def diagnose_path_grid(
        result: dict,
        *,
        mice=None,
        groups=None,
        row_by='session',
        col_by='segment',
        segment: str = 'inbound',
        segments=('outbound', 'inbound'),
        centroid_points=DEFAULT_CENTROID_POINTS,
        trials_key: str = 'trials',
        pose_key: str = 'dlc:position',
        mouse_column: str = 'mouseID',
        session_column: str = 'session',
        session_coord: str = 'session',
        time_coord: str = 'Time',
) -> pd.DataFrame:
    '''
    Return one diagnostic row per trial selected by the path-grid layout.

    This follows the same row/column/mouse/group filtering as ``plot_path_grid``,
    then checks the exact trial segment window against the DLC pose array. The
    ``status`` column is ``'would_plot'`` only when the trial has a session tag
    in pose, a valid time window, a non-empty pose slice, and finite x/y values
    for the requested centroid keypoints.
    '''

    all_trials = result[trials_key]
    pose = result[pose_key]
    trials = all_trials.copy()

    if groups is not None:
        trials = add_group_column(trials, groups, mouse_column=mouse_column)
        trials = trials[trials['group'].notna()]
    if mice is not None:
        trials = trials[trials[mouse_column].isin(list(mice))]
    if trials.empty:
        raise ValueError('no trials left after mice / groups filtering.')

    _, row_combos = _axis_combos(row_by, trials, segments)
    _, col_combos = _axis_combos(col_by, trials, segments)
    pose_sessions = _pose_session_values(pose, session_coord)

    rows = []
    for row_combo in row_combos:
        for col_combo in col_combos:
            seg = segment
            mask = pd.Series(True, index=trials.index)
            for key, value in row_combo + col_combo:
                if key == 'segment':
                    seg = value
                else:
                    mask &= trials[key] == value
            cell_trials = trials[mask]

            for trial_idx, trial in cell_trials.iterrows():
                session = trial[session_column]
                start, end = _trial_time_window(trial, seg)
                session_has_pose = session in pose_sessions
                valid_window = pd.notna(start) and pd.notna(end) and end >= start
                pose_samples = 0
                finite_centroid_points = False

                if not session_has_pose:
                    status = 'no_pose_for_session'
                elif not valid_window:
                    status = 'invalid_window'
                else:
                    sliced = slice_pose_for_trial(
                        pose,
                        trial,
                        segment=seg,
                        session_column=session_column,
                        session_coord=session_coord,
                        time_coord=time_coord,
                    )
                    pose_samples = int(sliced.sizes.get(time_coord, 0))
                    if pose_samples == 0:
                        status = 'empty_slice'
                    else:
                        finite_centroid_points = _has_finite_centroid_points(sliced, centroid_points)
                        status = 'would_plot' if finite_centroid_points else 'no_finite_centroid_points'

                rows.append({
                    'row_label': '\n'.join(str(v) for _, v in row_combo),
                    'col_label': '\n'.join(str(v) for _, v in col_combo),
                    'segment': seg,
                    'mouseID': trial[mouse_column],
                    'session': session,
                    'trial_index': trial.get('trial_index', trial_idx),
                    'start': start,
                    'end': end,
                    'session_has_pose': session_has_pose,
                    'valid_window': valid_window,
                    'pose_samples': pose_samples,
                    'finite_centroid_points': finite_centroid_points,
                    'status': status,
                })

    return pd.DataFrame(rows)

#===============================================================================



#===============================================================================
# 3| plot_path_grid
#===============================================================================
def plot_path_grid(
        result: dict,
        root,
        *,
        mice=None,
        groups=None,
        row_by='session',
        col_by='segment',
    color_by: str | None = None,
        segment: str = 'inbound',
        segments=('outbound', 'inbound'),
        centroid_points= ('nose', 'lear', 'rear', 'body', 'tailbase'),
        mouse_cmaps=None,
        cmap_range=(0.2, 1.0),
        trials_key: str = 'trials',
        pose_key: str = 'dlc:position',
        mouse_column: str = 'mouseID',
        session_column: str = 'session',
        session_coord: str = 'session',
        time_coord: str = 'Time',
        video_subdir: str = 'VideoData',
        colorbar: bool = True,
        marker_size: float = 1.0,
        fontsize: int = 13,
        figsize_per_cell=(6.4, 5.2),
        require_pose_for_trials: bool = True,
):
    '''
    Plot per-trial DLC centroid paths on a grid over the arena video frame.

    ``row_by`` and ``col_by`` each take a single axis or a LIST of axes to nest.
    An axis is a trials-table column (``'session'``, ``'day'``, ``'group'``,
    ``'mouseID'``) or ``'segment'`` (the ``segments`` pair). A cell overlays every
    trial matching its row and column values, drawn for the cell's segment, each
    trial coloured by the active colour basis at its within-row-group ramp
    position.

    ----------
    Parameters:
        result (dict):
            A loaded Q_C DataStructure output containing ``trials_key`` and
            ``pose_key``.
        root (str | Path):
            The data root (used to locate each session's background ``.avi``).
        mice (sequence | None):
            Restrict to these mice. Default: all mice present.
        groups (dict | None):
            ``{group: {mice}}``. If given, a ``group`` column is added and trials
            outside any group are dropped.
        row_by (str | list[str]):
            Row axis or nested axes. Default ``'session'``. The colour ramp spans
            each row group (per active colour key).
        col_by (str | list[str]):
            Column axis or nested axes. Default ``'segment'`` (outbound|inbound).
            Use e.g. ``['mouseID', 'segment']`` for mouse columns each split into
            outbound|inbound.
        color_by (str | None):
            Trials-table column controlling colourmap assignment and trial ramping.
            Defaults to ``'group'`` when ``groups`` is supplied, otherwise
            ``mouse_column``.
        segment (str):
            The segment drawn when no axis is ``'segment'``. Default ``'inbound'``.
        segments (sequence):
            The segments used when an axis is ``'segment'``. Default
            ``('outbound', 'inbound')``.
        centroid_points (sequence):
            Keypoints averaged into the centroid. Default ``('nose', 'lear', 'rear', 'body', 'tailbase')``.
        mouse_cmaps (dict | None):
            Explicit colourmaps keyed by the active colour basis; omitted labels
            are auto-assigned.
        cmap_range (tuple):
            ``(lo, hi)`` colourmap fractions the trial ramp spans. Default
            ``(0.2, 1.0)``.
        trials_key, pose_key (str):
            Keys of the trials table and pose array in ``result``.
        mouse_column, session_column (str):
            Trials-table columns naming the mouse and session.
        session_coord (str):
            Pose coordinate naming the session. Default ``'session'``.
        time_coord (str):
            Pose time dimension. Default ``'Time'``.
        video_subdir (str):
            Per-session video subfolder. Default ``'VideoData'``.
        colorbar (bool):
            If True (default), add a slim per-colour-group colourbar to each
            subpanel showing the trial ramp, plus a label / trial-count
            annotation.
        marker_size (float):
            Scatter marker size passed to ``movement``. Default 1.
        fontsize (int):
            Base font size for titles, row labels, and the per-panel annotation
            (colourbar ticks use ``fontsize - 2``). Default 13.
        figsize_per_cell (tuple):
            ``(width, height)`` inches per cell.
        require_pose_for_trials (bool):
            If True (default), raise when a selected trial from a session present
            in the DLC pose array does not produce drawable pose points for its
            path window. Sessions absent from the pose array still render as a
            bare frame, preserving the "missing DLC is visible" behaviour.
    Returns:
        tuple[matplotlib.figure.Figure, numpy.ndarray]:
            The figure and its 2D array of axes.
    '''

    # 1| Pull the trials table and pose; work on a copy of the trials.
    all_trials = result[trials_key]
    pose = result[pose_key]
    trials = all_trials.copy()

    # 1a| Normalise and validate centroid_points. A bare string is ONE keypoint,
    #     not a per-character list (``list('nose')`` -> ['n','o','s','e']), and an
    #     unknown keypoint should fail loudly here rather than as a cryptic KeyError
    #     deep inside movement's ``.sel(keypoint=...)``.
    if isinstance(centroid_points, str):
        centroid_points = (centroid_points,)
    centroid_points = tuple(centroid_points)
    available_keypoints = set(pose['keypoints'].values.tolist())
    unknown = [kp for kp in centroid_points if kp not in available_keypoints]
    if unknown:
        raise ValueError(
            f'unknown centroid_points {unknown}; '
            f'available keypoints: {sorted(available_keypoints)}.'
        )

    # 2| Optional grouping and mouse restriction.
    if groups is not None:
        trials = add_group_column(trials, groups, mouse_column=mouse_column)
        trials = trials[trials['group'].notna()]
    if mice is not None:
        trials = trials[trials[mouse_column].isin(list(mice))]
    if trials.empty:
        raise ValueError('no trials left after mice / groups filtering.')
    color_column = _resolve_color_column(color_by, groups, mouse_column)
    if color_column not in trials.columns:
        raise KeyError(f'color_by={color_column!r} is not present in the selected trials table.')
    panel_fontsize = fontsize
    header_fontsize = fontsize + 1
    colorbar_fontsize = max(fontsize - 2, 1)

    # 3| Per-trial colour ramp, computed WITHIN each row group (its non-segment
    #    axes) per active colour key, in chronological order (the table is already
    #    ordered by session then trial). So earlier trials in a row are lighter,
    #    and a per-day row ramps continuously across that day's sessions.
    row_specs = [row_by] if isinstance(row_by, str) else list(row_by)
    ramp_columns = list(dict.fromkeys([s for s in row_specs if s != 'segment'] + [color_column]))
    grouped = trials.groupby(ramp_columns, sort=False)
    rank = grouped.cumcount()
    size = grouped[mouse_column].transform('size')
    trials = trials.copy()
    trials['_ramp'] = (rank / (size - 1).clip(lower=1)).astype(float)

    # 4| Per-colour-group colourmaps for every label still present.
    resolved_cmaps = _resolve_color_cmaps(mouse_cmaps, list(pd.unique(trials[color_column])))

    # 5| Resolve the row and column cell combinations (nested axes -> product).
    _, row_combos = _axis_combos(row_by, trials, segments)
    _, col_combos = _axis_combos(col_by, trials, segments)

    # 6| A reference frame fixes the shared axis extent (whole arena visible in
    #    every cell). Pose is in pixels, so the extent is the frame's W x H.
    frame_cache: dict = {}
    reference_frame = None
    for session_id in pd.unique(trials[session_column]):
        reference_frame = _first_frame_for_session(root, session_id, video_subdir, frame_cache)
        if reference_frame is not None:
            break
    if reference_frame is None:
        raise FileNotFoundError(
            f'no {video_subdir} .avi found under {root} for any selected session.'
        )
    height, width = reference_frame.shape[:2]

    # 7| Build the grid.
    nrows, ncols = len(row_combos), len(col_combos)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )

    for ri, row_combo in enumerate(row_combos):
        for ci, col_combo in enumerate(col_combos):
            ax = axes[ri][ci]

            # 7a| The cell's segment comes from whichever axis is 'segment'; the
            #     other (key, value) pairs filter the trials down to this cell.
            seg = segment
            mask = pd.Series(True, index=trials.index)
            for key, value in row_combo + col_combo:
                if key == 'segment':
                    seg = value
                else:
                    mask &= trials[key] == value
            cell_trials = trials[mask]

            # 7b| Background: the first session appearing in this cell (all sessions
            #     share the fixed camera). A cell with no trials still gets no frame.
            if not cell_trials.empty:
                frame = _first_frame_for_session(
                    root, cell_trials[session_column].iloc[0], video_subdir, frame_cache,
                )
                if frame is not None:
                    ax.imshow(frame, origin='upper')

            # 7c| Overlay the trial paths.
            plotted_trials = _plot_paths_on_ax(
                ax, pose, cell_trials,
                segment=seg,
                centroid_points=centroid_points,
                color_cmaps=resolved_cmaps,
                cmap_range=cmap_range,
                color_column=color_column,
                session_column=session_column,
                session_coord=session_coord,
                ramp_column='_ramp',
                time_coord=time_coord,
                marker_size=marker_size,
                require_pose_for_trials=require_pose_for_trials,
            )

            # 7c-2| Annotate each panel: per-colour-group label + plotted-trial
            #       count
            #       (top-left). This is deliberately based on traces actually sent
            #       to movement, not merely trial-table rows selected for the panel.
            counts = plotted_trials.groupby(color_column, sort=False).size()
            if len(counts):
                ax.text(
                    0.02, 0.98,
                    '\n'.join(f'{label}  n={int(counts[label])}' for label in counts.index),
                    transform=ax.transAxes, va='top', ha='left', fontsize=fontsize, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.45, edgecolor='none'),
                )
            # Per-colour-group trial colourbar(s), placed to the RIGHT of the
            # panel (one per label present; ticks mark this cell's first / last
            # trial number).
            if colorbar:
                for label in counts.index:
                    sub = plotted_trials[plotted_trials[color_column] == label]
                    tlo, thi = int(sub['trial_index'].min()), int(sub['trial_index'].max())
                    sm = plt.cm.ScalarMappable(
                        norm=mcolors.Normalize(tlo, thi if thi > tlo else tlo + 1),
                        cmap=_truncate_cmap(resolved_cmaps[label], cmap_range),
                    )
                    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.025)
                    cbar.set_ticks([tlo, thi] if thi > tlo else [tlo])
                    cbar.ax.tick_params(labelsize=colorbar_fontsize)
                    cbar.ax.yaxis.set_ticks_position('right')
                    cbar.ax.yaxis.set_label_position('right')
                    cbar.set_label('trial', fontsize=colorbar_fontsize)

            # 7d| Fixed pixel axes (y inverted to match image origin), no ticks.
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_xticks([])
            ax.set_yticks([])

            # 7e| Column headers on the top row, row labels on the first column
            #     (nested axes stack their values on separate lines).
            if ri == 0:
                ax.set_title('\n'.join(str(v) for _, v in col_combo), fontsize=header_fontsize)
            if ci == 0:
                ax.set_ylabel('\n'.join(str(v) for _, v in row_combo), fontsize=panel_fontsize)

    fig.tight_layout()
    return fig, axes

#===============================================================================



################################################################################
