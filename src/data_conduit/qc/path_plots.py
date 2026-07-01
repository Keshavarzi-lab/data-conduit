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

    Colouring: each mouse has its own colourmap (so overlaid mice stay
    distinguishable). Within each ROW GROUP (per mouse), trials ramp across
    ``cmap_range`` (0.2 -> 1.0) in chronological order (session, then trial), so
    earlier trials are lighter. Set ``row_by='session'`` for a per-session ramp,
    ``row_by='day'`` for a per-day ramp across that day's sessions.

    Each cell draws that session's first video frame as a static background and
    fixes the axes to the frame's pixel size, so the whole arena is visible and
    every cell shares the same extent. A session with no DLC pose renders as the
    bare frame (no traces), which makes missing pose visible rather than hidden.

Contents:
--------------------------------
- DEFAULT_CENTROID_POINTS: the DLC keypoints averaged into the centroid by default.
- add_group_column:        add a ``group`` column from a {group: {mice}} mapping.
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

# Sequential colourmaps assigned to mice in turn when the caller gives none. Each
# ramps light -> dark, which is what the "earlier trials lighter" rule wants.
_CMAP_ROTATION = ('Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Greys', 'YlOrBr', 'PuBuGn')




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
# 1| Resolve a Mouse -> Colourmap Mapping
#===============================================================================
def _resolve_mouse_cmaps(mouse_cmaps, mice):
    '''
    Return a ``{mouse: Colormap}`` mapping, auto-filling any mouse not supplied.

    ----------
    Parameters:
        mouse_cmaps (dict | None):
            Caller-supplied ``{mouse: colourmap name or Colormap}``, possibly partial.
        mice (sequence):
            The mice that need a colourmap.
    Returns:
        dict:
            ``{mouse: matplotlib.colors.Colormap}`` for every mouse in ``mice``.
    '''
    resolved = {
        mouse: mpl.colormaps[_CMAP_ROTATION[i % len(_CMAP_ROTATION)]]
        for i, mouse in enumerate(mice)
    }
    if mouse_cmaps:
        for mouse, cmap in mouse_cmaps.items():
            resolved[mouse] = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    return resolved


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
# 4| Overlay One Cell's Trial Paths Onto an Axis
#===============================================================================
def _plot_paths_on_ax(
        ax,
        pose,
        cell_trials: pd.DataFrame,
        *,
        segment: str,
        centroid_points,
        mouse_cmaps: dict,
        cmap_range,
        mouse_column: str,
        ramp_column: str,
        time_coord: str,
        marker_size: float,
):
    '''
    Draw every trial in ``cell_trials`` as a centroid path on ``ax``.

    Each trial is sliced to its ``segment`` window and drawn with ``movement``'s
    ``plot_centroid_trajectory`` (the centroid of ``centroid_points``) in a single
    colour: its mouse's colourmap sampled at the trial's precomputed ramp position
    (``ramp_column``).

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
        mouse_cmaps (dict):
            ``{mouse: Colormap}``.
        cmap_range (tuple):
            ``(lo, hi)`` fractions of the colourmap the trial ramp spans.
        mouse_column (str):
            Trial-table column naming the mouse.
        ramp_column (str):
            Column holding each trial's 0..1 ramp position within its row group.
        time_coord (str):
            Pose time dimension (``'Time'``).
        marker_size (float):
            Scatter marker size passed to ``movement``.
    Returns:
        None.
    '''
    lo, hi = cmap_range
    keypoints = list(centroid_points)
    for _, row in cell_trials.iterrows():
        # Slice this trial's segment; skip trials with no pose in the window (e.g. a
        # missing target-zone time, or a session that has no DLC at all).
        sliced = slice_pose_for_trial(pose, row, segment=segment, time_coord=time_coord)
        if sliced.sizes.get(time_coord, 0) == 0:
            continue

        # Colour: the trial's mouse colourmap at its ramp position (earlier -> lo).
        frac = lo + (hi - lo) * row[ramp_column]
        colour = mcolors.to_hex(mouse_cmaps[row[mouse_column]](frac))

        # movement's scatter: recast to its schema; passing ``c`` overrides its
        # default time-gradient with our solid per-trial colour.
        movement_pose = sliced.rename({time_coord: 'time'}).expand_dims({'individuals': ['ind']})
        _plot_centroid_trajectory(movement_pose, keypoints=keypoints, ax=ax, c=colour, s=marker_size)

    # movement writes a title / axis labels on every call; clear them so the grid
    # can set its own row and column labels.
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')

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
# 2| plot_path_grid
#===============================================================================
def plot_path_grid(
        result: dict,
        root,
        *,
        mice=None,
        groups=None,
        row_by='session',
        col_by='segment',
        segment: str = 'inbound',
        segments=('outbound', 'inbound'),
        centroid_points= ('nose', 'lear', 'rear', 'body', 'tailbase'),
        mouse_cmaps=None,
        cmap_range=(0.2, 1.0),
        trials_key: str = 'trials',
        pose_key: str = 'dlc:position',
        mouse_column: str = 'mouseID',
        session_column: str = 'session',
        time_coord: str = 'Time',
        video_subdir: str = 'VideoData',
        colorbar: bool = True,
        marker_size: float = 1.0,
        fontsize: int = 13,
        figsize_per_cell=(6.4, 5.2),
):
    '''
    Plot per-trial DLC centroid paths on a grid over the arena video frame.

    ``row_by`` and ``col_by`` each take a single axis or a LIST of axes to nest.
    An axis is a trials-table column (``'session'``, ``'day'``, ``'group'``,
    ``'mouseID'``) or ``'segment'`` (the ``segments`` pair). A cell overlays every
    trial matching its row and column values, drawn for the cell's segment, each
    trial coloured by its mouse colourmap at its within-row-group ramp position.

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
            each row group (per mouse).
        col_by (str | list[str]):
            Column axis or nested axes. Default ``'segment'`` (outbound|inbound).
            Use e.g. ``['mouseID', 'segment']`` for mouse columns each split into
            outbound|inbound.
        segment (str):
            The segment drawn when no axis is ``'segment'``. Default ``'inbound'``.
        segments (sequence):
            The segments used when an axis is ``'segment'``. Default
            ``('outbound', 'inbound')``.
        centroid_points (sequence):
            Keypoints averaged into the centroid. Default ``('nose', 'lear', 'rear', 'body', 'tailbase')``.
        mouse_cmaps (dict | None):
            ``{mouse: colourmap}``; any mouse omitted is auto-assigned.
        cmap_range (tuple):
            ``(lo, hi)`` colourmap fractions the trial ramp spans. Default
            ``(0.2, 1.0)``.
        trials_key, pose_key (str):
            Keys of the trials table and pose array in ``result``.
        mouse_column, session_column (str):
            Trials-table columns naming the mouse and session.
        time_coord (str):
            Pose time dimension. Default ``'Time'``.
        video_subdir (str):
            Per-session video subfolder. Default ``'VideoData'``.
        colorbar (bool):
            If True (default), add a slim per-mouse colourbar to each subpanel
            showing the trial ramp, plus a mouse-id / trial-count annotation.
        marker_size (float):
            Scatter marker size passed to ``movement``. Default 1.
        fontsize (int):
            Base font size for titles, row labels, and the per-panel annotation
            (colourbar ticks use ``fontsize - 2``). Default 13.
        figsize_per_cell (tuple):
            ``(width, height)`` inches per cell.
    Returns:
        tuple[matplotlib.figure.Figure, numpy.ndarray]:
            The figure and its 2D array of axes.
    '''

    # 1| Pull the trials table and pose; work on a copy of the trials.
    all_trials = result[trials_key]
    pose = result[pose_key]
    trials = all_trials.copy()

    # 2| Optional grouping and mouse restriction.
    if groups is not None:
        trials = add_group_column(trials, groups, mouse_column=mouse_column)
        trials = trials[trials['group'].notna()]
    if mice is not None:
        trials = trials[trials[mouse_column].isin(list(mice))]
    if trials.empty:
        raise ValueError('no trials left after mice / groups filtering.')
    panel_fontsize = fontsize
    header_fontsize = fontsize + 1
    colorbar_fontsize = max(fontsize - 2, 1)

    # 3| Per-trial colour ramp, computed WITHIN each row group (its non-segment
    #    axes) per mouse, in chronological order (the table is already ordered by
    #    session then trial). So earlier trials in a row are lighter, and a per-day
    #    row ramps continuously across that day's sessions.
    row_specs = [row_by] if isinstance(row_by, str) else list(row_by)
    ramp_columns = list(dict.fromkeys([s for s in row_specs if s != 'segment'] + [mouse_column]))
    grouped = trials.groupby(ramp_columns, sort=False)
    rank = grouped.cumcount()
    size = grouped[mouse_column].transform('size')
    trials = trials.copy()
    trials['_ramp'] = (rank / (size - 1).clip(lower=1)).astype(float)

    # 4| Per-mouse colourmaps for every mouse still present.
    resolved_cmaps = _resolve_mouse_cmaps(mouse_cmaps, list(pd.unique(trials[mouse_column])))

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
            _plot_paths_on_ax(
                ax, pose, cell_trials,
                segment=seg,
                centroid_points=centroid_points,
                mouse_cmaps=resolved_cmaps,
                cmap_range=cmap_range,
                mouse_column=mouse_column,
                ramp_column='_ramp',
                time_coord=time_coord,
                marker_size=marker_size,
            )

            # 7c-2| Annotate each panel: per-mouse id + trial count (top-left).
            counts = cell_trials.groupby(mouse_column, sort=False).size()
            if len(counts):
                ax.text(
                    0.02, 0.98,
                    '\n'.join(f'{mouse}  n={int(counts[mouse])}' for mouse in counts.index),
                    transform=ax.transAxes, va='top', ha='left', fontsize=fontsize, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.45, edgecolor='none'),
                )
            # Per-mouse trial colourbar(s), placed to the RIGHT of the panel (one per
            # mouse present; ticks mark this cell's first / last trial number).
            if colorbar:
                for mouse in counts.index:
                    sub = cell_trials[cell_trials[mouse_column] == mouse]
                    tlo, thi = int(sub['trial_index'].min()), int(sub['trial_index'].max())
                    sm = plt.cm.ScalarMappable(
                        norm=mcolors.Normalize(tlo, thi if thi > tlo else tlo + 1),
                        cmap=_truncate_cmap(resolved_cmaps[mouse], cmap_range),
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
