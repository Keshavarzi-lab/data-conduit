'''
Slice DLC pose by trial-table time windows.
============================================

Description:
    Given the combined output of a Q_C ``DataStructure`` (a ``trials`` table plus
    ``dlc:position`` / ``dlc:confidence`` arrays on a ``Time`` axis), pull out the
    pose for a particular trial or path segment. Both the trial times and the pose
    Time live on the same Bonsai Seconds clock, so a slice is just "the pose entries
    of THIS session whose Time falls in THIS window".

    The session tag matters: after a cross-session combine, the ``Time`` values
    repeat per session (each session has its own clock), so a slice must filter on
    BOTH the session id (the ``session`` coordinate) AND the time range. That is
    what ``slice_pose`` does; the trial-oriented helpers read the window straight
    off a trial row.

    Path segments map to the trial table's time columns:
        outbound -> (outbound_start_time, outbound_end_time)
        inbound  -> (inbound_start_time,  inbound_end_time)
        trial    -> (start_time,          end_time)          # whole trial

Contents:
--------------------------------
- slice_pose:            Pose entries of one session within one time window.
- slice_pose_for_trial:  Pose for one trial row's chosen segment.
- slice_pose_per_trial:  One pose slice per row of a trials table.
'''





################################################################################
# Imports
################################################################################

import numpy as np
import pandas as pd

################################################################################



# Map a path-segment name to the (start, end) time columns on a trial row.
_SEGMENT_COLUMNS = {
    'outbound': ('outbound_start_time', 'outbound_end_time'),
    'inbound': ('inbound_start_time', 'inbound_end_time'),
    'trial': ('start_time', 'end_time'),
}




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| slice_pose (One Session's Pose Within One Time Window)
#===============================================================================
def slice_pose(
        pose,
        *,
        session,
        start: float,
        end: float,
        session_coord: str = 'session',
        time_coord: str = 'Time',
):
    '''
    Return the pose entries of one session whose time falls within ``[start, end]``.

    Filters on BOTH the session tag and the time range, because after a
    cross-session combine the ``Time`` values are per-session (not globally
    unique), so a time range alone would pull in matching times from other
    sessions.

    ----------
    Parameters:
        pose (xr.DataArray | xr.Dataset):
            A combined pose stream (e.g. ``dlc:position``) carrying a ``time_coord``
            dimension and a per-entry ``session_coord`` coordinate.
        session:
            The session id to keep (matched against ``pose[session_coord]``).
        start (float):
            Inclusive lower bound of the time window (Bonsai Seconds).
        end (float):
            Inclusive upper bound of the time window.
        session_coord (str):
            Name of the per-entry session coordinate. Default ``'session'``.
        time_coord (str):
            Name of the time dimension. Default ``'Time'``.
    Returns:
        xr.DataArray | xr.Dataset:
            The slice along ``time_coord``. Empty if the session is absent or the
            window is empty / contains NaN bounds (e.g. an undefined path segment).
    '''
    # Pull the per-entry session and time vectors once.
    sessions = pose.coords[session_coord].values
    times = pose.coords[time_coord].values

    # Keep entries from this session whose time is inside the window. NaN bounds
    # (e.g. a missing target-zone time) make every comparison False, so the slice
    # comes back empty rather than erroring.
    mask = (sessions == session) & (times >= start) & (times <= end)
    return pose.isel({time_coord: np.flatnonzero(mask)})

#===============================================================================



#===============================================================================
# 2| slice_pose_for_trial (Pose For One Trial Row's Segment)
#===============================================================================
def slice_pose_for_trial(
        pose,
        trial_row: pd.Series,
        *,
        segment: str = 'trial',
        session_column: str = 'session',
        session_coord: str = 'session',
        time_coord: str = 'Time',
):
    '''
    Return the pose for one trial row, over the chosen path segment.

    Reads the session and the segment's start / end times straight off the trial
    row, then defers to ``slice_pose``.

    ----------
    Parameters:
        pose (xr.DataArray | xr.Dataset):
            A combined pose stream (e.g. ``dlc:position``).
        trial_row (pd.Series):
            One row of a trials table (e.g. ``trials.iloc[0]``), carrying the
            session column and the segment's time columns.
        segment (str):
            Which window to take: ``'outbound'``, ``'inbound'``, or ``'trial'``
            (the whole start->poke span). Default ``'trial'``.
        session_column (str):
            Column on ``trial_row`` holding its session id. Default ``'session'``.
        session_coord (str):
            Per-entry session coordinate on ``pose``. Default ``'session'``.
        time_coord (str):
            Time dimension on ``pose``. Default ``'Time'``.
    Returns:
        xr.DataArray | xr.Dataset:
            The pose slice for this trial's segment (possibly empty).
    '''
    if segment not in _SEGMENT_COLUMNS:
        raise ValueError(f"segment must be one of {sorted(_SEGMENT_COLUMNS)}, got {segment!r}.")
    start_col, end_col = _SEGMENT_COLUMNS[segment]
    return slice_pose(
        pose,
        session=trial_row[session_column],
        start=trial_row[start_col],
        end=trial_row[end_col],
        session_coord=session_coord,
        time_coord=time_coord,
    )

#===============================================================================



#===============================================================================
# 3| slice_pose_per_trial (One Slice Per Trial)
#===============================================================================
def slice_pose_per_trial(
        pose,
        trials: pd.DataFrame,
        *,
        segment: str = 'trial',
        session_column: str = 'session',
        session_coord: str = 'session',
        time_coord: str = 'Time',
) -> list:
    '''
    Return one pose slice per row of a trials table, for the chosen segment.

    A thin loop over ``slice_pose_for_trial``. Filter ``trials`` first (by mouse,
    day, outcome, ...) to slice only the trials you care about, in that order.

    ----------
    Parameters:
        pose (xr.DataArray | xr.Dataset):
            A combined pose stream.
        trials (pd.DataFrame):
            A trials table (or a filtered subset).
        segment (str):
            ``'outbound'``, ``'inbound'``, or ``'trial'``. Default ``'trial'``.
        session_column, session_coord, time_coord (str):
            As in ``slice_pose_for_trial``.
    Returns:
        list:
            One pose slice per trial row, in the table's row order.
    '''
    return [
        slice_pose_for_trial(
            pose,
            row,
            segment=segment,
            session_column=session_column,
            session_coord=session_coord,
            time_coord=time_coord,
        )
        for _, row in trials.iterrows()
    ]

#===============================================================================



################################################################################
