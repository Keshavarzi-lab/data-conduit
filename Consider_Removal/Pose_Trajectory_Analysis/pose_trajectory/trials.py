'''
Per-trial reference table with inbound/outbound segment windows.
================================================================

Description:
    Turns each session's ExperimentEvents log into the per-trial summary/reference
    table the analysis is built around. It reuses the lab's canonical event parser
    (``parse_trials``) rather than re-implementing it, then derives the four segment
    time boundaries (outbound and inbound start/end) and stitches one tidy table
    across all sessions of a mouse.

    Trial phases, from ``parse_trials``, occur in order:
        Start Trial -> Target zone available -> Target zone triggered ->
        Await poke -> Poke
    From these we define, per trial i (within a session):

        outbound_start = poke_time(i-1)    [the previous trial's poke]
                         or, for the first trial, trial_start(i)
        outbound_end   = tz_triggered(i)
        inbound_start  = tz_triggered(i)
        inbound_end    = poke_time(i)

    So every trial has BOTH an outbound leg (the run up to entering the target zone)
    and an inbound leg (target zone to the poke). All times are in seconds on the
    session clock, the SAME clock as the DLC pose, so they slice a movement dataset
    directly with ``ds.sel(time=slice(start, end))``.

    When a trial has no ``tz_triggered`` (the animal was already in the zone when it
    became available), we fall back to ``tz_available`` for that boundary so the legs
    stay defined.

Contents:
--------------------------------
- parse_trials:        Re-exported lab-canonical events -> trials parser.
- segment_windows:     Add outbound/inbound start/end columns to a trials table.
- build_summary_table: Assemble the per-trial reference table across a SessionGroup.
'''

from __future__ import annotations

import pandas as pd

# parse_trials is the lab's agreed reference implementation; the package __init__ put
# the q_c_data_analysis tree on sys.path so we import it rather than copy it.
from firstdata.behavioural_metrics.event_parsing import parse_trials

from Consider_Removal.Pose_Trajectory_Analysis.pose_trajectory.workspace import day_order

# Re-exported so notebooks can `from pose_trajectory.trials import parse_trials`.
__all__ = ['parse_trials', 'segment_windows', 'build_summary_table']

# Capitalised display form for the outcome ('success' -> 'Success'), used in the table.
_RESULT_LABELS = {'success': 'Success', 'fail': 'Fail', 'miss': 'Miss'}


def segment_windows(
        trials: pd.DataFrame,
        *,
        tz_fallback: bool = True,
) -> pd.DataFrame:
    '''
    Compute outbound/inbound start and end times for each trial.

    Implements the per-trial leg definition described in this module's header. The
    input must be in chronological order (``parse_trials`` already returns it so) and
    is treated by position, so its index is reset internally.

    ----------
    Parameters:
        trials (pd.DataFrame):
            One session's trials from ``parse_trials``. Must have columns
            ``trial_start``, ``tz_triggered``, ``tz_available``, ``poke_time``.
        tz_fallback (bool):
            If True (default), use ``tz_available`` wherever ``tz_triggered`` is
            missing (animal already in the zone), so the legs stay defined. If False,
            those rows keep NaN boundaries.
    Returns:
        pd.DataFrame:
            A frame aligned to ``trials`` with columns ``outbound_start``,
            ``outbound_end``, ``inbound_start``, ``inbound_end`` (seconds).
    '''
    # Work positionally on a clean 0..n-1 index.
    t = trials.reset_index(drop=True)

    # The zone-entry boundary is tz_triggered, optionally backfilled with tz_available
    # for trials where the animal was already inside when the zone became available.
    tz_boundary = t['tz_triggered']
    if tz_fallback:
        tz_boundary = tz_boundary.fillna(t['tz_available'])

    # Outbound starts at the PREVIOUS trial's poke; shift(1) brings poke(i-1) onto row i.
    outbound_start = t['poke_time'].shift(1)
    # The first trial has no previous poke, so start its outbound at its own Start Trial.
    if len(t) > 0:
        outbound_start.iloc[0] = t['trial_start'].iloc[0]

    # Assemble the four boundaries. Outbound ends where inbound starts (zone entry),
    # and inbound ends at this trial's poke.
    return pd.DataFrame({
        'outbound_start': outbound_start.to_numpy(),
        'outbound_end': tz_boundary.to_numpy(),
        'inbound_start': tz_boundary.to_numpy(),
        'inbound_end': t['poke_time'].to_numpy(),
    })


def build_summary_table(
        group,
        animal_id: str,
        *,
        tz_fallback: bool = True,
) -> pd.DataFrame:
    '''
    Build the per-trial reference table across every session in a group.

    For each session it parses the events into trials, derives the segment windows,
    and emits one row per trial. After stacking all sessions it numbers the trials
    within each day (ordered across that day's sessions by time).

    ----------
    Parameters:
        group (SessionGroup):
            Loaded sessions (from ``loading.build_session_group``). Each session must
            expose an ``events`` member and carry ``day`` / ``label`` metadata.
        animal_id (str):
            The mouse identifier (typically the mouse folder name), written into the
            ``animal`` column. Passed in at the call site rather than inferred.
        tz_fallback (bool):
            Forwarded to ``segment_windows``. Default True.
    Returns:
        pd.DataFrame:
            One row per trial with columns, in order:
            ``animal``, ``day``, ``trial_in_day``, ``session``, ``trial_in_session``,
            ``result`` (Success/Fail/Miss), ``actual_poke`` (rewarded port),
            ``chosen_poke`` (poked port, -1 for a miss), ``outbound_start``,
            ``outbound_end``, ``inbound_start``, ``inbound_end``.
    '''
    # 1| One sub-table per session.
    per_session: list[pd.DataFrame] = []
    for session in group:
        day = session.metadata.get('day')
        label = session.metadata.get('label')

        # 1a| Parse this session's events into trials; skip sessions with none.
        trials = parse_trials(session.data['events'])
        if trials.empty:
            continue

        # 1b| Derive the four segment-time columns for these trials.
        seg = segment_windows(trials, tz_fallback=tz_fallback)

        # 1c| Build the rows: identity + outcome + segment windows. trial_in_session is
        #     1-based to read naturally. result is the capitalised outcome.
        per_session.append(pd.DataFrame({
            'animal': animal_id,
            'day': day,
            'session': label,
            'trial_in_session': range(1, len(trials) + 1),
            'result': trials['outcome'].map(_RESULT_LABELS).to_numpy(),
            'actual_poke': trials['correct_port'].to_numpy(),
            'chosen_poke': trials['chosen_port'].to_numpy(),
            'outbound_start': seg['outbound_start'].to_numpy(),
            'outbound_end': seg['outbound_end'].to_numpy(),
            'inbound_start': seg['inbound_start'].to_numpy(),
            'inbound_end': seg['inbound_end'].to_numpy(),
        }))

    # 2| Guard against an empty group / no parseable trials.
    if not per_session:
        raise ValueError('no trials parsed from any session in the group.')

    # 3| Stack all sessions, then order by day then session then within-session trial
    #    so the day-level numbering below counts in true chronological order.
    table = pd.concat(per_session, ignore_index=True)
    table['_day_sort'] = table['day'].map(day_order)
    table = table.sort_values(['_day_sort', 'session', 'trial_in_session']).reset_index(drop=True)

    # 4| Number trials within each day, across that day's sessions (1-based).
    table['trial_in_day'] = table.groupby('day').cumcount() + 1

    # 5| Drop the helper sort column and present the columns in the agreed order.
    columns = [
        'animal', 'day', 'trial_in_day', 'session', 'trial_in_session',
        'result', 'actual_poke', 'chosen_poke',
        'outbound_start', 'outbound_end', 'inbound_start', 'inbound_end',
    ]
    return table[columns]
