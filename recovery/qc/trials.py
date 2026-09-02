'''
Trial-table parser for Q_C ExperimentEvents logs.
=================================================

Description:
    Convert one session's ExperimentEvents log (a Time-indexed DataFrame with an
    ``Event`` column) into a trial table: one row per trial. The logic is ported
    from ``Q_C_Analysis_Workflow/new/parse_events_to_trials.py`` with two
    deliberate changes:

      1. The 1-second gap between trials is now a PARAMETER (``trial_start_buffer``)
         instead of a hardcoded ``+ 1``. It defaults to 1.0 so the parser
         reproduces the Q_C_Analysis_Workflow behaviour exactly; pass 0 (as the
         configured Q_C DataStructure does) to remove the buffer so a trial starts
         the instant the previous one ended.
      2. The four OUTBOUND / INBOUND path boundary times are emitted as columns,
         derived from the trial's start, its "Target zone triggered" time, and its
         end (poke). These bound the time ranges used to slice DLC pose:
             outbound = trial start          -> Target zone triggered
             inbound  = Target zone triggered -> poke (trial end)

    How a trial is delimited (unchanged from the source): the parser walks the
    "Poke:" events in time order. Each poke ENDS a trial. That trial's START is the
    previous poke's time plus ``trial_start_buffer`` (or, for the first trial, the
    "Start trial logic" event). Everything else (outcome, ports, TTP, LED, angle
    offset, target-zone size) is read from the events that fall inside the trial's
    [start, end] window.

Contents:
--------------------------------
- parse_events_to_trials: ExperimentEvents log -> one row per trial.
'''





################################################################################
# Imports
################################################################################

import numpy as np
import pandas as pd

################################################################################




################################################################################
# Private Helpers
################################################################################



#===============================================================================
# 1| Rows Within a [start, end] Time Window
#===============================================================================
def _within_window(
        events_df: pd.DataFrame,
        start_time: float,
        end_time: float,
) -> pd.DataFrame:
    '''
    Return the rows whose time index falls within ``[start_time, end_time]``.

    ----------
    Parameters:
        events_df (pd.DataFrame):
            Events frame with a numeric (Time) index.
        start_time (float):
            Inclusive lower bound of the window.
        end_time (float):
            Inclusive upper bound of the window.
    Returns:
        pd.DataFrame:
            The subset of ``events_df`` inside the window.
    '''
    # Both bounds inclusive: the start event and the closing poke both belong to
    # the trial they delimit.
    return events_df[(events_df.index >= start_time) & (events_df.index <= end_time)]

#===============================================================================



#===============================================================================
# 2| Trial Start Time (the buffer lives here)
#===============================================================================
def _get_start_time(
        events_df: pd.DataFrame,
        trial_index: int,
        previous_trial_end_time: float | None,
        trial_start_buffer: float,
) -> float:
    '''
    Return one trial's start time.

    The FIRST trial in a session starts at the "Start trial logic" event. Every
    later trial starts at the previous trial's end (its poke) plus
    ``trial_start_buffer``. This buffer is the only behavioural knob versus the
    Q_C_Analysis_Workflow source, which hardcoded a 1-second gap.

    ----------
    Parameters:
        events_df (pd.DataFrame):
            The session's events (used only to find "Start trial logic").
        trial_index (int):
            Zero-based position of this trial within the session.
        previous_trial_end_time (float | None):
            The previous trial's end time. None for the first trial.
        trial_start_buffer (float):
            Seconds added to the previous trial's end to get this trial's start.
            0 means trials are contiguous (this one starts when the last ended).
    Returns:
        float:
            The trial's start time.
    '''
    # First trial: anchor on the explicit "Start trial logic" marker.
    if trial_index == 0:
        return events_df[events_df['Event'].str.contains('Start trial logic', na=False)].index[0]
    # Later trials: previous end plus the (configurable) buffer.
    return previous_trial_end_time + trial_start_buffer

#===============================================================================



#===============================================================================
# 3| Trial Outcome From the Poke Event String
#===============================================================================
def _get_outcome(
        poke_event: str,
) -> str:
    '''
    Classify a trial from its closing "Poke:" event string.

    The poke string looks like ``"Poke: {Success=True, ChosenPort=16,
    CorrectPort=16-}"``. The mapping is: Success -> "Success"; a failed poke with
    no port chosen (ChosenPort=-1) -> "Miss"; any other failed poke -> "Failure".

    ----------
    Parameters:
        poke_event (str):
            The closing poke event string for a trial.
    Returns:
        str:
            One of "Success", "Miss", or "Failure".
    '''
    if 'Success=True' in poke_event:
        return 'Success'
    if 'Success=False' in poke_event and 'ChosenPort=-1' in poke_event:
        return 'Miss'
    if 'Success=False' in poke_event and 'ChosenPort=-1' not in poke_event:
        return 'Failure'
    # Anything else is an unexpected log shape; fail loudly rather than mislabel.
    raise ValueError(f'unexpected poke event format: {poke_event}')

#===============================================================================



#===============================================================================
# 4| "Target Zone Triggered" Time Within a Trial Window
#===============================================================================
def _get_tz_triggered_time(
        window: pd.DataFrame,
) -> float:
    '''
    Return the time of the first "Target zone triggered" event in a trial window.

    This is the boundary between the outbound and inbound path segments (and the
    start of the time-to-poke interval). NaN when the trial has no such event
    (e.g. a miss where the animal never entered the target zone).

    ----------
    Parameters:
        window (pd.DataFrame):
            The trial's events, restricted to its [start, end] window.
    Returns:
        float:
            The first "Target zone triggered" timestamp, or NaN if absent.
    '''
    matches = window[window['Event'].str.contains('Target zone triggered', na=False)]
    return matches.index[0] if not matches.empty else np.nan

#===============================================================================




#===============================================================================
# 4b| "Target Zone Available" Time Within a Trial Window
#===============================================================================
def _get_tz_available_time(
        window: pd.DataFrame,
) -> float:
    '''
    Return the time of the first "Target zone available" event in a trial window.

    This is when the target zone is set up / announced for the trial (the same
    event ``_get_target_zone_size`` reads the radius from), which fires shortly
    after the trial start and before "Target zone triggered". It provides an
    alternate outbound-path end (start -> zone available). Unlike the triggered
    time it is present even on a true miss, since the zone is announced whether
    or not the animal reaches it. NaN only when the window has no such event.

    ----------
    Parameters:
        window (pd.DataFrame):
            The trial's events, restricted to its [start, end] window.
    Returns:
        float:
            The first "Target zone available" timestamp, or NaN if absent.
    '''
    matches = window[window['Event'].str.contains('Target zone available', na=False)]
    return matches.index[0] if not matches.empty else np.nan

#===============================================================================



#===============================================================================
# 5| LED State During a Trial
#===============================================================================
def _get_LED(
        window: pd.DataFrame,
) -> str:
    '''
    Return "ON" if a "NosePokesLED ON" event occurs in the trial window, else "OFF".

    ----------
    Parameters:
        window (pd.DataFrame):
            The trial's events, restricted to its [start, end] window.
    Returns:
        str:
            "ON" or "OFF".
    '''
    if window['Event'].str.contains('NosePokesLED ON', na=False).any():
        return 'ON'
    return 'OFF'

#===============================================================================



#===============================================================================
# 6| Signed Angle Offset Between Chosen and Correct Ports
#===============================================================================
def _get_angle_offset(
        window: pd.DataFrame,
        nosepoke_count: int,
) -> float:
    '''
    Return the signed angle (in [-180, 180]) between the chosen and correct ports.

    Ports are evenly spaced around the arena, so the offset is
    ``(ChosenPort - CorrectPort) * (360 / nosepoke_count)`` wrapped into
    [-180, 180]. NaN when there was no poke (ChosenPort = -1).

    ----------
    Parameters:
        window (pd.DataFrame):
            The trial's events, restricted to its [start, end] window.
        nosepoke_count (int):
            Number of ports around the arena (sets the angular spacing).
    Returns:
        float:
            The signed angle offset in degrees, or NaN for a miss.
    '''
    # The closing poke is the first "Poke:" event in the window.
    poke = window[window['Event'].str.startswith('Poke:', na=False)]
    if poke.empty:
        # The window is delimited by a poke, so this should never happen.
        raise ValueError('no poke event found in the trial window.')
    poke_event = poke['Event'].iloc[0]

    # Pull the chosen / correct port integers out of the structured poke string.
    chosen_port = int(poke_event.split('ChosenPort=')[1].split(',')[0])
    correct_port = int(poke_event.split('CorrectPort=')[1].split('-')[0])

    # No poke -> no meaningful angle.
    if chosen_port == -1:
        return np.nan

    # Even spacing -> port difference times the per-port angle, wrapped to +/-180.
    angle_offset = (chosen_port - correct_port) * (360 / nosepoke_count)
    return ((angle_offset + 180) % 360) - 180

#===============================================================================



#===============================================================================
# 7| Target Zone Radius For a Trial
#===============================================================================
def _get_target_zone_size(
        window: pd.DataFrame,
) -> float:
    '''
    Return the target-zone radius from the "Target zone available" event, or NaN.

    The event reads ``"Target zone available (X Y : 740 687 - Radius : 150)"``; the
    radius is parsed out. The zone size can vary trial to trial, so it is read per
    trial rather than once per session.

    ----------
    Parameters:
        window (pd.DataFrame):
            The trial's events, restricted to its [start, end] window.
    Returns:
        float:
            The radius, or NaN if there is no such event in the window.
    '''
    tz = window[window['Event'].str.contains('Target zone available', na=False)]
    if tz.empty:
        return np.nan
    # Split out the integer after "Radius : " up to the closing parenthesis.
    radius_str = tz['Event'].iloc[0].split('Radius : ')[1].split(')')[0]
    return float(radius_str)

#===============================================================================



################################################################################




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| parse_events_to_trials (Events Log -> Trial Table)
#===============================================================================
def parse_events_to_trials(
        events_df: pd.DataFrame,
        *,
        nosepoke_count: int = 18,
        trial_start_buffer: float = 1.0,
) -> pd.DataFrame:
    '''
    Build a one-row-per-trial table from a session's ExperimentEvents log.

    Walks the "Poke:" events in time order. Each poke closes a trial; the trial's
    start is the previous poke plus ``trial_start_buffer`` (or, for the first
    trial, the "Start trial logic" event). Per-trial fields are read from the
    events inside the trial's [start, end] window.

    ----------
    Parameters:
        events_df (pd.DataFrame):
            One session's events: a Time-indexed DataFrame with an ``Event``
            column (i.e. a data-conduit ``ExperimentEvents.df``, flattened to one
            DataFrame).
        nosepoke_count (int):
            Number of ports around the arena, used for the angle offset. Default 18.
        trial_start_buffer (float):
            Seconds inserted between one trial's end and the next trial's start.
            Default 1.0 reproduces Q_C_Analysis_Workflow; pass 0 for contiguous
            trials (no buffer).
    Returns:
        pd.DataFrame:
            One row per trial, with columns:
              * trial_index          : 1-based trial number within the session.
              * start_time           : trial start (see buffer rule above).
              * end_time             : trial end (the closing poke time).
              * tz_triggered_time    : "Target zone triggered" time, or NaN.
              * tz_available_time    : "Target zone available" time (alternate
                                       outbound end: start -> zone announced), or
                                       NaN.
              * outbound_start_time  : = start_time.
              * outbound_end_time    : = tz_triggered_time.
              * inbound_start_time   : = tz_triggered_time.
              * inbound_end_time     : = end_time.
              * TTT                  : time to target = outbound duration
                                       (tz_triggered_time - start_time), or NaN.
              * TTP                  : time to poke = inbound duration
                                       (end_time - tz_triggered_time), or NaN.
              * ChosenPort           : port the animal poked (-1 if none).
              * CorrectPort          : rewarded port for the trial.
              * outcome              : "Success", "Miss", or "Failure".
              * LED                  : "ON" or "OFF".
              * angle_offset         : signed angle (deg) chosen vs correct, or NaN.
              * target_zone_size     : target-zone radius, or NaN.
            An events log with no pokes yields an empty DataFrame.
    '''

    # 1| Collect the closing poke of every trial, in time order. Each "Poke:" event
    #    ends exactly one trial.
    poke_rows = events_df[events_df['Event'].str.startswith('Poke:', na=False)]

    # 2| Walk the pokes, reconstructing each trial's [start, end] window and reading
    #    its fields out of that window.
    trials = []
    previous_trial_end_time = None
    for i, (poke_time, poke_row) in enumerate(poke_rows.iterrows()):
        # 2a| Delimit the trial. End is this poke; start is the previous end plus
        #     the buffer (or the session's "Start trial logic" for the first trial).
        start_time = _get_start_time(events_df, i, previous_trial_end_time, trial_start_buffer)
        end_time = poke_time
        window = _within_window(events_df, start_time, end_time)

        # 2b| The outbound/inbound boundary: when the animal first triggered the
        #     target zone. Reused both for the path-segment times and for TTP.
        tz_triggered_time = _get_tz_triggered_time(window)
        # Alternate outbound end: when the zone was announced (start -> available).
        tz_available_time = _get_tz_available_time(window)

        # 2c| Assemble the trial row.
        trials.append({
            'trial_index': i + 1,
            'start_time': start_time,
            'end_time': end_time,
            'tz_triggered_time': tz_triggered_time,
            'tz_available_time': tz_available_time,
            # Outbound = start -> target-zone-triggered; inbound = triggered -> poke.
            'outbound_start_time': start_time,
            'outbound_end_time': tz_triggered_time,
            'inbound_start_time': tz_triggered_time,
            'inbound_end_time': end_time,
            # Time to target is the OUTBOUND duration (start -> target zone triggered);
            # time to poke is the INBOUND duration (triggered -> poke). Both are NaN if
            # the zone never triggered. Together they sum to the start->poke duration.
            'TTT': (tz_triggered_time - start_time) if not pd.isna(tz_triggered_time) else np.nan,
            'TTP': (end_time - tz_triggered_time) if not pd.isna(tz_triggered_time) else np.nan,
            'ChosenPort': int(poke_row['Event'].split('ChosenPort=')[1].split(',')[0]),
            'CorrectPort': int(poke_row['Event'].split('CorrectPort=')[1].split('-')[0]),
            'outcome': _get_outcome(poke_row['Event']),
            'LED': _get_LED(window),
            'angle_offset': _get_angle_offset(window, nosepoke_count),
            'target_zone_size': _get_target_zone_size(window),
        })

        # 2d| This poke becomes the anchor for the next trial's start.
        previous_trial_end_time = end_time

    # 3| One row per trial (empty frame if the session had no pokes).
    return pd.DataFrame(trials)

#===============================================================================



################################################################################
