'''
The Q_C nosepoke task as a TrialSpec.
=====================================

Description:
    Everything experiment-specific about a Q_C trial, expressed as data for the
    general parser in ``data_conduit._refactor.datastructures.trials``. The event
    vocabulary ("Poke:", "Target zone triggered", ...), the arena port count and
    the TTT / TTP durations all live HERE, next to the experiment, rather than in
    the library.

    This reproduces ``data_conduit.qc.trials.parse_events_to_trials`` exactly --
    same columns, same values -- via ``parse_trials(events, qc_trial_spec(...))``.
    The trial model itself is unchanged: each "Poke:" event ends a trial, whose
    start is the previous poke plus ``trial_start_buffer`` (or, for the first
    trial, the "Start trial logic" event).

    Two named path segments come out of the spec:
        outbound = trial start           -> Target zone triggered
        inbound  = Target zone triggered -> poke (trial end)
    The whole-trial window (start_time -> end_time) is always available and does
    not need declaring; see ``segment_bounds``.

Contents:
--------------------------------
- qc_trial_spec:     Build the Q_C TrialSpec for a port count and inter-trial buffer.
- QC_TRIALS:         The Q_C TrialSpec with Q_C_Analysis_Workflow's defaults.
- qc_trials_reader:  A ``path -> trial table`` Catalog reader for the Q_C task.
'''




################################################################################
# Imports
################################################################################

from pathlib import Path

import numpy as np
import pandas as pd

from data_conduit.datastructures import TrialSpec, first_matching, parse_trials

################################################################################




################################################################################
# Private Helpers (Q_C event vocabulary)
################################################################################



#===============================================================================
# 1| Ports Out of a "Poke:" Event String
#===============================================================================
def _get_ports(
        poke_event: str,
) -> tuple[int, int]:
    '''
    Return ``(chosen_port, correct_port)`` from a closing "Poke:" event string.

    The poke string looks like ``"Poke: {Success=True, ChosenPort=16,
    CorrectPort=16-}"``. A chosen port of -1 means the animal never poked.

    ----------
    Parameters:
        poke_event (str):
            The closing poke event string for a trial.
    Returns:
        tuple[int, int]:
            The chosen and correct port numbers.
    '''
    chosen_port = int(poke_event.split('ChosenPort=')[1].split(',')[0])
    correct_port = int(poke_event.split('CorrectPort=')[1].split('-')[0])
    return chosen_port, correct_port

#===============================================================================



#===============================================================================
# 2| Trial Outcome From the Poke Event String
#===============================================================================
def _get_outcome(
        poke_event: str,
) -> str:
    '''
    Classify a trial from its closing "Poke:" event string.

    The mapping is: Success -> "Success"; a failed poke with no port chosen
    (ChosenPort=-1) -> "Miss"; any other failed poke -> "Failure".

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
# 3| Signed Angle Offset Between Chosen and Correct Ports
#===============================================================================
def _get_angle_offset(
        chosen_port: int,
        correct_port: int,
        nosepoke_count: int,
) -> float:
    '''
    Return the signed angle (in [-180, 180]) between the chosen and correct ports.

    Ports are evenly spaced around the arena, so the offset is
    ``(chosen_port - correct_port) * (360 / nosepoke_count)`` wrapped into
    [-180, 180]. NaN when there was no poke (chosen port -1).

    ----------
    Parameters:
        chosen_port (int):
            The port the animal poked (-1 if none).
        correct_port (int):
            The rewarded port for the trial.
        nosepoke_count (int):
            Number of ports around the arena (sets the angular spacing).
    Returns:
        float:
            The signed angle offset in degrees, or NaN for a miss.
    '''
    # No poke -> no meaningful angle.
    if chosen_port == -1:
        return np.nan

    # Even spacing -> port difference times the per-port angle, wrapped to +/-180.
    angle_offset = (chosen_port - correct_port) * (360 / nosepoke_count)
    return ((angle_offset + 180) % 360) - 180

#===============================================================================



#===============================================================================
# 4| Target Zone Radius From a "Target zone available" Event
#===============================================================================
def _get_target_zone_size(
        zone_row: pd.Series,
) -> float:
    '''
    Return the target-zone radius from a "Target zone available" event row.

    The event reads ``"Target zone available (X Y : 740 687 - Radius : 150)"``;
    the radius is parsed out. The zone size can vary trial to trial, so it is
    read per trial rather than once per session.

    ----------
    Parameters:
        zone_row (pd.Series):
            The matched event row (its ``Event`` text holds the radius).
    Returns:
        float:
            The radius in the same units as the log.
    '''
    # Split out the number after "Radius : " up to the closing parenthesis.
    return float(zone_row['Event'].split('Radius : ')[1].split(')')[0])

#===============================================================================



################################################################################




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| qc_trial_spec (The Q_C Trial Description)
#===============================================================================
def qc_trial_spec(
        *,
        nosepoke_count: int = 18,
        trial_start_buffer: float = 1.0,
) -> TrialSpec:
    '''
    Build the Q_C ``TrialSpec`` for a given port count and inter-trial buffer.

    ----------
    Parameters:
        nosepoke_count (int):
            Number of arena ports, used for the angle offset. Default 18.
        trial_start_buffer (float):
            Seconds inserted between one trial's end and the next trial's start.
            Default 1.0, reproducing Q_C_Analysis_Workflow; pass 0.0 for
            contiguous trials (what the Q_C Catalog uses).
    Returns:
        TrialSpec:
            The spec to hand to ``parse_trials`` alongside a session's events.
    '''

    def qc_fields(window, closing):
        '''Per-trial values read out of one Q_C trial's events.'''
        # The outbound / inbound boundary: when the animal first triggered the
        # target zone. NaN on a trial where the zone was never reached.
        tz_triggered_time = first_matching(window, 'Target zone triggered')
        # When the zone was announced. Present even on a true miss, so it gives
        # an alternate outbound end (start -> zone available).
        tz_available_time = first_matching(window, 'Target zone available')

        # The ports come from the closing poke's structured event string.
        poke_event = closing['Event']
        chosen_port, correct_port = _get_ports(poke_event)

        return {
            'tz_triggered_time': tz_triggered_time,
            'tz_available_time': tz_available_time,
            'ChosenPort': chosen_port,
            'CorrectPort': correct_port,
            'outcome': _get_outcome(poke_event),
            # The LED is a presence test: a single "on" event anywhere in the
            # trial means it was lit for that trial.
            'LED': first_matching(
                window, 'NosePokesLED ON', extract=lambda row: 'ON', default='OFF',
            ),
            'angle_offset': _get_angle_offset(chosen_port, correct_port, nosepoke_count),
            'target_zone_size': first_matching(
                window, 'Target zone available', extract=_get_target_zone_size,
            ),
        }

    def qc_derived(row):
        '''Durations worked out from a Q_C trial's delimiting times.'''
        # Time to target is the OUTBOUND duration (start -> target zone
        # triggered); time to poke is the INBOUND duration (triggered -> poke).
        # Both are NaN if the zone never triggered. Together they sum to the
        # start -> poke duration.
        tz_triggered_time = row['tz_triggered_time']
        if pd.isna(tz_triggered_time):
            return {'TTT': np.nan, 'TTP': np.nan}
        return {
            'TTT': tz_triggered_time - row['start_time'],
            'TTP': row['end_time'] - tz_triggered_time,
        }

    return TrialSpec(
        closes_trial='Poke:',
        session_start='Start trial logic',
        start_buffer=trial_start_buffer,
        fields=qc_fields,
        derived=qc_derived,
        segments={
            'outbound': ('start_time', 'tz_triggered_time'),
            'inbound': ('tz_triggered_time', 'end_time'),
        },
    )

#===============================================================================



#===============================================================================
# 2| QC_TRIALS (The Spec With Q_C_Analysis_Workflow's Defaults)
#===============================================================================
# 18 arena ports and the 1-second inter-trial gap: the values the original
# Q_C_Analysis_Workflow parser hardcoded. The Catalog builds its own spec when it
# needs a different buffer.
QC_TRIALS = qc_trial_spec()

#===============================================================================



#===============================================================================
# 3| qc_trials_reader (Trial Table As an Ordinary Catalog Reader)
#===============================================================================
def qc_trials_reader(
        spec: TrialSpec = QC_TRIALS,
):
    '''
    Return a ``path -> trial table`` reader for a Catalog.

    A trial table is a DataFrame, so it registers with
    ``Catalog.add_reader('trials', qc_trials_reader())`` exactly like ``events``
    or ``nosepoke`` -- no special handling. The reader owns its whole pipeline
    (read this session's event log, parse it to trials), so the raw event log is
    never carried into the combined output.

    ----------
    Parameters:
        spec (TrialSpec):
            The trial description to parse with. Default ``QC_TRIALS``; pass
            ``qc_trial_spec(trial_start_buffer=0.0)`` for contiguous trials.
    Returns:
        Callable[[Path], pd.DataFrame]:
            The reader, ready for ``Catalog.add_reader``.
    '''
    # Imported here so importing this module stays light and free of reader
    # dependencies (the spec itself only needs pandas).
    from data_conduit.datasources.monosource import ExperimentEvents
    from data_conduit.core.utils import _concat_split_dataframes

    def read_trials(path: Path) -> pd.DataFrame:
        # Flatten first: a split session logs several CSVs, and the trial parser
        # wants one continuous, time-ordered event frame.
        events = _concat_split_dataframes(ExperimentEvents(experiment_directory_path=path).df)
        return parse_trials(events, spec)

    return read_trials

#===============================================================================



################################################################################
