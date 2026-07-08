from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from pathlib import Path
import io

import pandas as pd

from ..data_conduit_shim import ExperimentEvents
from .parse_events_to_trials import parse_events_to_trials


# Data
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


# Sessions per PI_Data_Summary.xlsx (sheet "Tabelle1").
# Day_N matches the on-disk "Day N - <cue>" folder under Q_C_Analysis_Workflow/data/<mouse>/.
# Lists = multiple sessions on that day (either separate or to be combined; see comments).
# Day_5 = None: the on-disk Day 5 folder exists for both animals but the PI summary lists no sessions.

# TRAINING columns (cols 1-14). Row 9 marks all as "last 40 / last 25".
training_sessions = {
    'FL_M01569519': {
        'Day_1': '',
        'Day_2': '2026-05-11T125643Z',
        'Day_3': '2026-05-12T164637Z',
        'Day_4': '2026-05-13T161503Z',
        'Day_5': '2026-05-14T165819Z',
    },
    'FLR_M01569521': {
        'Day_1': '2026-04-23T160907Z',
        'Day_2': '2026-04-25T190915Z',
        'Day_3': '2026-04-28T124547Z',
        'Day_4': '2026-04-29T161826Z',
        'Day_5': '2026-04-30T151438Z',
    },
    'FbR_M01569522': {
        'Day_1': ['2026-04-21T155201Z','2026-04-21T162256Z'],
        'Day_2': '2026-04-23T165237Z',
        'Day_3': ['2026-04-25T181535Z','2026-04-25T182521Z'],
        'Day_4': ['2026-04-28T163449Z','2026-04-28T164540Z','2026-04-28T165257Z'],
        'Day_5': ['2026-04-29T153239Z', '2026-04-29T155431Z'],  # combine
    },
    'MbL_M01569517': {
        'Day_1': '2026-05-20T112502Z',
        'Day_2': '2026-05-21T120827Z',
        'Day_3': '2026-05-22T150830Z',
        'Day_4': '2026-05-28T154840Z',
        'Day_5': '2026-05-29T155941Z',
    },
    'MR_M01569515': {
        'Day_1': '',
        'Day_2': '2026-05-12T150308Z',
        'Day_3': '2026-05-13T121852Z',
        'Day_4': '2026-05-14T122109Z',
        'Day_5': '2026-05-16T173516Z',
    },
    'MbR_M01569518': {
        'Day_1': '2026-05-14T135225Z',
        'Day_2': '2026-05-16T155926Z',
        'Day_3': '2026-05-17T154506Z',
        'Day_4': '2026-05-19T110125Z',
        'Day_5': '2026-05-20T102711Z',
    },
}

# TEST columns marked "last 40 / last 25" in row 9.
test_sessions_last = {
    'FbR_M01569522': {
        'Day_1': '2026-04-30T170040Z',
        'Day_2': '2026-05-01T182520Z',
        'Day_3': '2026-05-02T170316Z',
        'Day_4': '2026-05-03T164101Z',
        'Day_5': None,
        'Day_6': ['2026-05-05T182002Z', '2026-05-05T183211Z', '2026-05-05T184441Z'],  # combine
        'Day_7': ['2026-05-06T153817Z', '2026-05-06T172454Z'],
    },
    'FLR_M01569521': {
        'Day_1': '2026-05-01T165212Z',
        'Day_2': '2026-05-02T175834Z',
        'Day_3': '2026-05-03T184605Z',
        'Day_4': '2026-05-04T160053Z',
        'Day_5': None,
        'Day_6': ['2026-05-06T161348Z', '2026-05-06T180453Z'],
        'Day_7': ['2026-05-07T184152Z', '2026-05-07T190228Z'],  # combine (LR)
    },
}

# TEST columns marked "first 40 / first 25" in row 9 (the blue cells: OFF-cols 22 & 34).
test_sessions_first = {
    'FbR_M01569522': {
        'Day_1': '2026-04-30T171128Z',
        'Day_4': '2026-05-03T170820Z',
    },
    'FLR_M01569521': {
        'Day_1': '2026-05-01T171522Z',
        'Day_4': '2026-05-04T163513Z',
    },
}


SESSION_DICTS = {
    'training_sessions': training_sessions,
    'test_sessions_last': test_sessions_last,
    'test_sessions_first': test_sessions_first,
}

SESSION_PHASES = {
    'training_sessions': 'Training',
    'test_sessions_last': 'Testing',
    'test_sessions_first': 'Testing',
}

FILTER_PROFILE_SPECS = {
    'all_trials': ('all', None),
    'last_40': ('last', 40),
    'last_25': ('last', 25),
    'first_40': ('first', 40),
    'first_25': ('first', 25),
}


def load_trials_from_sessions(
        data_dir: str | Path = DATA_DIR,
        *,
        session_dict: str = 'training_sessions',
        nosepoke_count: int = 18,
        verbose: bool = False,
) -> pd.DataFrame:
    '''
    Parse sessions listed in one session dictionary.

    Parameters
    ----------
    data_dir : str or Path
        Root folder containing the mouse folders. Defaults to
        ``Q_C_Analysis_Workflow/data``.
    session_dict : {'training_sessions', 'test_sessions_last', 'test_sessions_first'}
        Which session dictionary to use.
    nosepoke_count : int
        Number of nosepoke ports passed to ``parse_events_to_trials``.
    verbose : bool
        If True, print each session as it is loaded.

    Returns
    -------
    pd.DataFrame
        One long table containing the parsed trial rows. It is the output of
        ``parse_events_to_trials`` with four appended columns:
        ``mouseID``, ``training/testing``, ``day``, and ``session``.
    '''

    # Convert data_dir to Path if it's a string, and validate session_dict.
    data_dir = Path(data_dir)
    # Get the session dict for the specified name, and determine the phase (Training or Testing).
    sessions_by_mouse = _get_session_dict(session_dict)

    # Determine the phase (Training or Testing) based on the session_dict name.
    phase = SESSION_PHASES[session_dict]
    training_or_testing = phase.lower()
    
    # Loop through the sessions, find the corresponding session directory, load the events, 
    # parse to trials, and append to the list.
    trial_tables = []


    for mouse_id, sessions_by_day in sessions_by_mouse.items():
        # Loop through the days and session values for this mouse. 
        # The session value can be a single session ID, a list of session IDs, or None.
        for day, session_value in sessions_by_day.items():
            # Loop through the session IDs (if any) for this day. 
            # If session_value is None, this loop will not execute.
            for session_id in _as_session_list(session_value):
                # Find the session directory for this mouse, phase, and session ID.
                session_path = find_session_path(
                    data_dir=data_dir,
                    mouse=mouse_id,
                    phase=phase,
                    session_id=session_id,
                )
                if verbose:
                    print(f'[load] {session_path}')

                # Load the events for this session, suppressing output if not verbose.
                with _quiet_reader(verbose):
                    events_df = ExperimentEvents(
                        experiment_directory_path=session_path
                    ).df

                # Parse the events to trials, and append the trial table to the list.
                trials = parse_events_to_trials(
                    events_df,
                    nosepoke_count=nosepoke_count,
                ).copy()

                # Append metadata columns to the trial table.
                trials['mouseID'] = mouse_id
                trials['training/testing'] = training_or_testing
                trials['day'] = day
                trials['session'] = session_id

                trial_tables.append(trials)

    # Concatenate all the trial tables into one DataFrame. 
    # If there are no tables, return an empty DataFrame.
    if not trial_tables:
        return pd.DataFrame()
    return pd.concat(trial_tables, ignore_index=True)


def load_trials_from_session_dicts(
        data_dir: str | Path = DATA_DIR,
        *,
        session_dict: str = 'training_sessions',
        nosepoke_count: int = 18,
        verbose: bool = False,
) -> pd.DataFrame:
    '''Backward-compatible wrapper for ``load_trials_from_sessions``.'''
    return load_trials_from_sessions(
        data_dir,
        session_dict=session_dict,
        nosepoke_count=nosepoke_count,
        verbose=verbose,
    )


def filter_trials(
        trials: pd.DataFrame,
        *,
        include: dict | None = None,
        exclude: dict | None = None,
        drop_until_last_led_on: bool = False,
        session_columns: tuple[str, ...] = (
            'mouseID',
            'training/testing',
            'day',
            'session',
        ),
        reset_index: bool = True,
        keep_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    '''
    Include or exclude trials based on column values.

    Parameters
    ----------
    trials : pd.DataFrame
        Trial table returned by ``load_trials_from_sessions``.
    include : dict or None
        Rules for rows to keep. Examples:
        ``{'outcome': 'Success'}``, ``{'target_zone_size': [100, 150]}``,
        or ``{'TTP': lambda column: column < 5}``.
    exclude : dict or None
        Rules for rows to drop, using the same format as ``include``.
    drop_until_last_led_on : bool
        If True, for each session, drop every trial up to and including the
        final trial in that session where ``LED == 'ON'``.
    session_columns : tuple[str, ...]
        Columns that identify a session for the LED rule.
    reset_index : bool
        If True, return a DataFrame with a fresh RangeIndex.
    keep_columns : list of str or None
        If not None, list of columns to keep in the returned DataFrame. If None, keep all columns.
    drop_columns : list of str or None
        If not None, list of columns to drop from the returned DataFrame. If None, drop no columns.

    Returns
    -------
    pd.DataFrame
        Filtered trial table.
    '''
    filtered = trials.copy()

    if include:
        mask = pd.Series(True, index=filtered.index)
        for column, rule in include.items():
            mask &= _column_rule_mask(filtered, column, rule)
        filtered = filtered.loc[mask]

    if exclude:
        mask = pd.Series(False, index=filtered.index)
        for column, rule in exclude.items():
            mask |= _column_rule_mask(filtered, column, rule)
        filtered = filtered.loc[~mask]

    if drop_until_last_led_on:
        filtered = _drop_until_last_led_on(filtered, session_columns=session_columns)

    if reset_index:
        filtered = filtered.reset_index(drop=True)

    # Column selection is only an output/display step. It is deliberately done
    # after every row filter so it cannot change which trials are included.
    if keep_columns is not None:
        filtered = filtered.loc[:, _as_column_list(keep_columns)]

    if drop_columns is not None:
        filtered = filtered.drop(columns=_as_column_list(drop_columns), errors='ignore')

    return filtered


def apply_filter_profile(
        trials: pd.DataFrame,
        profile: str = 'all_trials',
        *,
        drop_until_last_led_on: bool = True,
        session_columns: tuple[str, ...] = (
            'mouseID',
            'training/testing',
            'day',
            'session',
        ),
        group_columns: tuple[str, ...] = (
            'mouseID',
            'day',
        ),
        reset_index: bool = True,
        keep_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    '''
    Apply one named filtering profile to a loaded trials table.

    All profiles first apply the session-level LED rule by default: within
    each session, drop trials up to and including the final ``LED == 'ON'``
    trial. Then first/last profiles keep trials within each mouse/day group,
    combining multiple sessions that share the same mouse/day label.

    Parameters
    ----------
    trials : pd.DataFrame
        Trial table returned by ``load_trials_from_sessions``.
    profile : {'all_trials', 'last_40', 'last_25', 'first_40', 'first_25'}
        Which filtering profile to apply after the session-level LED rule.
    drop_until_last_led_on : bool
        If True, first apply the session-level LED rule.
    session_columns : tuple[str, ...]
        Columns that identify a session for the LED rule.
    group_columns : tuple[str, ...]
        Columns that identify the group for first/last trial selection.
        Defaults to ``('mouseID', 'day')``.
    reset_index : bool
        If True, return a DataFrame with a fresh RangeIndex.
    keep_columns : list of str or None
        If not None, list of columns to keep in the returned DataFrame.
    drop_columns : list of str or None
        If not None, list of columns to drop from the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered trial table for the requested profile.
    '''
    if profile not in FILTER_PROFILE_SPECS:
        valid_names = ', '.join(FILTER_PROFILE_SPECS)
        raise ValueError(f"profile must be one of: {valid_names}. Got: {profile}")

    filtered = filter_trials(
        trials,
        drop_until_last_led_on=drop_until_last_led_on,
        session_columns=session_columns,
        reset_index=False,
    )

    mode, count = FILTER_PROFILE_SPECS[profile]
    if mode == 'first':
        filtered = _take_trials_by_group(
            filtered,
            group_columns=group_columns,
            count=count,
            position='first',
        )
    elif mode == 'last':
        filtered = _take_trials_by_group(
            filtered,
            group_columns=group_columns,
            count=count,
            position='last',
        )

    if reset_index:
        filtered = filtered.reset_index(drop=True)

    if keep_columns is not None:
        filtered = filtered.loc[:, _as_column_list(keep_columns)]

    if drop_columns is not None:
        filtered = filtered.drop(columns=_as_column_list(drop_columns), errors='ignore')

    return filtered


def build_filter_profiles(
        trials: pd.DataFrame,
        *,
        profiles: tuple[str, ...] = tuple(FILTER_PROFILE_SPECS),
        **kwargs,
) -> dict[str, pd.DataFrame]:
    '''
    Return multiple named filtering profiles for one loaded trials table.

    ``kwargs`` are passed to ``apply_filter_profile``.
    '''
    return {
        profile: apply_filter_profile(trials, profile=profile, **kwargs)
        for profile in profiles
    }


def find_session_path(
        *,
        data_dir: str | Path = DATA_DIR,
        mouse: str,
        phase: str,
        session_id: str,
) -> Path:
    '''
    Find a session directory under ``data_dir / mouse / phase / Day*``.

    This searches by session id rather than by ``Day_N`` because the dict day
    labels are analysis labels, while the on-disk day folder is the physical
    acquisition folder.
    '''
    phase_dir = Path(data_dir) / mouse / phase
    matches = sorted(
        path for path in phase_dir.glob(f'Day*/{session_id}')
        if path.is_dir()
    )

    if not matches:
        raise FileNotFoundError(
            f'No session directory found for {mouse}/{phase}/{session_id} '
            f'under {phase_dir}'
        )
    if len(matches) > 1:
        raise ValueError(
            f'Multiple session directories found for {mouse}/{phase}/{session_id}: '
            f'{matches}'
        )
    return matches[0]


def _as_session_list(session_value) -> list[str]:
    if session_value is None:
        return []
    if isinstance(session_value, list):
        return session_value
    return [session_value]


def _as_column_list(columns) -> list[str]:
    '''Return a list of column names, treating one string as one column.'''
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _get_session_dict(session_dict: str) -> dict:
    '''Validate session_dict and return the corresponding dict.'''

    # Validate session_dict and return the corresponding dict
    if session_dict not in SESSION_DICTS:
        valid_names = ', '.join(SESSION_DICTS)
        raise ValueError(
            f"session_dict must be one of: {valid_names}. Got: {session_dict}"
        )
    return SESSION_DICTS[session_dict]


def _column_rule_mask(df: pd.DataFrame, column: str, rule) -> pd.Series:
    '''Return a boolean mask for one include/exclude rule.'''
    if column not in df.columns:
        raise KeyError(f"Column not found in trials table: {column}")

    values = df[column]
    if callable(rule):
        mask = rule(values)
    elif isinstance(rule, (list, tuple, set)):
        mask = values.isin(rule)
    else:
        mask = values == rule

    return pd.Series(mask, index=df.index).fillna(False).astype(bool)


def _take_trials_by_group(
        trials: pd.DataFrame,
        *,
        group_columns: tuple[str, ...],
        count: int,
        position: str,
) -> pd.DataFrame:
    '''Keep the first or last count rows within each group.'''
    missing_columns = [
        column for column in group_columns
        if column not in trials.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing columns for profile grouping: {missing_columns}")

    grouped = trials.groupby(list(group_columns), sort=False, group_keys=False)
    if position == 'first':
        return grouped.head(count)
    if position == 'last':
        return grouped.tail(count)

    raise ValueError(f"position must be 'first' or 'last'. Got: {position}")


def _drop_until_last_led_on(
        trials: pd.DataFrame,
        *,
        session_columns: tuple[str, ...],
) -> pd.DataFrame:
    '''Drop trials up to and including the last LED ON trial in each session.'''
    
    # Validate that session_columns, 'trial_index', and 'LED' are in the DataFrame columns.
    missing_columns = [
        column for column in (*session_columns, 'trial_index', 'LED')
        if column not in trials.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing columns for LED filter: {missing_columns}")

    # For each session, find the last LED ON trial by trial_index and keep
    # only later trials from that same session.
    keep_masks = []
    for _, session_trials in trials.groupby(list(session_columns), sort=False):
        led_on = session_trials['LED'].eq('ON')
        if not led_on.any():
            keep_masks.append(pd.Series(True, index=session_trials.index))
            continue

        last_led_on_trial_index = session_trials.loc[led_on, 'trial_index'].max()
        keep_masks.append(
            session_trials['trial_index'] > last_led_on_trial_index
        )

    # If there were no LED ON trials in any session, return the original DataFrame.
    if not keep_masks:
        return trials

    # Concatenate the masks for all sessions and apply to the original DataFrame. Sort by index to preserve original order.
    keep_mask = pd.concat(keep_masks).sort_index()
    return trials.loc[keep_mask]


def _quiet_reader(verbose: bool):
    '''Context manager to suppress stdout from ExperimentEvents reader if not verbose.'''
    if verbose:
        return nullcontext()
    return redirect_stdout(io.StringIO())


__all__ = [
    'DATA_DIR',
    'SESSION_DICTS',
    'SESSION_PHASES',
    'training_sessions',
    'test_sessions_last',
    'test_sessions_first',
    'load_trials_from_sessions',
    'load_trials_from_session_dicts',
    'filter_trials',
    'apply_filter_profile',
    'build_filter_profiles',
    'find_session_path',
]
