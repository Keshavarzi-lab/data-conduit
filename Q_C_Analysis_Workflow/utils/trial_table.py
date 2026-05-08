'''
Configurable trial-table builders for ExperimentEvents data.

Contents
--------
- build_trial_table
    Walk an events DataFrame and return one row per trial. Trial-start,
    trial-end, intermediate phase markers, segments (e.g. inbound/outbound)
    and outcome extraction are all supplied at the call site.
- build_trial_table_for_directory
    Apply ``build_trial_table`` to every session described by a sessions
    DataFrame (as produced by ``collect_session_settings_directory``),
    tagging rows with USID / UTID / mouse / day / session.
- align_trials_with_bonsai
    Per-USID positional alignment between the behavioural trial table
    (from ``build_trial_table_for_directory``) and the Bonsai trials
    table (from ``collect_session_settings_directory``).

Design notes
------------
The matching rule for each trial marker is passed as a plain Python value:

- ``str``           -> substring match (case-sensitive).
- ``re.Pattern``    -> regex search (use ``re.compile(..., re.IGNORECASE)``
                       for case-insensitive matching).
- ``Callable``      -> arbitrary predicate ``(event_string) -> bool``.

There is no module-level "default rule" — the lab's vocabulary is supplied
in the notebook cell that calls these functions. Helpers like
``_flatten_nested_dict`` from ``data_conduit.utils`` are used to walk the
nested ``ExperimentEvents.df`` structure rather than hand-rolling
equivalent walkers.
'''

'''
Configurable trial-table builder for ExperimentEvents data. The trial-definition rules are passed at the call site — no module-level "default config" constants.

build_trial_table(events_df, *, start, end, phases=None, segments=None, outcome_parser=None, event_column='Event', time_column=None) — walks one session's events in time order. Each marker accepts str (substring), re.Pattern (regex), or Callable[[str], bool] (predicate). phases is {name: marker} for intermediate timestamps that produce {name}_time columns. segments is {name: (from, to)} that produces {name}_duration columns. outcome_parser is a Callable[[str], dict] whose returned mapping merges onto the trial row. Returns one row per completed trial.
build_trial_table_for_directory(base_dir, sessions_df, *, start, end, ...) — applies build_trial_table to every session referenced by a sessions DataFrame, tagging rows with USID/UTID/mouse/day/session.
align_trials_with_bonsai(behavioural_trials, bonsai_trials, *, ...) — per-USID positional outer join on trial_index. Bonsai columns get a bonsai_ prefix; an aligned boolean flags rows where both sides exist for that index.
compare_trial_counts(sessions_df, behavioural_trials, *, ...) — per-session two-way comparison of Bonsai-configured count vs behavioural-parsed count. One row per session, one matches boolean.
count_events_per_session(base_dir, sessions_df, *, marker, ...) — per-session count of events matching marker in the ExperimentEvents log. Returns a USID-indexed pd.Series. Used to verify the parser's per-session count against the raw event-marker count.
Helper reuse: _events_to_long_frame uses data_conduit.utils._flatten_nested_dict to walk ExperimentEvents.df rather than reimplementing the walker
'''


#===== Imports

from collections.abc import Callable
from pathlib import Path
import re
from typing import Any

import pandas as pd

from data_conduit.datasources.monosource import ExperimentEvents
from data_conduit.utils import _flatten_nested_dict


#===== Internals

PatternLike = str | re.Pattern | Callable[[str], bool]


def _match(value: str, pattern: PatternLike) -> bool:
    '''Substring match for str, regex search for re.Pattern, predicate call for Callable.'''
    if isinstance(pattern, str):
        return pattern in value
    if isinstance(pattern, re.Pattern):
        return pattern.search(value) is not None
    if callable(pattern):
        return bool(pattern(value))
    raise TypeError(
        f"Pattern must be str, re.Pattern, or Callable; got {type(pattern).__name__}."
    )


def _events_to_long_frame(events_obj: Any, event_column: str) -> pd.DataFrame:
    '''
    Reduce whatever ``ExperimentEvents.df`` returns (DataFrame, dict, or
    nested dict) to a single time-sorted DataFrame.

    Uses ``data_conduit.utils._flatten_nested_dict`` to collapse the nested
    structure, then concatenates and sorts by index. Empty input yields an
    empty DataFrame.
    '''
    if events_obj is None:
        return pd.DataFrame()
    if isinstance(events_obj, pd.DataFrame):
        return events_obj.sort_index()
    if isinstance(events_obj, dict):
        flat = _flatten_nested_dict(events_obj)
        frames = [df for df in flat.values() if isinstance(df, pd.DataFrame) and not df.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0).sort_index()
    raise TypeError(
        f"Unsupported events object type: {type(events_obj).__name__}."
    )


#===== Trial-table builder

def build_trial_table(
        events_df: pd.DataFrame,
        *,
        start: PatternLike,
        end: PatternLike,
        phases: dict[str, PatternLike] | None = None,
        segments: dict[str, tuple[str, str]] | None = None,
        outcome_parser: Callable[[str], dict] | None = None,
        event_column: str = 'Event',
        time_column: str | None = None,
) -> pd.DataFrame:
    '''
    Walk an events log in chronological order and return one row per trial.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events frame. Either ``data_conduit.ExperimentEvents`` shape (an
        ``Event`` column with timestamps on the index) or a frame with an
        explicit ``time_column``.
    start, end : str | re.Pattern | Callable[[str], bool]
        Markers identifying trial start and trial end events. A trial is
        opened when ``start`` matches and closed when ``end`` matches.
        If ``start`` matches again before ``end``, the in-progress trial
        is discarded (mid-trial restart).
    phases : dict[str, marker], optional
        Named intermediate markers, e.g.
        ``{'tz_available': 'Target zone available',
           'tz_triggered': 'Target zone triggered'}``.
        For each match between start and end, the event time is recorded
        on the trial row in a column named ``{phase}_time``. Repeats
        within a single trial overwrite (last wins) — adjust the pattern
        if you need first-only.
    segments : dict[str, (from_marker, to_marker)], optional
        Named segments between two markers, e.g.
        ``{'outbound': ('start', 'tz_triggered'),
           'inbound':  ('tz_triggered', 'end')}``.
        Each side is either ``'start'``, ``'end'``, or a key from
        ``phases``. Output adds a ``{segment}_duration`` column
        (``to_time - from_time``) per segment.
    outcome_parser : Callable[[str], dict], optional
        Called with the end-event string. Returned mapping is merged into
        the trial row. Use this to extract success / chosen_port / etc.
        from a structured event payload.
    event_column : str
        Column on ``events_df`` containing event strings. Default ``'Event'``.
    time_column : str, optional
        Column containing event timestamps. If ``None``, the DataFrame
        index is used (matches the ``ExperimentEvents`` convention).

    Returns
    -------
    pd.DataFrame
        One row per completed trial. Columns:
        ``trial_index, start_time, end_time, {phase}_time...,
        {segment}_duration..., <outcome_parser keys>``.
    '''
    if events_df is None or events_df.empty:
        return pd.DataFrame()
    if event_column not in events_df.columns:
        raise KeyError(
            f"event_column {event_column!r} not in events_df columns: "
            f"{list(events_df.columns)}"
        )

    if time_column is None:
        times = events_df.index.to_numpy()
    else:
        if time_column not in events_df.columns:
            raise KeyError(f"time_column {time_column!r} not in events_df columns")
        times = events_df[time_column].to_numpy()
    events = events_df[event_column].astype(str).to_numpy()

    phases = dict(phases or {})
    segments = dict(segments or {})

    valid_segment_refs = {'start', 'end', *phases.keys()}
    for name, (lhs, rhs) in segments.items():
        for side in (lhs, rhs):
            if side not in valid_segment_refs:
                raise KeyError(
                    f"Segment {name!r} references unknown marker {side!r}. "
                    f"Valid: {sorted(valid_segment_refs)}."
                )

    rows: list[dict] = []
    current: dict | None = None

    for time_value, event_str in zip(times, events):
        if _match(event_str, start):
            current = {'start_time': time_value}
            continue

        if current is None:
            continue

        for phase_name, phase_pattern in phases.items():
            if _match(event_str, phase_pattern):
                current[f'{phase_name}_time'] = time_value

        if _match(event_str, end):
            current['end_time'] = time_value
            if outcome_parser is not None:
                extracted = outcome_parser(event_str) or {}
                current.update(extracted)
            rows.append(current)
            current = None

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)

    for segment_name, (lhs, rhs) in segments.items():
        lhs_col = 'start_time' if lhs == 'start' else 'end_time' if lhs == 'end' else f'{lhs}_time'
        rhs_col = 'start_time' if rhs == 'start' else 'end_time' if rhs == 'end' else f'{rhs}_time'
        table[f'{segment_name}_duration'] = table[rhs_col] - table[lhs_col]

    table.insert(0, 'trial_index', range(1, len(table) + 1))
    return table


#===== Multi-session entrypoint

def build_trial_table_for_directory(
        base_dir: str | Path,
        sessions_df: pd.DataFrame,
        *,
        start: PatternLike,
        end: PatternLike,
        phases: dict[str, PatternLike] | None = None,
        segments: dict[str, tuple[str, str]] | None = None,
        outcome_parser: Callable[[str], dict] | None = None,
        event_column: str = 'Event',
        verbose: bool = False,
) -> pd.DataFrame:
    '''
    Build a behavioural trial table across every session in ``sessions_df``.

    For each row of ``sessions_df`` (which carries ``USID, mouse, day,
    session`` columns from :func:`collect_session_settings_directory`), the
    function locates the session directory under ``base_dir``, loads its
    ``ExperimentEvents`` via the data-conduit reader, runs
    :func:`build_trial_table` with the supplied markers, and tags each
    trial row with the parent identifiers.

    Parameters
    ----------
    base_dir : str or Path
        Same root that was passed to ``collect_session_settings_directory``.
    sessions_df : pd.DataFrame
        Sessions table with at least ``USID, mouse, day, session`` columns.
    start, end, phases, segments, outcome_parser, event_column
        Forwarded verbatim to :func:`build_trial_table`.
    verbose : bool
        If True, print a per-session progress line.

    Returns
    -------
    pd.DataFrame
        Behavioural trials concatenated across all sessions. Leading
        columns: ``UTID, USID, mouse, day, session``, then everything
        produced by :func:`build_trial_table` (starting with
        ``trial_index``).
    '''
    base = Path(base_dir)
    if not base.is_dir():
        raise NotADirectoryError(f"base_dir does not exist: {base}")

    required = {'USID', 'mouse', 'day', 'session'}
    missing = required - set(sessions_df.columns)
    if missing:
        raise KeyError(
            f"sessions_df missing required columns: {sorted(missing)}. "
            f"Build it with collect_session_settings_directory."
        )

    pieces: list[pd.DataFrame] = []
    for _, row in sessions_df.iterrows():
        session_path = base / row['mouse'] / row['day'] / row['session']
        if not session_path.is_dir():
            if verbose:
                print(f"[missing] {row['USID']}: {session_path}")
            continue

        try:
            events_obj = ExperimentEvents(
                experiment_directory_path=str(session_path),
                verbose=False,
            ).df
        except Exception as error:
            if verbose:
                print(f"[skip] {row['USID']}: ExperimentEvents load failed: {error}")
            continue

        events_df = _events_to_long_frame(events_obj, event_column)
        if events_df.empty:
            if verbose:
                print(f"[empty] {row['USID']}: no events")
            continue

        trials_for_session = build_trial_table(
            events_df,
            start=start,
            end=end,
            phases=phases,
            segments=segments,
            outcome_parser=outcome_parser,
            event_column=event_column,
        )
        if trials_for_session.empty:
            if verbose:
                print(f"[zero-trials] {row['USID']}: events present but no trials matched")
            continue

        usid = row['USID']
        trials_for_session.insert(0, 'session', row['session'])
        trials_for_session.insert(0, 'day', row['day'])
        trials_for_session.insert(0, 'mouse', row['mouse'])
        trials_for_session.insert(0, 'USID', usid)
        trials_for_session.insert(
            0,
            'UTID',
            [f"{usid}__trial_{i}" for i in trials_for_session['trial_index']],
        )
        pieces.append(trials_for_session)

        if verbose:
            print(f"[ok] {usid}: {len(trials_for_session)} trial(s)")

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=0, ignore_index=True)


#===== Bonsai alignment

def build_master_trial_table(
        behavioural_trials: pd.DataFrame,
        bonsai_trials: pd.DataFrame,
        *,
        usid_column: str = 'USID',
        behavioural_index_column: str = 'trial_index',
        bonsai_index_column: str = 'trial_index',
        bonsai_prefix: str = 'bonsai_',
) -> pd.DataFrame:
    '''
    Master per-trial table: behavioural columns + Bonsai metadata, one row
    per trial that actually fired.

    Inner-joins on ``(USID, trial_index)``. Bonsai columns get a
    ``bonsai_`` prefix to avoid name collisions. Use this for analysis
    (filtering, plotting); use :func:`align_trials_with_bonsai` if you
    additionally need to see the unused Bonsai trial slots.

    Parameters
    ----------
    behavioural_trials : pd.DataFrame
        Output of :func:`build_trial_table_for_directory`.
    bonsai_trials : pd.DataFrame
        Output of ``collect_session_settings_directory(...)['trials']``.
    usid_column : str
        Column carrying the USID on both frames.
    behavioural_index_column, bonsai_index_column : str
        Trial-position columns to align on, within each USID.
    bonsai_prefix : str
        Prefix applied to Bonsai-side columns to avoid name collisions.

    Returns
    -------
    pd.DataFrame
        One row per fired trial. Behavioural columns unprefixed; Bonsai
        columns prefixed. No ``aligned`` flag — every row is by
        construction.
    '''
    for required, frame_name in (
        ({usid_column, behavioural_index_column}, 'behavioural_trials'),
        ({usid_column, bonsai_index_column}, 'bonsai_trials'),
    ):
        frame = behavioural_trials if frame_name == 'behavioural_trials' else bonsai_trials
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(
                f"{frame_name} missing required columns: {sorted(missing)}."
            )

    bonsai_renamed = bonsai_trials.rename(
        columns={
            col: f'{bonsai_prefix}{col}'
            for col in bonsai_trials.columns
            if col not in (usid_column, bonsai_index_column)
        }
    )

    return (
        behavioural_trials
        .merge(
            bonsai_renamed,
            how='inner',
            left_on=[usid_column, behavioural_index_column],
            right_on=[usid_column, bonsai_index_column],
        )
        .reset_index(drop=True)
    )


def align_trials_with_bonsai(
        behavioural_trials: pd.DataFrame,
        bonsai_trials: pd.DataFrame,
        *,
        usid_column: str = 'USID',
        bonsai_index_column: str = 'trial_index',
        behavioural_index_column: str = 'trial_index',
        bonsai_prefix: str = 'bonsai_',
) -> pd.DataFrame:
    '''
    Per-USID positional alignment between behavioural and Bonsai trial rows.

    For each USID, the behavioural trials produced by
    :func:`build_trial_table_for_directory` and the Bonsai trials produced
    by :func:`collect_session_settings_directory` are joined on
    ``trial_index``. The merge is an outer join so unmatched rows on
    either side become NaN; an ``aligned`` column flags whether both
    sides were present for that index.

    Parameters
    ----------
    behavioural_trials : pd.DataFrame
        Output of :func:`build_trial_table_for_directory`.
    bonsai_trials : pd.DataFrame
        Output of ``collect_session_settings_directory(...)['trials']``.
    usid_column : str
        Column carrying the USID on both frames.
    bonsai_index_column, behavioural_index_column : str
        Trial-position columns to align on, within each USID.
    bonsai_prefix : str
        Prefix applied to Bonsai-side columns to avoid name collisions.

    Returns
    -------
    pd.DataFrame
        Aligned table with one row per ``(USID, trial_index)``. Includes
        all behavioural columns (unprefixed) plus all Bonsai columns
        (prefixed with ``bonsai_prefix``) plus a boolean ``aligned`` flag.
    '''
    for required, frame_name in (
        ({usid_column, behavioural_index_column}, 'behavioural_trials'),
        ({usid_column, bonsai_index_column}, 'bonsai_trials'),
    ):
        frame = behavioural_trials if frame_name == 'behavioural_trials' else bonsai_trials
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(
                f"{frame_name} missing required columns: {sorted(missing)}."
            )

    bonsai_renamed = bonsai_trials.rename(
        columns={
            col: f'{bonsai_prefix}{col}'
            for col in bonsai_trials.columns
            if col not in (usid_column, bonsai_index_column)
        }
    )

    merged = behavioural_trials.merge(
        bonsai_renamed,
        how='outer',
        left_on=[usid_column, behavioural_index_column],
        right_on=[usid_column, bonsai_index_column],
        suffixes=('', f'_{bonsai_prefix.rstrip("_")}'),
        indicator='_align_source',
    )
    merged['aligned'] = merged['_align_source'] == 'both'
    merged = merged.drop(columns='_align_source')
    return merged


def count_events_per_session(
        base_dir: str | Path,
        sessions_df: pd.DataFrame,
        *,
        marker: PatternLike,
        event_column: str = 'Event',
        usid_column: str = 'USID',
        verbose: bool = False,
) -> pd.Series:
    '''
    Count events matching ``marker`` in every session's ExperimentEvents log.

    Walks the same directory tree as :func:`build_trial_table_for_directory`,
    loading ``ExperimentEvents`` per session via the data-conduit reader.
    The marker is applied with the same ``str / re.Pattern / Callable``
    rules used elsewhere in this module, supplied at the call site.

    Parameters
    ----------
    base_dir : str or Path
        Same root passed to :func:`collect_session_settings_directory`.
    sessions_df : pd.DataFrame
        Sessions table; must carry ``usid_column``, ``mouse``, ``day``,
        ``session``.
    marker : str | re.Pattern | Callable[[str], bool]
        Pattern that identifies the events being counted (e.g.
        ``'Start Trial'`` for trial-start events).
    event_column : str
        Column on the events DataFrame to match against. Default ``'Event'``.
    usid_column : str
        Column carrying the session identifier on ``sessions_df``.
    verbose : bool
        If True, print a per-session progress line.

    Returns
    -------
    pd.Series
        Indexed by USID, named ``'n_trials_events'``. Sessions whose
        ``ExperimentEvents`` folder is missing or unreadable yield 0.
    '''
    base = Path(base_dir)
    if not base.is_dir():
        raise NotADirectoryError(f"base_dir does not exist: {base}")

    required = {usid_column, 'mouse', 'day', 'session'}
    missing = required - set(sessions_df.columns)
    if missing:
        raise KeyError(
            f"sessions_df missing required columns: {sorted(missing)}."
        )

    counts: dict[str, int] = {}
    for _, row in sessions_df.iterrows():
        usid = row[usid_column]
        session_path = base / row['mouse'] / row['day'] / row['session']
        if not session_path.is_dir():
            counts[usid] = 0
            if verbose:
                print(f"[missing] {usid}: {session_path}")
            continue

        try:
            events_obj = ExperimentEvents(
                experiment_directory_path=str(session_path),
                verbose=False,
            ).df
        except Exception as error:
            counts[usid] = 0
            if verbose:
                print(f"[skip] {usid}: ExperimentEvents load failed: {error}")
            continue

        events_df = _events_to_long_frame(events_obj, event_column)
        if events_df.empty or event_column not in events_df.columns:
            counts[usid] = 0
            continue

        events = events_df[event_column].astype(str).to_numpy()
        n = int(sum(1 for ev in events if _match(ev, marker)))
        counts[usid] = n
        if verbose:
            print(f"[ok] {usid}: {n}")

    series = pd.Series(counts, name='n_trials_events')
    series.index.name = usid_column
    return series


def compare_trial_counts(
        sessions_df: pd.DataFrame,
        behavioural_trials: pd.DataFrame,
        *,
        usid_column: str = 'USID',
        bonsai_count_column: str = 'n_trials',
) -> pd.DataFrame:
    '''
    Per-session comparison of behavioural trial count vs Bonsai trial count.

    Counts behavioural rows per USID, joins onto the sessions table, and
    returns one row per session with the Bonsai count, behavioural count,
    delta, and a boolean ``matches`` flag.

    Parameters
    ----------
    sessions_df : pd.DataFrame
        Sessions table from :func:`collect_session_settings_directory`.
    behavioural_trials : pd.DataFrame
        Trial table from :func:`build_trial_table_for_directory`. One row
        per behavioural trial.
    usid_column : str
        Column carrying the session identifier on both frames.
    bonsai_count_column : str
        Column on ``sessions_df`` holding the Bonsai-configured count.

    Returns
    -------
    pd.DataFrame
        One row per session with columns ``USID, mouse, day, session,
        n_trials, n_trials_behavioural, delta, matches``. ``delta`` is
        ``behavioural - bonsai``; ``matches`` is True when the two
        counts agree.
    '''
    for required, frame_name in (
        ({usid_column, bonsai_count_column}, 'sessions_df'),
        ({usid_column}, 'behavioural_trials'),
    ):
        frame = sessions_df if frame_name == 'sessions_df' else behavioural_trials
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(
                f"{frame_name} missing required columns: {sorted(missing)}."
            )

    behavioural_counts = (
        behavioural_trials
        .groupby(usid_column)
        .size()
        .rename('n_trials_behavioural')
        .reset_index()
    )

    keep = [c for c in (usid_column, 'mouse', 'day', 'session', bonsai_count_column)
            if c in sessions_df.columns]
    comparison = (
        sessions_df[keep]
        .merge(behavioural_counts, on=usid_column, how='left')
        .assign(
            n_trials_behavioural=lambda d: d['n_trials_behavioural'].fillna(0).astype(int),
            delta=lambda d: d['n_trials_behavioural'] - d[bonsai_count_column],
            matches=lambda d: d['n_trials_behavioural'] == d[bonsai_count_column],
        )
    )
    return comparison


def count_premature_pokes_per_directory(
        base_dir: str | Path,
        sessions_df: pd.DataFrame,
        *,
        start_marker: PatternLike,
        poke_marker: PatternLike,
        poke_extractor: Callable[[str], dict],
        port_field: str = 'chosen_port',
        correct_field: str = 'correct_port',
        event_column: str = 'Event',
        usid_column: str = 'USID',
        verbose: bool = False,
) -> pd.DataFrame:
    '''
    Per-session count of premature pokes — Poke events that occur AFTER one
    trial closed and BEFORE the next ``start_marker`` event fires.

    A premature poke is classified as *correct* when its chosen port equals
    the upcoming trial's correct port (animal poked the right port early)
    and *incorrect* otherwise.

    The whole rule lives at the call site:

    - ``start_marker`` — what marks the start of a real trial
      (matches the value passed to :func:`build_trial_table_for_directory`).
    - ``poke_marker`` — what marks any poke event in the events log.
    - ``poke_extractor`` — callable returning a dict for any poke string,
      with ``port_field`` and ``correct_field`` keys.

    Parameters
    ----------
    base_dir : str or Path
        Bonsai tree root.
    sessions_df : pd.DataFrame
        Sessions table from :func:`collect_session_settings_directory`.
    start_marker, poke_marker : str | re.Pattern | Callable[[str], bool]
        Markers identifying trial-start and poke events, same convention
        as elsewhere in this module.
    poke_extractor : Callable[[str], dict]
        Function called with every poke event string. Must return a dict
        with ``port_field`` (the poked port) and ``correct_field`` (the
        rewarded port for that poke). Use
        :func:`Q_C_Analysis_Workflow.utils.parse_poke_outcome`.
    port_field, correct_field : str
        Keys read from the dict returned by ``poke_extractor``.
    event_column : str
        Column on the events DataFrame to match against.
    usid_column : str
        USID column on ``sessions_df``.
    verbose : bool
        Print per-session progress.

    Returns
    -------
    pd.DataFrame
        One row per session with columns ``USID, mouse, day, session,
        n_pokes, n_premature_correct, n_premature_incorrect,
        n_premature_total``.
    '''
    base = Path(base_dir)
    if not base.is_dir():
        raise NotADirectoryError(f"base_dir does not exist: {base}")

    rows: list[dict] = []
    for _, session_row in sessions_df.iterrows():
        usid = session_row[usid_column]
        session_path = base / session_row['mouse'] / session_row['day'] / session_row['session']
        record = {
            'USID': usid,
            'mouse': session_row['mouse'],
            'day': session_row['day'],
            'session': session_row['session'],
            'n_pokes': 0,
            'n_premature_correct': 0,
            'n_premature_incorrect': 0,
            'n_premature_total': 0,
        }

        if not session_path.is_dir():
            rows.append(record)
            continue

        try:
            events_obj = ExperimentEvents(
                experiment_directory_path=str(session_path),
                verbose=False,
            ).df
        except Exception as error:
            if verbose:
                print(f"[skip] {usid}: ExperimentEvents load failed: {error}")
            rows.append(record)
            continue

        events_df = _events_to_long_frame(events_obj, event_column)
        if events_df.empty:
            rows.append(record)
            continue

        events = events_df[event_column].astype(str).to_numpy()

        # Walk events in time order, flagging Pokes by whether they occur
        # while a trial is open (= regular trial-end) or not (= premature).
        # The "upcoming correct port" for a premature poke is the chosen-correct
        # extracted from the next trial-end Poke we encounter.
        trial_open = False
        premature_buffer: list[dict] = []  # premature pokes awaiting next trial-end correct port

        for event_str in events:
            if _match(event_str, start_marker):
                trial_open = True
                continue

            if _match(event_str, poke_marker):
                record['n_pokes'] += 1
                extracted = poke_extractor(event_str) or {}

                if trial_open:
                    # This Poke closes the current trial. Resolve any pending
                    # premature pokes against this trial's correct port.
                    correct = extracted.get(correct_field)
                    for pending in premature_buffer:
                        if (
                            correct is not None
                            and pending.get(port_field) == correct
                        ):
                            record['n_premature_correct'] += 1
                        else:
                            record['n_premature_incorrect'] += 1
                    premature_buffer.clear()
                    trial_open = False
                else:
                    # Premature: hold until we see the next trial-end Poke,
                    # whose correct port tells us what the animal "should"
                    # have poked.
                    premature_buffer.append(extracted)

        # Any premature pokes still buffered have no following trial → can't
        # classify; count them as incorrect (no matching correct port).
        record['n_premature_incorrect'] += len(premature_buffer)
        record['n_premature_total'] = (
            record['n_premature_correct'] + record['n_premature_incorrect']
        )
        rows.append(record)

        if verbose:
            print(
                f"[ok] {usid}: pokes={record['n_pokes']} "
                f"premature={record['n_premature_total']} "
                f"(correct={record['n_premature_correct']})"
            )

    return pd.DataFrame(rows)


__all__ = [
    'align_trials_with_bonsai',
    'build_master_trial_table',
    'build_trial_table',
    'build_trial_table_for_directory',
    'compare_trial_counts',
    'count_events_per_session',
    'count_premature_pokes_per_directory',
]
