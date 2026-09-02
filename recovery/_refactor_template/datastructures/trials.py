'''
Turn a session's event log into a trial table: one row per trial.
=================================================================

Description:
    A general, experiment-agnostic trial parser. It follows the same data +
    functions shape as the rest of ``datastructures/``: a ``TrialSpec`` DATA
    OBJECT describes what a trial looks like for one experiment, and the
    ``parse_trials`` FUNCTION applies that description to one session's events.
    There is deliberately no ``Trials`` base class to subclass.

    The trial model this module implements:

      * A trial is ENDED by a closing event (``TrialSpec.closes_trial``, matched
        with ``startswith``). Every closing event closes exactly one trial, so
        the closing events, walked in time order, ARE the trials.
      * A trial STARTS where the previous one ended, plus ``start_buffer``
        seconds. The very first trial instead starts at ``session_start`` (or at
        the session's first event when that is None).
      * Everything else about a trial is read from the events that fall inside
        its ``[start, end]`` window, by the spec's ``fields`` callable, with any
        values computed FROM those values supplied by ``derived``.
      * Named time-ranges (``segments``) become ``{name}_start_time`` /
        ``{name}_end_time`` column pairs, copied from columns ``fields``
        produced. This is the shape a downstream interval type wants.

    NOTHING in this module knows about any one experiment: no event text, no
    port counts, no task vocabulary. That all lives in the experiment's own
    ``TrialSpec`` (see ``data_conduit.qc.trial_spec`` for a worked example).

Contents:
--------------------------------
- TrialSpec:        Data object describing one experiment's trial structure.
- parse_trials:     Apply a TrialSpec to one session's events -> trial table.
- within:           The rows of an events frame inside a [start, end] window.
- first_matching:   First event in a window matching a pattern (time or value).
- segment_bounds:   {segment name: (start column, end column)} for a spec.
'''




################################################################################
# Imports
################################################################################

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

################################################################################


__all__ = [
    'TrialSpec',
    'parse_trials',
    'within',
    'first_matching',
    'segment_bounds',
]




################################################################################
# The Spec (Data Object)
################################################################################



#===============================================================================
# 1| TrialSpec (What a Trial Looks Like For One Experiment)
#===============================================================================
@dataclass
class TrialSpec:
    '''
    Describes how to turn a session's event log into one row per trial.

    A trial ends with a CLOSING event (``closes_trial``). Its start is the end of
    the previous trial plus ``start_buffer`` -- or ``session_start`` for the very
    first trial. Values for each trial are read from the events between its start
    and end by the ``fields`` callable; values computed from those values (e.g.
    durations) come from ``derived``. Named time-ranges go in ``segments``.

    ----------
    Parameters:
        closes_trial (str):
            The start of the event text that ends a trial (matched with
            ``str.startswith``).
        session_start (str | None):
            Text identifying the event that marks the FIRST trial's start
            (matched as a literal substring). None -> use the session's first
            event. When given but absent from the log, parsing raises: a session
            whose start marker was never logged cannot be delimited.
        start_buffer (float):
            Seconds added between one trial's end and the next trial's start.
            0.0 makes trials contiguous.
        fields (Callable[[pd.DataFrame, pd.Series], dict]):
            Given the events in a trial (``window``) and the row that ended it
            (``closing_row``), return a dict of column values for that trial. Use
            NaN for a value that is missing. Default: no extra columns.
        derived (Callable[[dict], dict]):
            Given the trial row assembled so far (the delimiting times plus
            everything ``fields`` returned), return a dict of FURTHER column
            values worked out from it -- durations and the like. Kept separate so
            ``fields`` stays a plain "read values out of a window" function.
            Default: no extra columns.
        segments (dict[str, tuple[str, str]]):
            Named time-ranges as ``{name: (start_column, end_column)}``. Each
            becomes two columns, ``{name}_start_time`` and ``{name}_end_time``,
            copied from the named columns of the row. Default: none.
    '''
    closes_trial: str
    session_start: str | None = None
    start_buffer: float = 0.0
    fields: Callable[[pd.DataFrame, pd.Series], dict] = lambda window, closing_row: {}
    derived: Callable[[dict], dict] = lambda row: {}
    segments: dict[str, tuple[str, str]] = field(default_factory=dict)

#===============================================================================



################################################################################




################################################################################
# Shared Helpers
################################################################################



#===============================================================================
# 1| Rows Within a [start, end] Time Window
#===============================================================================
def within(
        events: pd.DataFrame,
        start: float,
        end: float,
) -> pd.DataFrame:
    '''
    Return the rows whose time index falls within ``[start, end]``.

    ----------
    Parameters:
        events (pd.DataFrame):
            Events frame with a numeric (Time) index.
        start (float):
            Inclusive lower bound of the window.
        end (float):
            Inclusive upper bound of the window.
    Returns:
        pd.DataFrame:
            The subset of ``events`` inside the window.
    '''
    # Both bounds inclusive: the start event and the closing event both belong to
    # the trial they delimit.
    return events[(events.index >= start) & (events.index <= end)]

#===============================================================================



#===============================================================================
# 2| First Event Matching a Pattern
#===============================================================================
def first_matching(
        window: pd.DataFrame,
        pattern: str,
        *,
        extract: Callable[[pd.Series], object] | None = None,
        default: object = np.nan,
        column: str = 'Event',
):
    '''
    Return the time of the first event whose text CONTAINS ``pattern``.

    The single "find the marker event for this trial" primitive: with no
    ``extract`` it answers WHEN something happened, and with one it answers WHAT
    that event said. ``pattern`` is matched literally (not as a regular
    expression), so event text full of brackets and colons can be searched for
    as-is.

    ----------
    Parameters:
        window (pd.DataFrame):
            The trial's events, restricted to its [start, end] window.
        pattern (str):
            Literal substring identifying the event of interest.
        extract (Callable[[pd.Series], object] | None):
            Applied to the first matching ROW to produce the return value. None
            (default) returns that row's time instead.
        default (object):
            Returned when no event matches. Default NaN.
        column (str):
            Column holding the event text. Default ``'Event'``.
    Returns:
        object:
            The first match's time, or ``extract(row)`` for it, or ``default``.
    '''
    matches = window[window[column].str.contains(pattern, na=False, regex=False)]
    if matches.empty:
        return default
    # index[0] is the time; iloc[0] is the row itself (whose .name is that time).
    if extract is None:
        return matches.index[0]
    return extract(matches.iloc[0])

#===============================================================================



#===============================================================================
# 3| Segment -> Column Pair Lookup
#===============================================================================
def segment_bounds(
        spec: TrialSpec,
) -> dict[str, tuple[str, str]]:
    '''
    Return ``{segment name: (start column, end column)}`` for a spec's trial table.

    The single source of truth for "which two columns bound segment X", so code
    that slices other streams by a trial's segments (pose, video, ...) reads the
    segment list off the spec instead of keeping its own copy. Every spec gets a
    ``'trial'`` entry for the whole start -> end span, which ``parse_trials``
    always emits; a spec may override it by declaring its own ``'trial'`` segment.

    ----------
    Parameters:
        spec (TrialSpec):
            The spec whose ``segments`` were used to build the trial table.
    Returns:
        dict[str, tuple[str, str]]:
            Segment name -> the pair of trial-table columns bounding it.
    '''
    bounds = {
        name: (f'{name}_start_time', f'{name}_end_time')
        for name in spec.segments
    }
    # The whole-trial window always exists, under the base delimiting columns.
    bounds.setdefault('trial', ('start_time', 'end_time'))
    return bounds

#===============================================================================



################################################################################




################################################################################
# Private Helper
################################################################################



#===============================================================================
# 1| Trial Start Time (the buffer lives here)
#===============================================================================
def _trial_start(
        events: pd.DataFrame,
        trial_index: int,
        previous_end: float | None,
        spec: TrialSpec,
) -> float:
    '''
    Return one trial's start time.

    The FIRST trial in a session starts at the spec's ``session_start`` marker
    (or at the session's first event when the spec has none). Every later trial
    starts at the previous trial's end plus ``spec.start_buffer``.

    ----------
    Parameters:
        events (pd.DataFrame):
            The session's events (used only to anchor the first trial).
        trial_index (int):
            Zero-based position of this trial within the session.
        previous_end (float | None):
            The previous trial's end time. None for the first trial.
        spec (TrialSpec):
            The spec supplying the start marker and the inter-trial buffer.
    Returns:
        float:
            The trial's start time.
    '''
    # Later trials: previous end plus the (spec-configured) buffer.
    if trial_index != 0:
        return previous_end + spec.start_buffer

    # First trial, no marker configured: anchor on the session's first event.
    if spec.session_start is None:
        return events.index[0]

    # First trial with a marker: it must actually be in the log. A truncated log
    # missing it cannot be delimited, so raise rather than guess a start time --
    # Catalog turns that into "no trial table for this session" with a warning.
    start = first_matching(events, spec.session_start, default=None)
    if start is None:
        raise ValueError(
            f'no {spec.session_start!r} event in the log: '
            f"cannot determine the first trial's start."
        )
    return start

#===============================================================================



################################################################################




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| parse_trials (Events Log -> Trial Table)
#===============================================================================
def parse_trials(
        events: pd.DataFrame,
        spec: TrialSpec,
) -> pd.DataFrame:
    '''
    Apply a ``TrialSpec`` to one session's event log, returning a trial table.

    Walks the closing events in time order. Each closes a trial; that trial's
    start is the previous closing event plus ``spec.start_buffer`` (or, for the
    first trial, the spec's ``session_start`` marker). For each trial it builds
    the set of events in the trial's ``[start, end]`` window, asks
    ``spec.fields`` for that trial's values, asks ``spec.derived`` for anything
    worked out from them, and expands ``spec.segments`` into start / end time
    columns. It knows nothing about any one experiment.

    ----------
    Parameters:
        events (pd.DataFrame):
            One session's events: a Time-indexed DataFrame with an ``Event``
            column, flattened to a single DataFrame.
        spec (TrialSpec):
            The experiment's trial description (see ``TrialSpec``).
    Returns:
        pd.DataFrame:
            One row per trial. Always carries ``trial_index`` (1-based),
            ``start_time`` and ``end_time``; the remaining columns are whatever
            ``spec.fields`` / ``spec.derived`` returned plus two per segment. An
            events log with no closing events yields an empty DataFrame.
    '''

    # 1| Collect the closing event of every trial, in time order. Each one ends
    #    exactly one trial, so these rows ARE the session's trials.
    closing_rows = events[events['Event'].str.startswith(spec.closes_trial, na=False)]

    # 2| Walk the closing events, reconstructing each trial's [start, end] window
    #    and reading its values out of that window.
    trials = []
    previous_end = None
    for i, (close_time, close_row) in enumerate(closing_rows.iterrows()):
        # 2a| Delimit the trial: end is this closing event, start comes from the
        #     previous end plus the buffer (or the session's start marker).
        start_time = _trial_start(events, i, previous_end, spec)
        window = within(events, start_time, close_time)

        # 2b| The delimiting columns every trial table has, then the
        #     experiment's own values read out of the window.
        row = {'trial_index': i + 1, 'start_time': start_time, 'end_time': close_time}
        row.update(spec.fields(window, close_row))

        # 2c| Values worked out FROM the row (durations and the like), which is
        #     why they run after fields rather than alongside them.
        row.update(spec.derived(row))

        # 2d| Named time-ranges become {name}_start_time / {name}_end_time pairs,
        #     copied from the columns the steps above produced.
        for name, (start_column, end_column) in spec.segments.items():
            row[f'{name}_start_time'] = row.get(start_column)
            row[f'{name}_end_time'] = row.get(end_column)

        trials.append(row)

        # 2e| This closing event becomes the anchor for the next trial's start.
        previous_end = close_time

    # 3| One row per trial (empty frame if the session had no closing events).
    return pd.DataFrame(trials)

#===============================================================================



################################################################################
