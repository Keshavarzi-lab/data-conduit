"""
Convert a session event log into a generic trial table: one row per trial.
----------------------------------------------------------------------------

Description:
    ``trials.py`` implements experiment-agnostic event-log -> trial-table
    processing. The module deliberately separates DATA describing what a trial is
    from the FUNCTION that applies that description:

        TrialSpec (experiment definition)
            + one session's events DataFrame
            -> parse_trials(...)
            -> one generic trial DataFrame.

    The parser assumes only a small structural model:
      * every trial ends with a closing event whose text starts with
        ``TrialSpec.closes_trial``;
      * later trials begin at the preceding trial's end plus ``start_buffer``;
      * the first trial begins either at ``session_start`` or the first event;
      * ``fields`` reads experiment-specific values from the events inside one
        trial window;
      * ``derived`` computes values from the row assembled so far;
      * ``segments`` exposes named time intervals as standard
        ``{segment}_start_time`` / ``{segment}_end_time`` columns.

    Nothing here knows Q_C-specific event strings, ports, outcomes, or scientific
    interpretation. Those belong in the experiment's TrialSpec/configuration.

Contents:
--------------------------------
- TrialSpec:       Data object describing one experiment's trial structure.
- within:          Select event rows inside an inclusive [start, end] window.
- first_matching:  Find the first event containing a literal pattern.
- segment_bounds:  Resolve segment names to trial-table bound columns.
- _trial_start:    Resolve first/subsequent trial start times.
- parse_trials:    Apply a TrialSpec to one session event log.
"""


################################################################################
# Imports
################################################################################

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

################################################################################


__all__ = [
    "TrialSpec",
    "parse_trials",
    "within",
    "first_matching",
    "segment_bounds",
]


################################################################################
# Trial Specification (Data Object)
################################################################################


# ===============================================================================
# 1| TrialSpec (Describe What One Experiment Means by a Trial)
# ===============================================================================


@dataclass
class TrialSpec:
    """
    Describe how one session's event log should be converted into trial rows.

    ``TrialSpec`` contains experiment-specific policy but no parsing behaviour of
    its own. ``parse_trials`` reads these fields and applies the generic workflow.
    This keeps task vocabulary/configuration outside the reusable parser.

    Parameters
    ----------
    closes_trial : str
        Prefix identifying events that close trials. Matching uses
        ``Series.str.startswith`` rather than substring matching, so each event
        whose text begins with this string is treated as one trial-ending row.
    session_start : str | None
        Literal substring identifying the first trial's start marker. If None,
        the first event in the session log becomes the first trial's start. If a
        string is supplied but never occurs, parsing raises rather than guessing a
        start for a truncated/incomplete event log.
    start_buffer : float
        Seconds added to the previous trial's end when defining the next trial's
        start. ``0.0`` makes consecutive trials contiguous at their boundary;
        positive values deliberately exclude an inter-trial buffer.
    fields : Callable[[pd.DataFrame, pd.Series], dict[str, Any]]
        Experiment-specific function called once per trial. It receives the
        events inside that trial's inclusive ``[start, end]`` window and the
        closing event row, then returns additional column values read directly
        from those events. Default returns no extra fields.
    derived : Callable[[dict[str, Any]], dict[str, Any]]
        Experiment-specific function called after ``fields``. It receives the row
        assembled so far and returns values computed FROM existing row values,
        such as durations or categorical outcomes. Keeping this separate from
        ``fields`` makes event extraction and derived calculations explicit.
        Default returns no extra values.
    segments : dict[str, tuple[str, str]]
        Mapping from segment name to two EXISTING row-column names representing
        its start and end. For example
        ``{'outbound': ('target_time', 'poke_time')}``. ``parse_trials`` copies
        those values into standard ``outbound_start_time`` /
        ``outbound_end_time`` columns for downstream generic slicing.
    """

    closes_trial: str  # Event-text prefix that defines one trial-ending event.
    session_start: str | None = None  # Optional first-trial start marker matched as a literal substring.
    start_buffer: float = 0.0  # Seconds added after previous close before next trial begins.
    fields: Callable[[pd.DataFrame, pd.Series], dict[str, Any]] = (  # Reads experiment-specific values directly from one trial window.
        lambda window, closing_row: {}
    )
    derived: Callable[[dict[str, Any]], dict[str, Any]] = (  # Computes additional values from the row assembled so far.
        lambda row: {}
    )
    segments: dict[str, tuple[str, str]] = field(default_factory=dict)  # Segment name -> source start/end column names.


# ===============================================================================


################################################################################
# Shared Helpers
################################################################################


# ===============================================================================
# 1| Select Event Rows Inside an Inclusive Time Window
# ===============================================================================


def within(
    events: pd.DataFrame,  # Time-indexed event rows to restrict.
    start: float,  # Inclusive lower time bound.
    end: float,  # Inclusive upper time bound.
) -> pd.DataFrame:  # Returns event rows whose index lies within [start, end].
    """
    Return event rows whose time index lies inside the inclusive ``[start, end]``.

    Parameters
    ----------
    events : pd.DataFrame
        Event table whose index contains numeric event times. The function does
        not require a specific event-text column because it only performs temporal
        slicing.
    start : float
        Inclusive lower time bound.
    end : float
        Inclusive upper time bound.

    Returns
    -------
    pd.DataFrame
        Subset of ``events`` whose index satisfies ``start <= time <= end``. The
        original index and columns are preserved.
    """

    if start > end:
        raise ValueError(f"start must be less than or equal to end; got {start!r} > {end!r}.")

    # === 1| Apply Both Inclusive Bounds to the Event-Time Index ===================
    # Inclusive bounds are deliberate: the marker defining the start and the
    # closing event defining the end both conceptually belong to the trial they
    # delimit and may need to be inspected by ``fields``.

    return events[(events.index >= start) & (events.index <= end)]  # One boolean expression preserves original row/index ordering.


# ===============================================================================


# ===============================================================================
# 2| Find the First Event Containing a Literal Text Pattern
# ===============================================================================


def first_matching(
    window: pd.DataFrame,  # Event rows in which to search.
    pattern: str,  # Literal substring identifying the event of interest.
    *,
    extract: Callable[[pd.Series], object] | None = None,  # Optional row->value extraction function.
    default: object = np.nan,  # Value returned when no event matches.
    column: str = "Event",  # Column containing event text.
) -> object:  # Returns first matching time/value or ``default``.
    """
    Return the first event matching a literal substring, optionally extracting a value.

    With ``extract=None`` the helper answers WHEN the event first occurred by
    returning the matching row's index value. With an ``extract`` callable it
    instead answers WHAT should be read from that first matching row. Literal
    matching (``regex=False``) is intentional because experiment event text often
    contains brackets, colons, or other characters that should not be interpreted
    as regular-expression syntax.

    Parameters
    ----------
    window : pd.DataFrame
        Event rows to search, typically one trial's inclusive time window. The
        DataFrame index is treated as event time when ``extract`` is None.
    pattern : str
        Literal substring that must occur in ``column`` for a row to match.
    extract : Callable[[pd.Series], object] | None
        Optional function applied to the FIRST matching row. Use this when the
        desired result is encoded elsewhere in the row rather than being its time.
        None returns the matching row's index value instead.
    default : object
        Value returned when no event matches. Defaults to ``np.nan`` so missing
        numeric trial fields naturally remain representable in pandas tables.
    column : str
        Name of the column containing event text. Default ``'Event'``.

    Returns
    -------
    object
        First match's index/time when ``extract`` is None, ``extract(first_row)``
        when a callable is supplied, or ``default`` when there are no matches.
    """

    # === 1| Find All Rows Containing the Literal Pattern ==========================

    matches = window[
        window[column].str.contains(
            pattern,  # Caller-provided event substring.
            na=False,  # Missing event strings are simply non-matches.
            regex=False,  # Event text is literal data, not regex syntax.
        )
    ]

    # === 2| Return the Caller-Specified Missing Value When Nothing Matches ========

    if matches.empty:
        return default

    # === 3| Return Either the Matching Time or an Extracted Row Value =============

    if extract is None:
        return matches.index[0]  # First row in preserved event order; its index is event time.

    return extract(matches.iloc[0])  # Apply caller's row->value rule only to the first matching event.


# ===============================================================================


# ===============================================================================
# 3| Resolve Every Trial Segment to Standard Start/End Column Names
# ===============================================================================


def segment_bounds(
    spec: TrialSpec,  # TrialSpec whose declared segments should be resolved.
) -> dict[str, tuple[str, str]]:  # Returns segment name -> emitted trial-table bound columns.
    """
    Return the standard trial-table start/end columns for each segment in a spec.

    ``TrialSpec.segments`` names the SOURCE row values used when constructing
    segments. After parsing, each segment is exposed under the STANDARD emitted
    columns ``{name}_start_time`` and ``{name}_end_time``. This helper describes
    that output schema to downstream generic code such as ``slicing.py``.

    The full trial is always available under the synthetic ``'trial'`` segment,
    mapped to base ``start_time`` / ``end_time``. If a spec explicitly defines a
    segment named ``'trial'``, that explicit emitted segment pair is retained.

    Parameters
    ----------
    spec : TrialSpec
        Trial specification whose segment declarations define the emitted segment
        columns.

    Returns
    -------
    dict[str, tuple[str, str]]
        Mapping from segment name to ``(start_column, end_column)`` in the parsed
        trial table.
    """

    # === 1| Convert Declared Segment Names to their Emitted Standard Columns ======

    bounds = {
        name: (f"{name}_start_time", f"{name}_end_time")  # parse_trials always emits this standard pair for each named segment.
        for name in spec.segments
    }

    # === 2| Guarantee a Whole-Trial Segment Exists ================================
    # ``setdefault`` deliberately does NOT overwrite a caller-declared ``trial``
    # segment. Ordinarily, however, it exposes the parser's universal base bounds.

    bounds.setdefault("trial", ("start_time", "end_time"))  # Full trial always has base delimiting columns.

    return bounds


# ===============================================================================


################################################################################
# Private Helper
################################################################################


# ===============================================================================
# 1| Resolve One Trial's Start Time
# ===============================================================================


def _trial_start(
    events: pd.DataFrame,  # Complete session event log used to anchor the first trial.
    trial_index: int,  # Zero-based position of the trial currently being parsed.
    previous_end: float | None,  # Previous trial's close time; None for the first trial.
    spec: TrialSpec,  # TrialSpec supplying first-start marker and inter-trial buffer.
) -> float:  # Returns start time for the current trial.
    """
    Resolve the current trial's start time from its position and TrialSpec.

    Later trials have a simple rule: previous trial end + ``start_buffer``. The
    first trial has no previous close and therefore uses either the configured
    ``session_start`` marker or the first event time in the log.

    Parameters
    ----------
    events : pd.DataFrame
        Complete session event log. Only the first trial needs to inspect it for
        a first-event time or configured ``session_start`` marker.
    trial_index : int
        Zero-based trial position. ``0`` means first trial; any non-zero value
        follows the previous-end + buffer rule.
    previous_end : float | None
        Previous trial's closing time. Must contain a numeric value for later
        trials; None is expected only when ``trial_index == 0``.
    spec : TrialSpec
        Trial specification containing ``session_start`` and ``start_buffer``.

    Returns
    -------
    float
        Resolved start time for the current trial.
    """

    # === 1| Later Trials Start after the Previous Close + Configured Buffer =======

    if trial_index != 0:
        return previous_end + spec.start_buffer  # Parser controls previous_end, so later trials always have a close anchor.

    # === 2| First Trial Without an Explicit Marker Starts at First Logged Event ===

    if spec.session_start is None:
        return events.index[0]  # No invented pre-session time; anchor on first actual event.

    # === 3| First Trial with a Marker Must Find that Marker =======================
    # Guessing from the first event when an explicit marker was requested would
    # silently accept truncated logs and produce incorrectly delimited trials.

    start = first_matching(
        events,
        spec.session_start,  # Literal substring identifying first-trial start marker.
        default=None,  # None distinguishes "marker missing" from any numeric time value.
    )

    if start is None:
        raise ValueError(f"no {spec.session_start!r} event in the log: cannot determine the first trial's start.")

    return start


# ===============================================================================


################################################################################
# Public API
################################################################################


# ===============================================================================
# 1| Parse One Session Event Log into One Row per Trial
# ===============================================================================


def parse_trials(
    events: pd.DataFrame,  # One session's Time-indexed event log with an Event text column.
    spec: TrialSpec,  # Experiment-specific trial description applied to this log.
) -> pd.DataFrame:  # Returns generic trial table, one row per closing event/trial.
    """
    Apply ``spec`` to one session's event log and return a trial table.

    Every event whose text starts with ``spec.closes_trial`` defines exactly one
    trial end. Walking those closing rows in time order therefore provides the
    trial sequence. For each trial, the parser reconstructs its inclusive
    ``[start, end]`` event window, asks the spec for extracted fields and derived
    values, expands named segments into standard time-bound columns, and appends
    one completed row.

    Parameters
    ----------
    events : pd.DataFrame
        One session's event table. The DataFrame must have a time-like numeric
        index and an ``'Event'`` text column. Rows are assumed to be in the event
        order represented by that index; the parser does not reorder them.
    spec : TrialSpec
        Experiment-specific description of how trials close/start, which values
        should be extracted from each trial window, which derived values should be
        calculated, and which named segments should be emitted.

    Returns
    -------
    pd.DataFrame
        One row per closing event/trial. Every non-empty output contains
        ``trial_index`` (1-based), ``start_time``, and ``end_time``, plus fields
        returned by ``spec.fields``, values returned by ``spec.derived``, and two
        standard columns per declared segment. If no closing events are present,
        an empty DataFrame is returned.
    """

    # === 1| Validate the Event Stream and Trial Policy =============================

    if not isinstance(events, pd.DataFrame):
        raise TypeError(f"events must be a pandas DataFrame, got {type(events).__name__}.")
    if not isinstance(spec, TrialSpec):
        raise TypeError(f"spec must be a TrialSpec, got {type(spec).__name__}.")
    if "Event" not in events.columns:
        raise KeyError("events must contain an 'Event' column.")
    if not pd.api.types.is_numeric_dtype(events.index.dtype):
        raise TypeError("events index must contain numeric event times.")
    event_times = events.index.to_numpy(dtype=float)
    if not np.isfinite(event_times).all():
        raise ValueError("events index must contain only finite event times.")
    if not events.index.is_monotonic_increasing:
        raise ValueError("events must be ordered by a monotonically non-decreasing time index.")
    if not isinstance(spec.start_buffer, Real) or not np.isfinite(spec.start_buffer) or spec.start_buffer < 0:
        raise ValueError("TrialSpec.start_buffer must be a finite, non-negative number.")

    protected_columns = {
        "trial_index",
        "start_time",
        "end_time",
        *(f"{name}_{bound}_time" for name in spec.segments for bound in ("start", "end")),
    }

    # === 2| Identify the Closing Event that Defines Each Trial ====================
    # Matching uses ``startswith`` because the spec declares a closing-event
    # prefix. Each matching row ends one trial, so the resulting row sequence is
    # also the sequence of trials to reconstruct.

    closing_rows = events[
        events["Event"].str.startswith(
            spec.closes_trial,  # Experiment-specific closing-event prefix.
            na=False,  # Missing event text cannot close a trial.
        )
    ]

    # === 3| Walk Closing Events and Build One Trial Row for Each ==================

    trials: list[dict[str, Any]] = []  # Accumulate plain row dictionaries before one final DataFrame construction.
    previous_end: float | None = None  # First trial has no preceding close anchor.

    for i, (close_time, close_row) in enumerate(closing_rows.iterrows()):
        # === 2.1| Resolve the Trial's Inclusive [Start, End] Bounds ================

        start_time = _trial_start(
            events,  # Complete session log needed only for first-trial anchoring.
            i,  # Zero-based loop index tells helper whether this is the first trial.
            previous_end,  # Previous closing event anchors later trial starts.
            spec,  # Supplies first marker and inter-trial buffer policy.
        )
        if start_time > close_time:
            raise ValueError(f"trial {i + 1} has start_time {start_time!r} after its closing event at {close_time!r}; check event ordering and TrialSpec.start_buffer.")

        # With a zero buffer the next trial's numeric start equals the previous
        # close. The previous closing event belongs only to the trial it closed,
        # so exclude that shared boundary row from the next field-extraction
        # window while retaining the contiguous start_time value in the table.
        window = events[(events.index > start_time) & (events.index <= close_time)] if i and start_time == previous_end else within(events, start_time, close_time)

        # === 2.2| Create the Universal Base Trial Columns =========================

        row: dict[str, Any] = {
            "trial_index": i + 1,  # User-facing trial index is 1-based even though loop position is 0-based.
            "start_time": start_time,  # Resolved inclusive lower bound.
            "end_time": close_time,  # Current closing event's time is the inclusive upper bound.
        }

        # === 2.3| Read Experiment-Specific Fields from the Trial Window ============
        # ``fields`` receives both the restricted event window and the closing row
        # because some tasks encode useful trial values directly in the close event.

        field_values = spec.fields(window, close_row)
        if not isinstance(field_values, dict):
            raise TypeError(f"TrialSpec.fields must return a dict, got {type(field_values).__name__} for trial {i + 1}.")
        collisions = protected_columns.intersection(field_values)
        if collisions:
            raise ValueError(f"TrialSpec.fields cannot overwrite reserved trial columns: {sorted(collisions)}.")
        row.update(field_values)  # Extracted values join the same plain row dictionary.

        # === 2.4| Compute Values Derived from the Row Assembled So Far =============
        # Derived calculations run after field extraction so they can depend on
        # both universal delimiting times and experiment-specific extracted values.

        derived_values = spec.derived(row)
        if not isinstance(derived_values, dict):
            raise TypeError(f"TrialSpec.derived must return a dict, got {type(derived_values).__name__} for trial {i + 1}.")
        collisions = set(row).intersection(derived_values) | protected_columns.intersection(derived_values)
        if collisions:
            raise ValueError(f"TrialSpec.derived cannot overwrite existing or reserved trial columns: {sorted(collisions)}.")
        row.update(derived_values)  # Durations/outcomes/etc. can depend on fields already present.

        # === 2.5| Expand Named Segments into Standard Start/End Output Columns =====
        # Segment definitions refer to values already present in the row. ``get``
        # intentionally yields None for a missing source column rather than
        # raising here; the resulting missing boundary remains visible in output
        # and naturally produces an empty slice in downstream comparisons.

        for name, (start_column, end_column) in spec.segments.items():
            missing = [column for column in (start_column, end_column) if column not in row]
            if missing:
                raise KeyError(f"segment {name!r} references missing trial columns: {missing}.")

            segment_start = row[start_column]
            segment_end = row[end_column]
            if pd.notna(segment_start) and pd.notna(segment_end) and segment_start > segment_end:
                raise ValueError(f"segment {name!r} has start {segment_start!r} after end {segment_end!r} for trial {i + 1}.")
            row[f"{name}_start_time"] = segment_start  # Copy source value to the standard generic segment-start column.
            row[f"{name}_end_time"] = segment_end  # Copy source value to the standard generic segment-end column.

        # === 2.6| Store the Completed Trial Row ===================================

        trials.append(row)

        # === 2.7| Make this Close Event the Anchor for the Next Trial ==============

        previous_end = close_time  # Next iteration applies ``previous_end + start_buffer``.

    # === 4| Convert Row Dictionaries into the Final Trial DataFrame ===============
    # Construct once after the loop rather than repeatedly concatenating rows,
    # which is both clearer and substantially cheaper for many trials.

    return pd.DataFrame(trials)


# ===============================================================================
#