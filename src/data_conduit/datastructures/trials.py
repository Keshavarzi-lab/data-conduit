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

    The parser supports two ways to choose trial starts:
      * ``trial_start=None`` retains the original previous-end + buffer model;
      * a ``trial_start`` rule opens each trial independently, allowing gaps;
      * ``closes_trial`` closes an open trial; ``session_start`` can override its
        first start and ``last_trial_end`` can close its final unfinished trial;
      * start/end inclusion and insufficient inter-trial separation are explicit
        policies, with the original defaults retained;
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
- _matching_positions: Resolve a boundary rule to matching event-row positions.
- parse_trials:    Apply a TrialSpec or direct options to one session event log.
"""


################################################################################
# Imports
################################################################################

from collections.abc import Callable
from dataclasses import dataclass, field
import warnings
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
    closes_trial : str | Callable | None
        Required closing rule. A string retains the original prefix matching:
        ``'Poke:'`` matches event text beginning with ``'Poke:'``. A callable
        receives the complete event DataFrame and returns one Boolean per row.
        With implicit starts every eligible close produces a trial; with explicit
        starts a close is used only when a trial is open. None raises on parsing.
    session_start : str | Callable | None
        Optional FIRST-trial start override; a string matches a literal substring,
        and a callable returns one Boolean per event row. The first matching row
        becomes the first start, and earlier events are ignored. With explicit
        starts this replaces at most one ordinary start before the first close;
        two ordinary starts before that close still raise. If None, implicit
        trials begin at the first event and explicit trials at their first start
        match. A requested marker that is absent raises for a non-empty log.
    start_buffer : float
        Non-negative separation in the event index's units (normally seconds).
        With ``trial_start=None``, later starts are previous end + this buffer.
        With explicit starts it is the minimum permitted gap after the preceding
        close; ``buffer_effect`` handles smaller gaps. No buffer precedes the
        first trial. Default 0.0 retains the original contiguous-trial behavior.
    fields : Callable[[pd.DataFrame, pd.Series], dict[str, Any]]
        Experiment-specific function called once per trial. It receives the
        events inside that trial's selected start/end boundaries and the
        closing event row, then returns additional column values read directly
        from those events. It must not modify either supplied input. Default
        returns no extra fields.
    derived : Callable[[dict[str, Any]], dict[str, Any]]
        Experiment-specific function called after ``fields``. It receives a
        shallow copy of the row assembled so far and returns values computed
        FROM existing row values, such as durations or categorical outcomes.
        Keeping this separate from ``fields`` makes event extraction and derived
        calculations explicit.
        Only returned values are added to the output; assigning keys on the copy
        does not change trial boundaries. Do not modify mutable values inside
        that copy. Default returns no extra values.
    segments : dict[str, tuple[str, str]]
        Mapping from segment name to two EXISTING row-column names representing
        its start and end. For example
        ``{'outbound': ('target_time', 'poke_time')}``. ``parse_trials`` copies
        those values into standard ``outbound_start_time`` /
        ``outbound_end_time`` columns for downstream generic slicing. Segment
        inclusion flags inherit whole-trial flags when their source is
        ``start_time`` or ``end_time``; internal source boundaries are inclusive.
    trial_start : str | Callable | None
        Ordinary explicit-start rule. Strings match a literal Event substring;
        callables receive the complete event table. None retains the previous-end
        model. Explicit starts pair with the next close at or after that row.
        Another start while a trial is open raises; an unfinished final start is
        warned about and omitted unless ``last_trial_end`` supplies a close.
    last_trial_end : str | Callable | None
        Optional final-close override. Uses the LAST matching row, excludes all
        later events, and closes any trial still open there. It does not extend
        an explicit trial that already closed. In the implicit model another
        trial begins after every ordinary close, so a later final marker closes
        that remaining interval. A requested absent marker raises.
    buffer_effect : {'warn', 'raise', 'skip', 'force'}
        Policy for insufficient separation. Default 'raise' retains the original
        refusal of an impossible previous-end + buffer interval. For explicit
        starts, 'warn' retains the observed start, 'skip' consumes the close but
        emits no row, and 'force' moves the start to previous end + buffer. If the
        moved start would exceed its close, 'force' raises. For implicit starts
        beyond the close, 'warn' falls back to the preceding close, 'skip' omits
        the candidate, and 'raise'/'force' raise. Equality with the close is valid.
    start_inclusive : bool | None
        Whether the starting boundary belongs to field extraction. None retains
        the original default: include the first start and buffered starts, but
        exclude a preceding close reused as the next start. Explicit starts are
        included by default. True/False override this automatic choice for every
        trial. Exclusion of an event boundary removes that ROW, not all other
        events tied at its timestamp; synthetic buffered boundaries use time.
    end_inclusive : bool
        Whether the closing boundary row belongs to field extraction. Default
        True. The closing row is still passed separately to ``fields`` when
        False, allowing outcomes encoded in the close to be extracted.
    """

    closes_trial: str | Callable | None = None                                # Event prefix or Boolean callable identifying trial-ending rows.
    session_start: str | Callable | None = None                               # Optional first-trial start marker matched as a literal substring.
    start_buffer: float = 0.0                                                 # Minimum gap after the previous close, in event-time units.
    fields: Callable[[pd.DataFrame, pd.Series], dict[str, Any]] = (           # Reads experiment-specific values directly from one trial window.
        lambda window, closing_row: {}
    )

    derived: Callable[[dict[str, Any]], dict[str, Any]] = (                   # Computes additional values from the row assembled so far.
        lambda row: {}
    )

    segments: dict[str, tuple[str, str]] = field(default_factory=dict)        # Segment name -> source start/end column names.

    trial_start: str | Callable | None = None                                 # None reuses the previous close; a rule opens separate trials.
    last_trial_end: str | Callable | None = None                              # Last match caps the recording and closes a remaining open trial.
    buffer_effect: str = "raise"                                              # Determines what happens when the requested separation cannot fit.
    start_inclusive: bool | None = None                                       # None includes new starts but excludes a reused previous closing row.
    end_inclusive: bool = True                                                # Include closing row in fields; closing callback argument always exists.


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
# 1| Resolve a Boundary Rule to Original Event-Row Positions
# ===============================================================================


def _matching_positions(
    events: pd.DataFrame,                             # Complete recording log, with its original row order intact.
    rule: str | Callable,                             # Literal event text or full-table Boolean condition.
    *,
    prefix: bool = False,                             # True keeps the original closes_trial prefix convention.
) -> np.ndarray:                                      # Returns increasing zero-based positions of matching rows.
    """
    Evaluate one boundary rule without sorting or changing the event table.

    Parameters
    ----------
    events : pd.DataFrame
        Complete recording's event table. String rules require an ``Event``
        column; callable rules may use any columns in the supplied table.
    rule : str | Callable
        A string matches literal text; a callable returns a one-dimensional
        Boolean mask in the same row order as ``events``. A returned Series must
        have exactly the same index, including the ordering of tied timestamps.
        Missing values in a nullable Boolean Series count as non-matches.
        Callables must read the supplied event table without modifying it.
    prefix : bool
        False uses literal substring matching, as the original session_start
        option did. True uses prefix matching, as original closes_trial did.
        It has no effect on callable rules.

    Returns
    -------
    np.ndarray
        Integer positions of matches, in original row order. Positions identify
        separate events even when their timestamp labels are identical.
    """

    # === 1| Match Literal Event Text or Evaluate the Caller's Boolean Rule ========

    if isinstance(rule, str):                          # Text rules deliberately share the original Event column convention.
        if "Event" not in events.columns:
            raise KeyError("string boundary rules require an 'Event' column.")

        if prefix:
            mask = events["Event"].str.startswith(rule, na=False)
        else:
            mask = events["Event"].str.contains(rule, na=False, regex=False)

    elif callable(rule):                              # Caller can use &, | and ~ to combine conditions across columns.
        mask = rule(events)

    else:
        raise TypeError("boundary rules must be literal strings or Boolean-mask callables.")

    # === 2| Reject Misaligned or Non-Boolean Results Before Using Row Positions ===

    if isinstance(mask, pd.Series):
        if not mask.index.equals(events.index):       # Reindexing a mask would be ambiguous when event times are repeated.
            raise ValueError("boundary mask Series must have the same index and order as events.")

        if not pd.api.types.is_bool_dtype(mask.dtype):
            raise TypeError("boundary callables must return Boolean values.")

        mask = mask.fillna(False).to_numpy(dtype=bool) # Missing Boolean matches cannot open or close a trial.

    mask = np.asarray(mask)
    if mask.ndim != 1 or len(mask) != len(events):     # One decision is required for every original event row.
        raise ValueError("boundary mask must be one-dimensional with one value per event row.")

    if mask.dtype != bool:
        raise TypeError("boundary callables must return Boolean values.")

    # === 3| Return Row Positions Without Changing Equal-Time Event Order =========

    return np.flatnonzero(mask)                       # Position, rather than timestamp, identifies each boundary event.


# ===============================================================================


################################################################################
# Public API
################################################################################


# ===============================================================================
# 1| Parse One Session Event Log into One Row per Trial
# ===============================================================================


def parse_trials(
    events: pd.DataFrame,                                            # One recording's numeric-time-indexed event log, already in event order.
    spec: TrialSpec | None = None,                                   # Existing experiment description; alternatively supply direct options below.
    *,
    first_trial_start: str | Callable | None = None,                 # First matching row overrides only the first trial's start.
    last_trial_end: str | Callable | None = None,                    # Last matching row caps the log and closes a remaining open trial.
    trial_start: str | Callable | None = None,                       # None reuses previous end + buffer; a rule identifies independent starts.
    trial_end: str | Callable | None = None,                         # Event-text prefix or Boolean callable identifying closing rows.
    buffer: float = 0.0,                                             # Minimum separation, in the same time units as the event index.
    buffer_effect: str = "raise",                                    # warn / raise / skip / force controls insufficient separation.
    start_inclusive: bool | None = None,                             # None retains first/new-start inclusion and shared-close exclusion.
    end_inclusive: bool = True,                                      # Include the closing row in the event window passed to field extraction.
    event_fields: Callable | None = None,                            # Receives selected events and the closing row; returns extracted columns.
    derived_values: Callable | None = None,                          # Receives the assembled trial row; returns additional calculated columns.
) -> pd.DataFrame:                                                   # Returns one row per completed, retained trial in original event order.
    """
    Apply a TrialSpec or direct boundary/extraction options to one event log.

    The original previous-end + buffer model remains the default. Supplying an
    explicit ``trial_start`` instead pairs each start with the next eligible
    close, allowing events and gaps outside trial windows. For each completed
    candidate, the parser selects its event rows, calls the field extractor and
    derived-value function, expands named segments, and appends one trial row.

    Pass either an existing ``spec`` or direct options. Direct names correspond
    to existing spec fields: first_trial_start -> session_start, trial_end ->
    closes_trial, buffer -> start_buffer, event_fields -> fields, and
    derived_values -> derived. This translation reuses TrialSpec; it does not
    introduce a second configuration class. Named segments are configured on a
    TrialSpec, as before. Non-default direct options together with spec raise.

    Parameters
    ----------
    events : pd.DataFrame
        One recording's event table, with a finite numeric, monotonically
        non-decreasing time index. The parser does NOT sort or mutate this table;
        tied timestamps retain their input row order. String boundary rules
        require an ``Event`` column; callable rules can inspect any columns.
        An optional ``event_index`` column contains stable, unique, increasing
        integer IDs assigned to the original recording before any filtering.
        Parse the complete log before filtering events for analysis. These IDs
        allow later slicing of the same row set or a further subset, even when
        timestamps tie. Only the first/last included IDs are stored: if rows were
        removed before parsing, slicing a fuller log can restore those omitted
        rows. IDs are not generated here. Boundary and field callbacks must not
        modify their supplied event tables or closing rows.
    spec : TrialSpec | None
        Existing description of starts, closes, event extraction, derived
        values and named segments. The original six fields and positional order
        remain supported. With None, the direct options create that same spec.
    first_trial_start : str | Callable | None
        Direct equivalent of spec.session_start. A string matches a literal
        Event substring; a callable returns one Boolean per event row. Uses the
        FIRST match and ignores earlier events. For explicit trials it replaces
        at most one ordinary start before the first eligible close. If None,
        the first event anchors implicit trials, whereas explicit trials await
        their first ordinary start. A requested absent marker raises.
    last_trial_end : str | Callable | None
        Same rule forms as first_trial_start, using the LAST match. The parser
        ignores later events and uses this row to close any remaining open
        trial. A completed explicit trial is not extended. Implicit trials
        start again after every close, so a later last marker closes their final
        interval. Missing requested markers or a final marker before the first
        special start raise. None uses ordinary closing events only.
    trial_start : str | Callable | None
        Literal Event substring or full-table Boolean callable identifying each
        ordinary start. None uses previous close + buffer after the first trial.
        An explicit start pairs with the next closing row at or after it; a
        second start before that close raises. Unmatched closing rows outside
        explicit trials are ignored. An unclosed final start warns and is omitted.
    trial_end : str | Callable | None
        Required when spec is None. A string uses PREFIX matching, preserving
        closes_trial behavior; a callable returns a Boolean mask of closing rows.
        For exact equality, use ``lambda log: log['Event'].eq('end')``. Combine
        arbitrary pandas conditions with &, | and ~ inside the callable.
    buffer : float
        Non-negative minimum separation, in event-index units. Default 0.0.
        Implicit starts are previous end + buffer. Explicit starts are checked
        against that minimum, with buffer_effect determining any conflict.
        No buffer is applied before the first trial.
    buffer_effect : {'warn', 'raise', 'skip', 'force'}
        Default 'raise'. For an explicit start below previous end + buffer:
        'warn' retains the observed start; 'raise' stops; 'skip' consumes that
        trial's close without emitting its row; 'force' moves its start forward.
        For an implicit buffered start beyond its close: 'warn' instead uses the
        previous close, 'skip' omits the row, and 'raise'/'force' stop. A forced
        start beyond the close always raises; equality is allowed. A skipped
        trial's close still anchors the following trial, and numbering has a gap.
    start_inclusive : bool | None
        None preserves automatic defaults: first/explicit/buffered starts are
        included, but a previous closing row reused as the next start is not.
        True or False includes/excludes every starting boundary. A boundary
        event is distinguished by ROW POSITION, so excluding it does not discard
        later events tied at the same time. A synthetic buffered boundary uses
        the numeric time comparison instead. Default None.
    end_inclusive : bool
        Include the closing boundary row in extraction when True (default).
        Rows after that boundary are excluded even if their timestamps tie.
        The separate closing-row callback argument is supplied in either case.
    event_fields : Callable | None
        Called with selected events and the closing row; returns extracted
        columns. None supplies no extra fields. It cannot overwrite base trial,
        exact-event, or generated segment columns. Use first_matching inside
        this callback for optional event values, as existing specifications do.
    derived_values : Callable | None
        Called after event_fields with a shallow copy of the assembled trial row;
        returns additional calculated columns. Only returned values are added.
        Do not modify mutable values held inside the copy. None adds nothing.
        Existing or reserved column names cannot be overwritten. The original
        callable convention is retained.

    Returns
    -------
    pd.DataFrame
        One row per completed, retained trial. Base columns are trial_index
        (1-based candidate number), start_time, end_time, start_inclusive and
        end_inclusive. Extraction, derived and declared segment columns follow.
        Segment flags inherit whole-trial flags for corresponding outer source
        columns; internal segment boundaries remain inclusive.

        If events supplies event_index, first_event_index and last_event_index
        record the first and last INCLUDED event IDs, both inclusive. An empty
        extraction window has pd.NA for both. These IDs describe whole trials;
        internal named-segment boundaries remain time-based. Session identity is
        attached by the surrounding DataStructure workflow, not inferred here.
        If no candidates are retained, an empty DataFrame is returned, as before.
    """

    # === 1| Resolve Direct Options onto the Existing TrialSpec ====================
    # Existing parse_trials(events, spec) callers retain their configuration.
    # Direct arguments describe the same fields; mixing both forms would obscure
    # which definition wins, so require one source of trial policy.

    if spec is None:
        spec = TrialSpec(
            closes_trial=trial_end,                                   # Direct end rule uses the existing prefix/callable closing field.
            session_start=first_trial_start,                          # Original field already represented the special first start.
            start_buffer=buffer,                                     # Original buffer field remains the sole stored separation value.
            trial_start=trial_start,
            last_trial_end=last_trial_end,
            buffer_effect=buffer_effect,
            start_inclusive=start_inclusive,
            end_inclusive=end_inclusive,
        )

        if event_fields is not None:
            spec.fields = event_fields                               # Keep the original window-plus-closing-row callback convention.

        if derived_values is not None:
            spec.derived = derived_values                            # Calculations still receive the row after extracted fields were added.

    else:
        if not isinstance(spec, TrialSpec):
            raise TypeError(f"spec must be a TrialSpec, got {type(spec).__name__}.")

        direct_options = (
            first_trial_start is not None, last_trial_end is not None,
            trial_start is not None, trial_end is not None,
            buffer != 0.0, buffer_effect != "raise",
            start_inclusive is not None, end_inclusive is not True,
            event_fields is not None, derived_values is not None,
        )
        if any(direct_options):                                      # Prevent supplied direct arguments being silently ignored.
            raise ValueError("pass either spec or direct trial options, not both.")

    # === 2| Validate Event Order, Optional Event IDs and Trial Policy =============

    if not isinstance(events, pd.DataFrame):
        raise TypeError(f"events must be a pandas DataFrame, got {type(events).__name__}.")

    if not pd.api.types.is_numeric_dtype(events.index.dtype):
        raise TypeError("events index must contain numeric event times.")

    event_times = events.index.to_numpy(dtype=float)                   # Numeric comparisons use the original time unit, without conversion.
    if not np.isfinite(event_times).all():
        raise ValueError("events index must contain only finite event times.")

    if not events.index.is_monotonic_increasing:                       # Reordering here could hide an upstream event-order error.
        raise ValueError("events must be ordered by a monotonically non-decreasing time index.")

    if not isinstance(spec.start_buffer, Real) or not np.isfinite(spec.start_buffer) or spec.start_buffer < 0:
        raise ValueError("TrialSpec.start_buffer must be a finite, non-negative number.")

    if spec.closes_trial is None:
        raise ValueError("supply trial_end directly or TrialSpec.closes_trial.")

    if spec.buffer_effect not in ("warn", "raise", "skip", "force"):
        raise ValueError("buffer_effect must be 'warn', 'raise', 'skip', or 'force'.")

    if spec.start_inclusive is not None and not isinstance(spec.start_inclusive, (bool, np.bool_)):
        raise TypeError("start_inclusive must be True, False, or None.")

    if not isinstance(spec.end_inclusive, (bool, np.bool_)):
        raise TypeError("end_inclusive must be True or False.")

    if not callable(spec.fields) or not callable(spec.derived):        # Both callbacks retain the original callable-only contract.
        raise TypeError("event_fields/fields and derived_values/derived must be callable.")

    has_event_ids = "event_index" in events.columns
    if has_event_ids:
        event_ids = events["event_index"]                             # IDs must have been attached to the original recording before selection.
        if not pd.api.types.is_integer_dtype(event_ids.dtype) or event_ids.isna().any():
            raise TypeError("event_index must contain non-missing integer event IDs.")

        if not event_ids.is_unique or not event_ids.is_monotonic_increasing:
            raise ValueError("event_index must contain unique, increasing event IDs.")

    protected_columns = {
        "trial_index", "start_time", "end_time",
        "start_inclusive", "end_inclusive",
        "first_event_index", "last_event_index",
    }
    for name in spec.segments:                                       # Reserve each emitted segment column before field callbacks run.
        for bound in ("start", "end"):
            protected_columns.add(f"{name}_{bound}_time")
            protected_columns.add(f"{name}_{bound}_inclusive")

    if events.empty:
        return pd.DataFrame()                                        # Empty recordings contain no trial candidates or boundary rows to inspect.

    # === 3| Locate Ordinary Closings and Optional Recording-End Override =========
    # Positions distinguish different events whose times tie. The final override
    # caps ordinary closes, then supplies one extra closing row if it is not
    # already an ordinary close. Explicit trials only use it when still open.

    closing_positions = _matching_positions(events, spec.closes_trial, prefix=True)
    last_position = len(events) - 1
    if spec.last_trial_end is not None:
        last_matches = _matching_positions(events, spec.last_trial_end)
        if len(last_matches) == 0:
            raise ValueError("last_trial_end did not match an event in the log.")

        last_position = int(last_matches[-1])                         # The final matching row is the requested end of the parsed recording.
        closing_positions = closing_positions[closing_positions <= last_position]
        if last_position not in closing_positions:
            closing_positions = np.append(closing_positions, last_position)

    # === 4| Locate the First Anchor and Any Independent Ordinary Starts ===========

    first_position = 0
    if spec.session_start is not None:
        first_matches = _matching_positions(events, spec.session_start)
        if len(first_matches) == 0:
            raise ValueError("first_trial_start/session_start did not match an event in the log.")

        first_position = int(first_matches[0])                       # Ignore any preamble preceding the requested first-trial marker.
        if first_position > last_position:
            raise ValueError("first_trial_start occurs after last_trial_end.")

    closing_positions = closing_positions[closing_positions >= first_position]
    explicit_starts = spec.trial_start is not None
    start_positions = np.empty(0, dtype=int)
    if explicit_starts:
        start_positions = _matching_positions(events, spec.trial_start)
        start_positions = start_positions[
            (start_positions >= first_position) & (start_positions <= last_position)
        ]

        if spec.session_start is not None:
            # The special marker replaces the ordinary first start rather than
            # creating two starts for that trial. Drop at most one ordinary start
            # before the first close; a second ordinary start still raises below.
            first_close = closing_positions[0] if len(closing_positions) else last_position
            if len(start_positions) and start_positions[0] <= first_close:
                start_positions = start_positions[1:]

            start_positions = np.insert(start_positions, 0, first_position)

    # === 5| Walk Closing Events and Build One Row per Completed Trial =============

    trials: list[dict[str, Any]] = []                                # Accumulate ordinary row dictionaries for one final DataFrame construction.
    previous_end: float | None = None                               # No preceding close exists until the first candidate has been consumed.
    previous_close_position: int | None = None                      # Keep the exact preceding row when its timestamp is shared by other events.
    start_cursor = 0                                               # Next unmatched explicit start in the ordered start-position array.
    candidate_index = 0                                            # Skipped candidates still consume a 1-based trial number.

    for close_position in closing_positions:
        close_position = int(close_position)
        close_time = event_times[close_position]
        close_row = events.iloc[close_position]                     # Pass the actual closing row separately even when end_inclusive is False.

        # === 5.1| Pair an Explicit Start or Reuse the Previous Closing Boundary ===

        if explicit_starts:
            next_cursor = int(np.searchsorted(start_positions, close_position, side="right"))
            unmatched_count = next_cursor - start_cursor            # Count starts since the preceding matched close, including a start on this row.
            if unmatched_count == 0:
                continue                                            # No trial is open; an unmatched closing event outside trials is ignored.

            if unmatched_count > 1:
                raise ValueError(f"multiple trial_start events before closing row {close_position}; an open trial must close before another starts.")

            start_position = int(start_positions[start_cursor])
            start_time = event_times[start_position]
            start_cursor = next_cursor                              # This closing event consumes the paired start, even if buffer policy skips it.

        elif previous_end is None:
            start_position = first_position
            start_time = event_times[first_position]                 # First implicit trial uses the special marker or the first logged event.

        else:
            start_time = previous_end + spec.start_buffer
            start_position = previous_close_position if spec.start_buffer == 0 else None

        candidate_index += 1

        # === 5.2| Apply the Minimum Separation without Inventing an Invalid Trial ==
        # Explicit starts can occur before the requested minimum. Implicit starts
        # already equal that minimum but may then fall after their closing event.
        # 'warn' keeps observed explicit starts, or removes an impossible implicit
        # buffer. 'skip' still consumes this close as the next trial's anchor.

        if previous_end is not None:
            earliest_start = previous_end + spec.start_buffer
            insufficient_gap = explicit_starts and start_time < earliest_start
            impossible_buffer = not explicit_starts and start_time > close_time

            if insufficient_gap or impossible_buffer:
                message = (
                    f"trial {candidate_index}: start {start_time} and close {close_time} "
                    f"cannot meet buffer {spec.start_buffer} after previous close {previous_end}."
                )
                if spec.buffer_effect == "raise":
                    raise ValueError(message)

                if spec.buffer_effect == "skip":
                    previous_end = close_time                        # The following candidate starts/checks its buffer after this consumed close.
                    previous_close_position = close_position
                    continue

                if spec.buffer_effect == "warn":
                    warnings.warn(message, UserWarning, stacklevel=2)
                    if impossible_buffer:
                        start_time = previous_end                    # Keep a valid implicit interval by removing the buffer that could not fit.
                        start_position = previous_close_position

                if spec.buffer_effect == "force":
                    start_time = earliest_start                      # A forced explicit start becomes a numeric boundary, not an invented event.
                    start_position = None

        if start_time > close_time:
            raise ValueError(f"trial {candidate_index} has start_time {start_time!r} after its closing event at {close_time!r}; the requested buffer cannot fit.")

        # === 5.3| Select Exact Event Rows with Independent Start/End Inclusion =====
        # An actual boundary row is included/excluded by position. That preserves
        # later same-time events after an excluded start and prevents same-time
        # events after the close leaking backwards into this trial. Only buffered
        # starts without a boundary event require a numeric timestamp comparison.

        include_start = spec.start_inclusive
        if include_start is None:
            reused_close = previous_close_position is not None and start_position == previous_close_position
            include_start = not reused_close                         # Retain original first/buffered inclusion and shared-close exclusion.

        if start_position is None:
            side = "left" if include_start else "right"
            window_start = int(np.searchsorted(event_times, start_time, side=side))
        else:
            window_start = start_position
            if not include_start:
                window_start += 1                                   # Begin after the excluded starting event.

        window_stop = close_position
        if spec.end_inclusive:
            window_stop += 1                                        # iloc excludes its stop, so advance it past the closing event.

        window = events.iloc[window_start:window_stop]               # iloc preserves the input order and naturally returns empty for crossed bounds.

        # === 5.4| Create Universal Bounds and Optional Exact-Event Boundaries ======

        row: dict[str, Any] = {
            "trial_index": candidate_index,                          # Candidate numbering remains 1-based and records gaps where policy skipped trials.
            "start_time": start_time,                                # Numeric boundary used by pose/video/time-based stream slicing.
            "end_time": close_time,
            "start_inclusive": bool(include_start),                   # Emit the actual automatic/explicit decision for downstream selection.
            "end_inclusive": bool(spec.end_inclusive),
        }
        if has_event_ids:
            row["first_event_index"] = pd.NA if window.empty else int(window["event_index"].iloc[0])
            row["last_event_index"] = pd.NA if window.empty else int(window["event_index"].iloc[-1])

        # === 5.5| Read Experiment-Specific Fields from the Trial Window ============
        # fields receives both the restricted window and the closing row because
        # some tasks encode useful outcome/port values directly in the close.

        field_values = spec.fields(window, close_row)
        if not isinstance(field_values, dict):
            raise TypeError(f"TrialSpec.fields must return a dict, got {type(field_values).__name__} for trial {candidate_index}.")

        collisions = [column for column in field_values if column in protected_columns]
        if collisions:
            raise ValueError(f"TrialSpec.fields cannot overwrite reserved trial columns: {sorted(collisions)}.")

        row.update(field_values)                                     # Extracted values join the same plain row dictionary used by derived calculations.

        # === 5.6| Compute Values Derived from the Row Assembled So Far =============
        # Derived calculations run after field extraction so they can depend on
        # both universal delimiting times and experiment-specific extracted values.
        # Give the callback a copy so assigning a protected key cannot bypass the
        # checks below; only its returned dictionary can add output columns.

        calculated_values = spec.derived(row.copy())                 # Copy keys while retaining existing values for the caller's calculations.
        if not isinstance(calculated_values, dict):
            raise TypeError(f"TrialSpec.derived must return a dict, got {type(calculated_values).__name__} for trial {candidate_index}.")

        collisions = [column for column in calculated_values if column in row or column in protected_columns]
        if collisions:
            raise ValueError(f"TrialSpec.derived cannot overwrite existing or reserved trial columns: {sorted(collisions)}.")

        row.update(calculated_values)                                # Durations/outcomes can depend on fields already present without overwriting them.

        # === 5.7| Expand Named Segments into Standard Bounds and Inclusion Flags ===
        # Segment definitions refer to columns already present in the row. An
        # absent source column raises; an existing column containing NaN remains
        # missing so downstream slicing can return an empty interval. A segment
        # inherits a trial inclusion flag only when its source names that outer
        # boundary; internal event-derived boundaries are inclusive by default.

        for name, (start_column, end_column) in spec.segments.items():
            missing = [column for column in (start_column, end_column) if column not in row]
            if missing:
                raise KeyError(f"segment {name!r} references missing trial columns: {missing}.")

            segment_start = row[start_column]
            segment_end = row[end_column]
            if pd.notna(segment_start) and pd.notna(segment_end) and segment_start > segment_end:
                raise ValueError(f"segment {name!r} has start {segment_start!r} after end {segment_end!r} for trial {candidate_index}.")

            row[f"{name}_start_time"] = segment_start                 # Copy source value to the standard generic segment-start column.
            row[f"{name}_end_time"] = segment_end                     # Copy source value to the standard generic segment-end column.
            source_inclusion = {"start_time": row["start_inclusive"], "end_time": row["end_inclusive"]}
            row[f"{name}_start_inclusive"] = source_inclusion.get(start_column, True)
            row[f"{name}_end_inclusive"] = source_inclusion.get(end_column, True)

        # === 5.8| Store the Completed Row and Anchor the Next Candidate ============

        trials.append(row)
        previous_end = close_time                                    # Next implicit start or explicit-buffer check follows this actual close.
        previous_close_position = close_position

    # === 6| Report an Unclosed Explicit Start and Construct the Trial Table =======
    # Implicit mode stops after its final close, as before. Explicit starts carry
    # independent information that a final trial began but never closed; report
    # that omission rather than silently returning an apparently complete table.

    if explicit_starts and start_cursor < len(start_positions):
        if len(start_positions) - start_cursor > 1:
            raise ValueError("multiple trial_start events remain without a closing event.")

        warnings.warn("final trial_start has no closing event; that unfinished trial was omitted.", UserWarning, stacklevel=2)

    return pd.DataFrame(trials)                                       # Construct once after the loop rather than repeatedly concatenating rows.


# ===============================================================================
#







# """
# Convert a session event log into a generic trial table: one row per trial.
# ----------------------------------------------------------------------------

# Description:
#     ``trials.py`` implements experiment-agnostic event-log -> trial-table
#     processing. The module deliberately separates DATA describing what a trial is
#     from the FUNCTION that applies that description:

#         TrialSpec (experiment definition)
#             + one session's events DataFrame
#             -> parse_trials(...)
#             -> one generic trial DataFrame.

#     The parser supports two explicit trial models:
#       * ``trial_start=None`` retains the original previous-end + buffer model;
#       * a ``trial_start`` rule opens each trial independently, allowing gaps;
#       * ``closes_trial`` closes an open trial; ``session_start`` can override its
#         first start and ``last_trial_end`` can close its final unfinished trial;
#       * start/end inclusion and insufficient inter-trial separation are explicit
#         policies, with the original defaults retained;
#       * ``fields`` reads experiment-specific values from the events inside one
#         trial window;
#       * ``derived`` computes values from the row assembled so far;
#       * ``segments`` exposes named time intervals as standard
#         ``{segment}_start_time`` / ``{segment}_end_time`` columns.

#     Nothing here knows Q_C-specific event strings, ports, outcomes, or scientific
#     interpretation. Those belong in the experiment's TrialSpec/configuration.

# Contents:
# --------------------------------
# - TrialSpec:       Data object describing one experiment's trial structure.
# - within:          Select event rows inside an inclusive [start, end] window.
# - first_matching:  Find the first event containing a literal pattern.
# - segment_bounds:  Resolve segment names to trial-table bound columns.
# - _matching_positions: Resolve a boundary rule to matching event-row positions.
# - parse_trials:    Apply a TrialSpec or direct options to one session event log.
# """


# ################################################################################


# ################################################################################
# # Imports
# ################################################################################

# from collections.abc import Callable
# from dataclasses import dataclass, field
# from numbers import Real
# from typing import Any
# import warnings

# import numpy as np
# import pandas as pd

# ################################################################################


# __all__ = [
#     "TrialSpec",
#     "parse_trials",
#     "within",
#     "first_matching",
#     "segment_bounds",
# ]


# ################################################################################
# # Types
# ################################################################################

# # A boundary rule names literal event text or evaluates the complete event table.
# # Callables can combine pandas comparisons with &, | and ~; no expression strings
# # are evaluated. Matching positions always retain the caller's original row order.
# EventRule = str | Callable[[pd.DataFrame], pd.Series | np.ndarray]
# EventFields = Callable[[pd.DataFrame, pd.Series], dict[str, Any]]       # Selected events + closing row -> extracted trial columns.
# DerivedValues = Callable[[dict[str, Any]], dict[str, Any]]             # Assembled row -> additional calculated columns.


# ################################################################################



# ################################################################################
# # Trial Specification (Data Object)
# ################################################################################


# # ===============================================================================
# # 1| TrialSpec (Describe What One Experiment Means by a Trial)
# # ===============================================================================


# @dataclass
# class TrialSpec:
#     """
#     Describe how one session's event log should be converted into trial rows.

#     ``TrialSpec`` contains experiment-specific policy but no parsing behaviour of
#     its own. ``parse_trials`` reads these fields and applies the generic workflow.
#     This keeps task vocabulary/configuration outside the reusable parser.

#     Parameters
#     ----------
#     closes_trial : str
#         Prefix identifying events that close trials. Matching uses
#         ``Series.str.startswith`` rather than substring matching, so each event
#         whose text begins with this string is treated as one trial-ending row.
#     session_start : str | None
#         Literal substring identifying the first trial's start marker. If None,
#         the first event in the session log becomes the first trial's start. If a
#         string is supplied but never occurs, parsing raises rather than guessing a
#         start for a truncated/incomplete event log.
#     start_buffer : float
#         Seconds added to the previous trial's end when defining the next trial's
#         start. ``0.0`` makes consecutive trials contiguous at their boundary;
#         positive values deliberately exclude an inter-trial buffer.
#     fields : Callable[[pd.DataFrame, pd.Series], dict[str, Any]]
#         Experiment-specific function called once per trial. It receives the
#         events inside that trial's inclusive ``[start, end]`` window and the
#         closing event row, then returns additional column values read directly
#         from those events. Default returns no extra fields.
#     derived : Callable[[dict[str, Any]], dict[str, Any]]
#         Experiment-specific function called after ``fields``. It receives the row
#         assembled so far and returns values computed FROM existing row values,
#         such as durations or categorical outcomes. Keeping this separate from
#         ``fields`` makes event extraction and derived calculations explicit.
#         Default returns no extra values.
#     segments : dict[str, tuple[str, str]]
#         Mapping from segment name to two EXISTING row-column names representing
#         its start and end. For example
#         ``{'outbound': ('target_time', 'poke_time')}``. ``parse_trials`` copies
#         those values into standard ``outbound_start_time`` /
#         ``outbound_end_time`` columns for downstream generic slicing.
#     """

#     closes_trial: str                                                           # Event-text prefix that defines one trial-ending event.
#     session_start: str | None = None                                            # Optional first-trial start marker matched as a literal substring.
#     start_buffer: float = 0.0                                                   # Seconds added after previous close before next trial begins.
#     fields: Callable[[pd.DataFrame, pd.Series], dict[str, Any]] = (             # Reads experiment-specific values directly from one trial window.
#         lambda window, closing_row: {}
#     )

#     derived: Callable[[dict[str, Any]], dict[str, Any]] = (                     # Computes additional values from the row assembled so far.
#         lambda row: {}
#     )

#     segments: dict[str, tuple[str, str]] = field(default_factory=dict)          # Segment name -> source start/end column names.


# # ===============================================================================


# ################################################################################
# # Shared Helpers
# ################################################################################


# # ===============================================================================
# # 1| Select Event Rows Inside an Inclusive Time Window
# # ===============================================================================


# def within(
#     events: pd.DataFrame,  # Time-indexed event rows to restrict.
#     start: float,  # Inclusive lower time bound.
#     end: float,  # Inclusive upper time bound.
# ) -> pd.DataFrame:  # Returns event rows whose index lies within [start, end].
#     """
#     Return event rows whose time index lies inside the inclusive ``[start, end]``.

#     Parameters
#     ----------
#     events : pd.DataFrame
#         Event table whose index contains numeric event times. The function does
#         not require a specific event-text column because it only performs temporal
#         slicing.
#     start : float
#         Inclusive lower time bound.
#     end : float
#         Inclusive upper time bound.

#     Returns
#     -------
#     pd.DataFrame
#         Subset of ``events`` whose index satisfies ``start <= time <= end``. The
#         original index and columns are preserved.
#     """

#     if start > end:
#         raise ValueError(f"start must be less than or equal to end; got {start!r} > {end!r}.")

#     # === 1| Apply Both Inclusive Bounds to the Event-Time Index ===================
#     # Inclusive bounds are deliberate: the marker defining the start and the
#     # closing event defining the end both conceptually belong to the trial they
#     # delimit and may need to be inspected by ``fields``.

#     return events[(events.index >= start) & (events.index <= end)]  # One boolean expression preserves original row/index ordering.


# # ===============================================================================


# # ===============================================================================
# # 2| Find the First Event Containing a Literal Text Pattern
# # ===============================================================================


# def first_matching(
#     window: pd.DataFrame,  # Event rows in which to search.
#     pattern: str,  # Literal substring identifying the event of interest.
#     *,
#     extract: Callable[[pd.Series], object] | None = None,  # Optional row->value extraction function.
#     default: object = np.nan,  # Value returned when no event matches.
#     column: str = "Event",  # Column containing event text.
# ) -> object:  # Returns first matching time/value or ``default``.
#     """
#     Return the first event matching a literal substring, optionally extracting a value.

#     With ``extract=None`` the helper answers WHEN the event first occurred by
#     returning the matching row's index value. With an ``extract`` callable it
#     instead answers WHAT should be read from that first matching row. Literal
#     matching (``regex=False``) is intentional because experiment event text often
#     contains brackets, colons, or other characters that should not be interpreted
#     as regular-expression syntax.

#     Parameters
#     ----------
#     window : pd.DataFrame
#         Event rows to search, typically one trial's inclusive time window. The
#         DataFrame index is treated as event time when ``extract`` is None.
#     pattern : str
#         Literal substring that must occur in ``column`` for a row to match.
#     extract : Callable[[pd.Series], object] | None
#         Optional function applied to the FIRST matching row. Use this when the
#         desired result is encoded elsewhere in the row rather than being its time.
#         None returns the matching row's index value instead.
#     default : object
#         Value returned when no event matches. Defaults to ``np.nan`` so missing
#         numeric trial fields naturally remain representable in pandas tables.
#     column : str
#         Name of the column containing event text. Default ``'Event'``.

#     Returns
#     -------
#     object
#         First match's index/time when ``extract`` is None, ``extract(first_row)``
#         when a callable is supplied, or ``default`` when there are no matches.
#     """

#     # === 1| Find All Rows Containing the Literal Pattern ==========================

#     matches = window[
#         window[column].str.contains(
#             pattern,  # Caller-provided event substring.
#             na=False,  # Missing event strings are simply non-matches.
#             regex=False,  # Event text is literal data, not regex syntax.
#         )
#     ]

#     # === 2| Return the Caller-Specified Missing Value When Nothing Matches ========

#     if matches.empty:
#         return default

#     # === 3| Return Either the Matching Time or an Extracted Row Value =============

#     if extract is None:
#         return matches.index[0]  # First row in preserved event order; its index is event time.

#     return extract(matches.iloc[0])  # Apply caller's row->value rule only to the first matching event.


# # ===============================================================================


# # ===============================================================================
# # 3| Resolve Every Trial Segment to Standard Start/End Column Names
# # ===============================================================================


# def segment_bounds(
#     spec: TrialSpec,  # TrialSpec whose declared segments should be resolved.
# ) -> dict[str, tuple[str, str]]:  # Returns segment name -> emitted trial-table bound columns.
#     """
#     Return the standard trial-table start/end columns for each segment in a spec.

#     ``TrialSpec.segments`` names the SOURCE row values used when constructing
#     segments. After parsing, each segment is exposed under the STANDARD emitted
#     columns ``{name}_start_time`` and ``{name}_end_time``. This helper describes
#     that output schema to downstream generic code such as ``slicing.py``.

#     The full trial is always available under the synthetic ``'trial'`` segment,
#     mapped to base ``start_time`` / ``end_time``. If a spec explicitly defines a
#     segment named ``'trial'``, that explicit emitted segment pair is retained.

#     Parameters
#     ----------
#     spec : TrialSpec
#         Trial specification whose segment declarations define the emitted segment
#         columns.

#     Returns
#     -------
#     dict[str, tuple[str, str]]
#         Mapping from segment name to ``(start_column, end_column)`` in the parsed
#         trial table.
#     """

#     # === 1| Convert Declared Segment Names to their Emitted Standard Columns ======

#     bounds = {
#         name: (f"{name}_start_time", f"{name}_end_time")  # parse_trials always emits this standard pair for each named segment.
#         for name in spec.segments
#     }

#     # === 2| Guarantee a Whole-Trial Segment Exists ================================
#     # ``setdefault`` deliberately does NOT overwrite a caller-declared ``trial``
#     # segment. Ordinarily, however, it exposes the parser's universal base bounds.

#     bounds.setdefault("trial", ("start_time", "end_time"))  # Full trial always has base delimiting columns.

#     return bounds


# # ===============================================================================


# ################################################################################
# # Private Helper
# ################################################################################


# # ===============================================================================
# # 1| Resolve One Trial's Start Time
# # ===============================================================================


# def _trial_start(
#     events: pd.DataFrame,  # Complete session event log used to anchor the first trial.
#     trial_index: int,  # Zero-based position of the trial currently being parsed.
#     previous_end: float | None,  # Previous trial's close time; None for the first trial.
#     spec: TrialSpec,  # TrialSpec supplying first-start marker and inter-trial buffer.
# ) -> float:  # Returns start time for the current trial.
#     """
#     Resolve the current trial's start time from its position and TrialSpec.

#     Later trials have a simple rule: previous trial end + ``start_buffer``. The
#     first trial has no previous close and therefore uses either the configured
#     ``session_start`` marker or the first event time in the log.

#     Parameters
#     ----------
#     events : pd.DataFrame
#         Complete session event log. Only the first trial needs to inspect it for
#         a first-event time or configured ``session_start`` marker.
#     trial_index : int
#         Zero-based trial position. ``0`` means first trial; any non-zero value
#         follows the previous-end + buffer rule.
#     previous_end : float | None
#         Previous trial's closing time. Must contain a numeric value for later
#         trials; None is expected only when ``trial_index == 0``.
#     spec : TrialSpec
#         Trial specification containing ``session_start`` and ``start_buffer``.

#     Returns
#     -------
#     float
#         Resolved start time for the current trial.
#     """

#     # === 1| Later Trials Start after the Previous Close + Configured Buffer =======

#     if trial_index != 0:
#         return previous_end + spec.start_buffer  # Parser controls previous_end, so later trials always have a close anchor.

#     # === 2| First Trial Without an Explicit Marker Starts at First Logged Event ===

#     if spec.session_start is None:
#         return events.index[0]  # No invented pre-session time; anchor on first actual event.

#     # === 3| First Trial with a Marker Must Find that Marker =======================
#     # Guessing from the first event when an explicit marker was requested would
#     # silently accept truncated logs and produce incorrectly delimited trials.

#     start = first_matching(
#         events,
#         spec.session_start,  # Literal substring identifying first-trial start marker.
#         default=None,  # None distinguishes "marker missing" from any numeric time value.
#     )

#     if start is None:
#         raise ValueError(f"no {spec.session_start!r} event in the log: cannot determine the first trial's start.")

#     return start


# # ===============================================================================


# ################################################################################
# # Public API
# ################################################################################


# # ===============================================================================
# # 1| Parse One Session Event Log into One Row per Trial
# # ===============================================================================


# def parse_trials(
#     events: pd.DataFrame,  # One session's Time-indexed event log with an Event text column.
#     spec: TrialSpec,  # Experiment-specific trial description applied to this log.
# ) -> pd.DataFrame:  # Returns generic trial table, one row per closing event/trial.
#     """
#     Apply ``spec`` to one session's event log and return a trial table.

#     Every event whose text starts with ``spec.closes_trial`` defines exactly one
#     trial end. Walking those closing rows in time order therefore provides the
#     trial sequence. For each trial, the parser reconstructs its inclusive
#     ``[start, end]`` event window, asks the spec for extracted fields and derived
#     values, expands named segments into standard time-bound columns, and appends
#     one completed row.

#     Parameters
#     ----------
#     events : pd.DataFrame
#         One session's event table. The DataFrame must have a time-like numeric
#         index and an ``'Event'`` text column. Rows are assumed to be in the event
#         order represented by that index; the parser does not reorder them.
#     spec : TrialSpec
#         Experiment-specific description of how trials close/start, which values
#         should be extracted from each trial window, which derived values should be
#         calculated, and which named segments should be emitted.

#     Returns
#     -------
#     pd.DataFrame
#         One row per closing event/trial. Every non-empty output contains
#         ``trial_index`` (1-based), ``start_time``, and ``end_time``, plus fields
#         returned by ``spec.fields``, values returned by ``spec.derived``, and two
#         standard columns per declared segment. If no closing events are present,
#         an empty DataFrame is returned.
#     """

#     # === 1| Validate the Event Stream and Trial Policy =============================

#     if not isinstance(events, pd.DataFrame):
#         raise TypeError(f"events must be a pandas DataFrame, got {type(events).__name__}.")
#     if not isinstance(spec, TrialSpec):
#         raise TypeError(f"spec must be a TrialSpec, got {type(spec).__name__}.")
#     if "Event" not in events.columns:
#         raise KeyError("events must contain an 'Event' column.")
#     if not pd.api.types.is_numeric_dtype(events.index.dtype):
#         raise TypeError("events index must contain numeric event times.")
#     event_times = events.index.to_numpy(dtype=float)
#     if not np.isfinite(event_times).all():
#         raise ValueError("events index must contain only finite event times.")
#     if not events.index.is_monotonic_increasing:
#         raise ValueError("events must be ordered by a monotonically non-decreasing time index.")
#     if not isinstance(spec.start_buffer, Real) or not np.isfinite(spec.start_buffer) or spec.start_buffer < 0:
#         raise ValueError("TrialSpec.start_buffer must be a finite, non-negative number.")

#     protected_columns = {
#         "trial_index",
#         "start_time",
#         "end_time",
#         *(f"{name}_{bound}_time" for name in spec.segments for bound in ("start", "end")),
#     }

#     # === 2| Identify the Closing Event that Defines Each Trial ====================
#     # Matching uses ``startswith`` because the spec declares a closing-event
#     # prefix. Each matching row ends one trial, so the resulting row sequence is
#     # also the sequence of trials to reconstruct.

#     closing_rows = events[
#         events["Event"].str.startswith(
#             spec.closes_trial,  # Experiment-specific closing-event prefix.
#             na=False,  # Missing event text cannot close a trial.
#         )
#     ]

#     # === 3| Walk Closing Events and Build One Trial Row for Each ==================

#     trials: list[dict[str, Any]] = []  # Accumulate plain row dictionaries before one final DataFrame construction.
#     previous_end: float | None = None  # First trial has no preceding close anchor.

#     for i, (close_time, close_row) in enumerate(closing_rows.iterrows()):
#         # === 2.1| Resolve the Trial's Inclusive [Start, End] Bounds ================

#         start_time = _trial_start(
#             events,  # Complete session log needed only for first-trial anchoring.
#             i,  # Zero-based loop index tells helper whether this is the first trial.
#             previous_end,  # Previous closing event anchors later trial starts.
#             spec,  # Supplies first marker and inter-trial buffer policy.
#         )
#         if start_time > close_time:
#             raise ValueError(f"trial {i + 1} has start_time {start_time!r} after its closing event at {close_time!r}; check event ordering and TrialSpec.start_buffer.")

#         # With a zero buffer the next trial's numeric start equals the previous
#         # close. The previous closing event belongs only to the trial it closed,
#         # so exclude that shared boundary row from the next field-extraction
#         # window while retaining the contiguous start_time value in the table.
#         window = events[(events.index > start_time) & (events.index <= close_time)] if i and start_time == previous_end else within(events, start_time, close_time)

#         # === 2.2| Create the Universal Base Trial Columns =========================

#         row: dict[str, Any] = {
#             "trial_index": i + 1,  # User-facing trial index is 1-based even though loop position is 0-based.
#             "start_time": start_time,  # Resolved inclusive lower bound.
#             "end_time": close_time,  # Current closing event's time is the inclusive upper bound.
#         }

#         # === 2.3| Read Experiment-Specific Fields from the Trial Window ============
#         # ``fields`` receives both the restricted event window and the closing row
#         # because some tasks encode useful trial values directly in the close event.

#         field_values = spec.fields(window, close_row)
#         if not isinstance(field_values, dict):
#             raise TypeError(f"TrialSpec.fields must return a dict, got {type(field_values).__name__} for trial {i + 1}.")
#         collisions = protected_columns.intersection(field_values)
#         if collisions:
#             raise ValueError(f"TrialSpec.fields cannot overwrite reserved trial columns: {sorted(collisions)}.")
#         row.update(field_values)  # Extracted values join the same plain row dictionary.

#         # === 2.4| Compute Values Derived from the Row Assembled So Far =============
#         # Derived calculations run after field extraction so they can depend on
#         # both universal delimiting times and experiment-specific extracted values.

#         derived_values = spec.derived(row)
#         if not isinstance(derived_values, dict):
#             raise TypeError(f"TrialSpec.derived must return a dict, got {type(derived_values).__name__} for trial {i + 1}.")
#         collisions = set(row).intersection(derived_values) | protected_columns.intersection(derived_values)
#         if collisions:
#             raise ValueError(f"TrialSpec.derived cannot overwrite existing or reserved trial columns: {sorted(collisions)}.")
#         row.update(derived_values)  # Durations/outcomes/etc. can depend on fields already present.

#         # === 2.5| Expand Named Segments into Standard Start/End Output Columns =====
#         # Segment definitions refer to values already present in the row. ``get``
#         # intentionally yields None for a missing source column rather than
#         # raising here; the resulting missing boundary remains visible in output
#         # and naturally produces an empty slice in downstream comparisons.

#         for name, (start_column, end_column) in spec.segments.items():
#             missing = [column for column in (start_column, end_column) if column not in row]
#             if missing:
#                 raise KeyError(f"segment {name!r} references missing trial columns: {missing}.")

#             segment_start = row[start_column]
#             segment_end = row[end_column]
#             if pd.notna(segment_start) and pd.notna(segment_end) and segment_start > segment_end:
#                 raise ValueError(f"segment {name!r} has start {segment_start!r} after end {segment_end!r} for trial {i + 1}.")
#             row[f"{name}_start_time"] = segment_start  # Copy source value to the standard generic segment-start column.
#             row[f"{name}_end_time"] = segment_end  # Copy source value to the standard generic segment-end column.

#         # === 2.6| Store the Completed Trial Row ===================================

#         trials.append(row)

#         # === 2.7| Make this Close Event the Anchor for the Next Trial ==============

#         previous_end = close_time  # Next iteration applies ``previous_end + start_buffer``.

#     # === 4| Convert Row Dictionaries into the Final Trial DataFrame ===============
#     # Construct once after the loop rather than repeatedly concatenating rows,
#     # which is both clearer and substantially cheaper for many trials.

#     return pd.DataFrame(trials)


# # ===============================================================================
# #