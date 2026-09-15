"""
Select pandas/xarray data directly or use trial rows to select matching streams.
---------------------------------------------------------------------

Description:
    ``slicing.py`` selects DataFrame rows and xarray coordinates using caller-
    supplied values, intervals, or Boolean conditions. Direct selection does not
    require a session identifier, trial table, or second data stream.

    A combined stream can contain repeated time values because each session often
    starts its own local clock at or near zero. Time alone is therefore not a
    globally unique selector. Correct slicing must constrain BOTH:
      * which source session contributed the data; and
      * which time interval inside that session is required.

    ``slice_stream`` performs direct selection and retains optional session/time
    keywords for the existing callers. ``slice_stream_for_trial`` obtains the
    recording identity and bounds from one selected trial row. The wrapper uses
    stable event identifiers when present on both inputs, so tied event times do
    not change which rows belong to the trial. ``slice_stream_per_trial`` repeats
    that operation for every row in a trial subset.

    Selection never changes the input in place or automatically changes another
    stream. Use pandas/xarray ``assign`` and ``drop`` on the returned data when
    adding, modifying, or removing values; pass selected trial rows to the trial
    wrappers when the same recording/time selection should determine pose.

    This module is intentionally experiment-agnostic. It understands the generic
    trial-segment column conventions emitted by ``parse_trials`` but does not know
    what experiment-specific names such as "outbound" or "search" mean.

Contents:
--------------------------------
- _segment_columns:        Resolve one segment name to trial-table bound columns.
- _selection_mask:        Translate one value/interval/condition into a mask.
- _apply_mask:            Apply a Boolean mask without reordering retained data.
- slice_stream:            Select rows/coordinates, optionally by session + time.
- slice_stream_for_trial:  Slice a stream using one trial row's segment bounds.
- slice_stream_per_trial:  Produce one stream slice per trial-table row.
"""


################################################################################
# Imports
################################################################################

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .streams import DataObject
from .trials import TrialSpec, segment_bounds

################################################################################


################################################################################
# Private Helpers
################################################################################


# ===============================================================================
# 1| Resolve a Trial Segment Name to its Start/End Table Columns
# ===============================================================================


def _segment_columns(
    segment: str,  # Segment name requested by the caller, e.g. 'trial' or 'outbound'.
    spec: TrialSpec | None = None,  # Optional TrialSpec used to validate/resolve segment definitions.
) -> tuple[str, str]:  # Returns ``(start_column, end_column)`` in the trial table.
    """
    Resolve a segment name into the two trial-table columns that bound it.

    When a ``TrialSpec`` is available, ``segment_bounds(spec)`` is treated as the
    source of truth. This validates the requested segment against the exact spec
    that produced the table. Without a spec, the helper falls back to the generic
    naming convention emitted by ``parse_trials``:
    ``{segment}_start_time`` / ``{segment}_end_time``. The special whole-trial
    segment maps to the base ``start_time`` / ``end_time`` columns.

    Parameters
    ----------
    segment : str
        Segment name to resolve. ``'trial'`` always denotes the full trial span.
        Other values are experiment-defined names already represented in the trial
        table, such as ``'outbound'``.
    spec : TrialSpec | None
        Optional TrialSpec corresponding to the trial table. If supplied, the
        requested segment must occur in ``segment_bounds(spec)``; otherwise a
        ``ValueError`` is raised instead of guessing a column pair.

    Returns
    -------
    tuple[str, str]
        Two-element tuple ``(start_column, end_column)`` containing the column
        names that hold the requested segment's time bounds.
    """

    # === 1| Prefer the TrialSpec When Available ==================================
    # The spec is the strongest source of truth because a future/experiment-specific
    # spec could intentionally map a segment differently from the default naming
    # convention. Validate before returning to catch typos early.

    if spec is not None:
        bounds = segment_bounds(spec)  # Build the complete legal segment->column mapping for this spec.

        if segment not in bounds:  # Caller requested a name the supplied spec does not define.
            raise ValueError(f"segment must be one of {sorted(bounds)}, got {segment!r}.")

        return bounds[segment]  # Exact pair defined by the supplied TrialSpec.

    # === 2| Handle the Whole-Trial Special Case ==================================

    if segment == "trial":  # Base delimiting columns are always emitted by parse_trials.
        return "start_time", "end_time"

    # === 3| Fall Back to the Generic Segment Naming Convention ===================

    return f"{segment}_start_time", f"{segment}_end_time"  # Matches parse_trials segment-column output convention.


# ===============================================================================


# ===============================================================================
# 2| Translate One Field Rule into a Boolean Mask
# ===============================================================================


def _selection_mask(
    values: pd.Series | xr.DataArray,                         # One named field/coordinate on the currently selected data.
    rule: Any,                                                # Scalar, collection, inclusive slice, or Boolean-returning callable.
) -> Any:                                                     # Boolean mask retaining pandas/xarray labels when available.
    """
    Evaluate one coordinate/field condition without selecting the data yet.

    Parameters
    ----------
    values : pd.Series | xr.DataArray
        Field values retaining their row index or coordinate dimensions.
    rule : Any
        Callable, inclusive slice, membership collection, None, or equality value.
        A callable receives ``values`` and must return a Boolean mask.

    Returns
    -------
    Any
        Boolean mask consumed and validated by ``_apply_mask``. Missing values in
        the mask are treated as False there.
    """

    # === 1| Let a Callable Express a Field-Specific Condition ===================

    if callable(rule):
        return rule(values)                                    # Example: confidence >= 0.9 without a fixed confidence API.

    # === 2| Interpret Slices as Inclusive Coordinate Bounds =====================

    if isinstance(rule, slice):
        if rule.step is not None:
            raise ValueError("selector slices accept bounds only; use native iloc/isel for positional steps.")

        mask = values.notnull()                                # Missing coordinates cannot fall inside a requested interval.
        if rule.start is not None:
            mask = mask & (values >= rule.start)                # Lower bound uses coordinate values rather than row positions.
        if rule.stop is not None:
            mask = mask & (values <= rule.stop)                 # Upper bound is inclusive for both pandas and xarray.
        return mask

    # === 3| Distinguish Membership, Missing Values, and Scalar Equality ===========

    if isinstance(rule, list | tuple | set | np.ndarray | pd.Index):
        return values.isin(list(rule))                         # A collection retains each coordinate found in the collection.
    if rule is None:
        return values.isnull()                                  # None requests missing values; it does not disable this selector.
    return values == rule                                      # Strings are whole values, never executable query expressions.


# ===============================================================================
# 3| Apply a Boolean Mask While Preserving Source Order
# ===============================================================================


def _apply_mask(
    stream: DataObject,                                        # Object against which the condition was evaluated.
    mask: Any,                                                # Boolean Series/array for pandas or labelled DataArray for xarray.
) -> DataObject:                                              # Selected rows/coordinates in their existing order.
    """
    Validate one mask and apply it without guessing xarray dimensions.

    Parameters
    ----------
    stream : DataObject
        DataFrame, DataArray or Dataset being restricted.
    mask : Any
        For pandas, a Boolean Series with exactly the same index or a one-
        dimensional Boolean array with one value per row. For xarray, a Boolean
        DataArray with matching dimensions and coordinate labels.

    Returns
    -------
    DataObject
        Selected data. One-dimensional xarray conditions use ``isel`` to retain
        integer/string dtypes. Multi-dimensional conditions use ``where`` and may
        introduce missing cells to preserve a rectangular array. Scalar coordinate
        conditions retain the whole object when True; False empties its dimensions
        (or returns a missing scalar for a zero-dimensional input).
    """

    # === 1| Require Exactly One Boolean Decision per DataFrame Row ===============

    if isinstance(stream, pd.DataFrame):
        if isinstance(mask, pd.Series) and not mask.index.equals(stream.index):
            raise ValueError("a pandas mask must have exactly the currently selected row index.")
        mask = pd.Series(mask, copy=False)                      # Also accepts an unlabelled Boolean array for positional selection.
        if not pd.api.types.is_bool_dtype(mask.dtype):
            raise TypeError("selection masks must contain booleans.")
        if len(mask) != len(stream):
            raise ValueError("a pandas mask must have one value per currently selected row.")
        return stream.iloc[np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool))]

    # === 2| Require xarray Dimension Labels and Matching Coordinate Indexes =======

    if not isinstance(mask, xr.DataArray) or mask.dtype.kind != "b":
        raise TypeError("an xarray mask must be a Boolean DataArray with explicit dimension labels.")
    for dimension in mask.dims:
        if dimension not in stream.dims or mask.sizes[dimension] != stream.sizes[dimension]:
            raise ValueError(f"mask dimension {dimension!r} does not match the currently selected stream.")
    xr.align(stream, mask, join="exact", copy=False)             # Reject mismatched labels rather than silently aligning away rows.

    # === 3| Select One Dimension Directly or Mask a Rectangular Array =============

    if mask.ndim == 1:
        dimension = mask.dims[0]
        return stream.isel({dimension: np.flatnonzero(mask.values)})
    if mask.ndim == 0:
        if bool(mask.item()):
            return stream                                     # A matching scalar coordinate describes the entire input object.
        return stream.isel({dimension: slice(0, 0) for dimension in stream.dims}).where(mask)
    return stream.where(mask, drop=True)                        # A joint time/keypoint condition can leave holes inside the rectangle.


# ===============================================================================


################################################################################
# Public API
################################################################################


# ===============================================================================
# 1| Select DataFrame Rows or xarray Coordinates Using Explicit Conditions
# ===============================================================================


def slice_stream(
    stream: DataObject,                                      # Data to select; session/trial metadata is optional.
    *,
    session: Any = None,                                      # Optional recording filter retained for existing callers.
    start: float | None = None,                               # Optional lower bound on time_coord.
    end: float | None = None,                                 # Optional upper bound on time_coord.
    session_coord: str = "session",                          # Coordinate/column/index level used by session.
    time_coord: str = "Time",                                # Coordinate/column/index level used by start/end.
    selectors: Mapping[str, Any] | None = None,                # Named coordinate/field conditions combined with AND.
    where: Callable[[DataObject], Any] | pd.Series | xr.DataArray | np.ndarray | None = None,    # Final Boolean mask or callable receiving selected data.
    start_inclusive: bool = True,                            # Include equality at the optional time start.
    end_inclusive: bool = True,                              # Include equality at the optional time end.
) -> DataObject:                                              # Same broad object type with requested elements retained.
    """
    Select pandas rows or xarray coordinates without requiring a trial table.

    Named selectors run in mapping order; each condition sees data retained by
    earlier conditions. ``session``, ``start`` and ``end`` then add optional
    constraints, and ``where`` runs last. All supplied conditions must hold.
    Retained elements keep their original order, including duplicate timestamps.

    Parameters
    ----------
    stream : DataObject
        DataFrame, DataArray or Dataset. DataFrame selectors name a column or
        named index level; a column takes precedence. xarray selectors name any
        existing coordinate, which need not be a dimension coordinate.
    session : Any, optional
        Recording identifier compared to ``session_coord``. None adds no recording
        filter. Use ``selectors={"session": None}`` to select missing identities.
        When several recordings share a clock, supply their identity explicitly
        if the time interval should apply to only one of them.
    start, end : float | None
        Optional lower/upper time bounds. None leaves that side unrestricted;
        NaN produces an empty selection. Bounds use the input coordinate's units.
    session_coord, time_coord : str
        Names used by the convenience arguments; defaults ``'session'`` and
        ``'Time'``. No corresponding coordinate is required if its convenience
        argument is omitted. A time coordinate need not be a dimension itself.
    selectors : Mapping[str, Any] | None
        One rule per field/coordinate: a scalar selects equality; None selects
        missing values; a list/tuple/set/array selects membership; a ``slice``
        selects inclusive lower/upper bounds; a callable receives the field and
        returns a Boolean mask. Slice steps are unsupported. Independent xarray
        dimensions can be selected in the same mapping.
    where : callable or Boolean mask, optional
        Callable receiving the already selected object, or a Boolean mask aligned
        with that object. This permits conditions involving several DataFrame
        columns or xarray data variables. For pandas, a Series must have exactly
        the selected index; a Boolean array must have one entry per retained row.
        For xarray, supply a labelled DataArray so its dimensions are explicit.
    start_inclusive, end_inclusive : bool
        Whether ``start``/``end`` retain equal coordinate values; both default
        True. These flags do not change interval rules inside ``selectors``.

    Returns
    -------
    DataObject
        Selected object without modifying the input. DataFrame results are copies.
        One-dimensional xarray masks select positions and preserve dtype; masks
        spanning several dimensions retain a rectangular array and fill excluded
        cells with missing values. No changes propagate to other streams.
    """

    # === 1| Validate the Input Type and the Two Endpoint Flags ==================

    if not isinstance(stream, pd.DataFrame | xr.DataArray | xr.Dataset):
        raise TypeError(f"cannot slice {type(stream).__name__}; expected DataArray / Dataset / DataFrame.")
    if selectors is not None and not isinstance(selectors, Mapping):
        raise TypeError("selectors must be a mapping from field names to conditions.")
    if not isinstance(start_inclusive, bool) or not isinstance(end_inclusive, bool):
        raise TypeError("start_inclusive and end_inclusive must be booleans.")

    # === 2| Keep All Explicit Selectors and Append Optional Recording/Time Rules ===
    # A list preserves two conditions on the same field. For example a Time
    # selector and start= must both hold rather than one silently replacing the other.

    conditions = list(selectors.items()) if selectors is not None else []

    if session is not None:
        conditions.append((session_coord, session))            # Existing session= callers retain their identity filter.

    if start is not None:
        if start_inclusive:
            conditions.append((time_coord, lambda values: values >= start))  # Keep values equal to the starting time.
        else:
            conditions.append((time_coord, lambda values: values > start))   # Exclude values equal to the starting time.

    if end is not None:
        if end_inclusive:
            conditions.append((time_coord, lambda values: values <= end))    # Keep values equal to the ending time.
        else:
            conditions.append((time_coord, lambda values: values < end))     # Exclude values equal to the ending time.

    # === 3| Resolve Each Requested Field on the Currently Retained Data ===========

    selected = stream

    for name, rule in conditions:
        if isinstance(selected, pd.DataFrame):
            if name in selected.columns:
                values = selected[name]                       # A real column takes precedence over a same-named index level.
            elif name in selected.index.names:
                values = pd.Series(selected.index.get_level_values(name), index=selected.index)
            else:
                raise KeyError(f"stream has no column or index level named {name!r}.")
        else:
            if name not in selected.coords:
                raise KeyError(f"stream has no coordinate named {name!r}; use where for data-variable conditions.")
            values = selected.coords[name]                    # Auxiliary coordinates retain their own dimension labels.

        mask = _selection_mask(values, rule)
        selected = _apply_mask(selected, mask)                 # Every new rule further restricts the previous result.

    # === 4| Apply the Final Cross-Field Condition and Return Independent Table Rows ===

    if where is not None:
        mask = where(selected) if callable(where) else where   # The callable sees the same data its returned mask selects.
        selected = _apply_mask(selected, mask)

    if isinstance(selected, pd.DataFrame):
        return selected.copy()                                # Preserve the original table when the caller edits selected rows.
    return selected                                           # Retain xarray's native selection behavior without an extra full-array copy.


# ===============================================================================


# ===============================================================================
# 2| Slice One Combined Stream Using One Trial Row's Segment Bounds
# ===============================================================================


def slice_stream_for_trial(
    stream: DataObject,  # Combined stream to restrict to one trial/segment.
    trial_row: pd.Series,  # One row from a trial table carrying session + time bounds.
    *,
    segment: str = "trial",  # Segment name whose time bounds should be used.
    spec: TrialSpec | None = None,  # Optional TrialSpec for explicit segment validation/resolution.
    session_column: str = "session",  # Trial-table column identifying the source session.
    session_coord: str = "session",  # Stream coordinate/column identifying source session.
    time_coord: str = "Time",  # Stream time coordinate/column/index name.
) -> DataObject:  # Returns stream subset corresponding to this one trial segment.
    """
    Slice a combined stream to the time interval represented by one trial row.

    Parameters
    ----------
    stream : DataObject
        Stream to slice. Recording identity must be available under
        ``session_coord``; times must be available under ``time_coord`` unless
        exact whole-trial event identifiers can be used instead.
    trial_row : pd.Series
        One row of a trial table. It must contain the source-session identifier and
        the two columns that bound the requested segment. Stored
        ``start_inclusive`` / ``end_inclusive`` flags control whole-trial bounds;
        ``{segment}_start_inclusive`` / ``{segment}_end_inclusive`` control named
        segments. If segment flags are absent, a boundary shared with the whole
        trial inherits that trial's flag; internal boundaries are inclusive.
        Older tables without inclusion columns retain inclusive bounds.
    segment : str
        Trial segment to extract. ``'trial'`` uses the full trial's
        ``start_time`` / ``end_time``. Other names use the segment mapping from
        ``spec`` when supplied or the generic ``{segment}_start_time`` /
        ``{segment}_end_time`` convention otherwise. A spec may explicitly define
        a narrower segment called ``'trial'``; its emitted bounds and flags then
        take precedence over the base whole-trial interval.
    spec : TrialSpec | None
        Optional TrialSpec used to validate and resolve the requested segment.
    session_column : str
        Column in ``trial_row`` containing the source-session id. Default
        ``'session'``.
    session_coord : str
        Coordinate/column on ``stream`` containing source-session identity.
        Default ``'session'``.
    time_coord : str
        Time coordinate/column/named index on ``stream``. Default ``'Time'``.

    Returns
    -------
    DataObject
        Stream subset containing only the selected trial row's source session and
        requested segment interval. A missing trial or segment endpoint (including
        None or NaN) produces an empty selection. If a DataFrame has ``event_index`` and the row
        has ``first_event_index`` / ``last_event_index``, those inclusive IDs
        reproduce the parser's exact whole-trial event window, including tied
        timestamps. Missing IDs in those columns mean an empty event window.

    Notes
    -----
    ``event_index`` must identify rows in the original recording and remain
    unchanged after filtering; this function never generates IDs from the current
    row positions. The catalog assigns them before parsing. Parse the complete
    log first, then filter events for analysis. Exact windows are preserved on
    the same row set or a further subset; first/last IDs cannot preserve holes
    from a filtered parser input when slicing a fuller log later.

    Named event segments are restricted to those same trial IDs, then their time
    bounds. A bound matching the outer trial is redundant only when its inclusion
    flag also matches; a different segment flag still applies its time comparison.
    An internal segment boundary only specifies a timestamp, not a particular tied
    event row. Pose continues to use numeric time bounds and inclusion flags.
    """

    # === 1| Resolve the Segment's Trial-Table Bound Columns =======================

    start_column, end_column = _segment_columns(segment, spec)  # Either spec-defined pair or generic naming convention.
    whole_trial = (start_column, end_column) == ("start_time", "end_time")  # A spec may deliberately redefine the segment named 'trial'.

    # === 2| Validate that the Trial Row Contains Everything Needed ================
    # Collect all missing names and report them together; this is more useful than
    # failing sequentially on the first Series lookup.

    required = [session_column, start_column, end_column]
    missing = [column for column in required if column not in trial_row.index]

    if missing:
        raise KeyError(f"trial row is missing required column(s): {missing}.")

    # === 3| Read Segment Flags or Inherit Flags at Shared Trial Boundaries =========
    # A named segment ending inside a trial is inclusive by default. A segment
    # starting at an excluded trial start inherits that exclusion instead of
    # silently adding the previous trial's endpoint back into the pose slice.

    bounds = {"start": trial_row[start_column], "end": trial_row[end_column]}
    missing_bounds = any(pd.isna(value) for value in bounds.values())
    if missing_bounds:
        # Trial endpoints describe an observed interval. None here means an
        # unavailable interval, whereas the direct slicer uses None for no bound.
        bounds = {"start": np.nan, "end": np.nan}
    inclusive: dict[str, bool] = {}

    for bound in ("start", "end"):
        flag_column = f"{bound}_inclusive" if whole_trial else f"{segment}_{bound}_inclusive"
        flag = trial_row.get(flag_column, None)

        if flag is None:
            flag = True                                       # Older tables and genuinely internal segment bounds remain inclusive.
            other_bound = "end" if bound == "start" else "start"
            for outer_bound in (bound, other_bound):
                outer_time = trial_row.get(f"{outer_bound}_time", np.nan)
                if pd.notna(bounds[bound]) and pd.notna(outer_time) and bounds[bound] == outer_time:
                    flag = trial_row.get(f"{outer_bound}_inclusive", True)
                    break                                     # A matching outer boundary has the same inclusion rule as its trial.
        if not isinstance(flag, bool | np.bool_):
            raise TypeError(f"trial inclusion flag {flag_column!r} must be boolean.")
        inclusive[bound] = bool(flag)

    # === 4| Use Stable Source Event IDs When Both Inputs Carry Them ===============
    # Time alone cannot distinguish a closing event from a later row at the same
    # timestamp. Inclusive first/last IDs encode the already resolved extraction
    # window. Do not recreate IDs here: the input may already be filtered.

    selectors: dict[str, Any] = {session_coord: trial_row[session_column]}
    exact_events = isinstance(stream, pd.DataFrame) and "event_index" in stream.columns
    event_columns = ("first_event_index", "last_event_index")
    if exact_events and any(column in trial_row.index for column in event_columns):
        if not all(column in trial_row.index for column in event_columns):
            raise KeyError("exact event slicing requires both first_event_index and last_event_index.")
    exact_events = exact_events and all(column in trial_row.index for column in event_columns)

    if exact_events:
        first_event = trial_row["first_event_index"]
        last_event = trial_row["last_event_index"]
        if missing_bounds or pd.isna(first_event) or pd.isna(last_event):
            selectors["event_index"] = []                      # An empty extraction window must not fall back to its timestamps.
        else:
            selectors["event_index"] = slice(first_event, last_event)

        if whole_trial or missing_bounds:
            bounds = {"start": None, "end": None}              # Numeric exclusive bounds would discard valid equal-time rows.
        else:
            for bound in ("start", "end"):
                outer_time = trial_row.get(f"{bound}_time", np.nan)
                outer_inclusive = trial_row.get(f"{bound}_inclusive", True)
                if pd.notna(bounds[bound]) and pd.notna(outer_time) and bounds[bound] == outer_time:
                    if inclusive[bound] == outer_inclusive:
                        bounds[bound] = None                  # IDs encode the same bound and flag; a different segment flag still needs its comparison.

    # === 5| Apply the Recording Identity and Remaining Interval Constraints =======

    return slice_stream(
        stream,
        selectors=selectors,                                  # The selected row supplies recording identity and optional stable event IDs.
        start=bounds["start"],                                # Pose uses numeric bounds; exact events omit redundant outer bounds.
        end=bounds["end"],
        start_inclusive=inclusive["start"],                   # Named segments and whole trials share the documented inclusion policy.
        end_inclusive=inclusive["end"],
        time_coord=time_coord,                                # Supports movement's lowercase time as well as legacy Time.
    )


# ===============================================================================



# ===============================================================================
# 3| Produce One Session-Aware Stream Slice for Every Trial-Table Row
# ===============================================================================


def slice_stream_per_trial(
    stream: DataObject,  # Combined stream to slice repeatedly.
    trials: pd.DataFrame,  # Trial table or already-filtered subset, processed in row order.
    *,
    segment: str = "trial",  # Trial segment extracted for every row.
    spec: TrialSpec | None = None,  # Optional TrialSpec for segment validation/resolution.
    session_column: str = "session",  # Trial-table source-session column.
    session_coord: str = "session",  # Stream source-session coordinate/column.
    time_coord: str = "Time",  # Stream time coordinate/column/index name.
) -> list[DataObject]:  # Returns one stream subset per input trial row.
    """
    Return one stream slice per trial-table row, preserving trial row order.

    This function intentionally does NOT filter trials by outcome, mouse, day,
    condition, etc. Callers should filter the trial DataFrame first and pass the
    resulting subset here. That keeps experiment-specific scientific selection
    outside the generic datastructure machinery while still centralising the
    technically important session-aware cross-stream slicing step.

    Parameters
    ----------
    stream : DataObject
        Combined stream to slice for every supplied trial row.
    trials : pd.DataFrame
        Trial table or filtered trial subset. Rows are processed in existing
        DataFrame order and each row must satisfy ``slice_stream_for_trial``'s
        required column contract.
    segment : str
        Segment to extract for every trial. Default ``'trial'``.
    spec : TrialSpec | None
        Optional TrialSpec used to validate/resolve the segment name.
    session_column : str
        Trial-table column containing source-session identity. Default
        ``'session'``.
    session_coord : str
        Combined-stream coordinate/column containing source-session identity.
        Default ``'session'``.
    time_coord : str
        Combined-stream time coordinate/column/named index. Default ``'Time'``.

    Returns
    -------
    list[DataObject]
        One stream subset per input trial row, in exactly the same row order as
        ``trials``. An empty trial DataFrame naturally returns an empty list.
    """

    # === 1| Apply the Single-Trial Primitive Row-by-Row ===========================
    # Reusing ``slice_stream_for_trial`` keeps all segment resolution, required
    # column validation, and session-aware slicing semantics in one place rather
    # than duplicating them in a vectorised-looking wrapper.

    return [
        slice_stream_for_trial(
            stream,  # Same combined stream is queried for every trial row.
            row,  # One trial row supplies source session + time bounds.
            segment=segment,
            spec=spec,
            session_column=session_column,
            session_coord=session_coord,
            time_coord=time_coord,
        )
        for _, row in trials.iterrows()  # DataFrame iteration order defines returned slice order.
    ]


# ===============================================================================


################################################################################




# """
# Slice combined streams by source session and trial-aware time windows.
# ---------------------------------------------------------------------

# Description:
#     ``slicing.py`` contains generic cross-stream slicing operations used after
#     streams have been combined across sessions.

#     A combined stream can contain repeated time values because each session often
#     starts its own local clock at or near zero. Time alone is therefore not a
#     globally unique selector. Correct slicing must constrain BOTH:
#       * which source session contributed the data; and
#       * which time interval inside that session is required.

#     ``slice_stream`` is the primitive that performs that joint selection for
#     DataFrames and xarray streams. ``slice_stream_for_trial`` obtains the session
#     and time bounds from one row of a trial table. ``slice_stream_per_trial``
#     repeats the same operation for every row in a trial subset.

#     This module is intentionally experiment-agnostic. It understands the generic
#     trial-segment column conventions emitted by ``parse_trials`` but does not know
#     what experiment-specific names such as "outbound" or "search" mean.

# Contents:
# --------------------------------
# - _segment_columns:        Resolve one segment name to trial-table bound columns.
# - slice_stream:            Slice one combined stream by session + [start, end].
# - slice_stream_for_trial:  Slice a stream using one trial row's segment bounds.
# - slice_stream_per_trial:  Produce one stream slice per trial-table row.
# """


# ################################################################################
# # Imports
# ################################################################################

# from typing import Any

# import numpy as np
# import pandas as pd
# import xarray as xr

# from .streams import DataObject
# from .trials import TrialSpec, segment_bounds

# ################################################################################


# ################################################################################
# # Private Helpers
# ################################################################################


# # ===============================================================================
# # 1| Resolve a Trial Segment Name to its Start/End Table Columns
# # ===============================================================================


# def _segment_columns(
#     segment: str,  # Segment name requested by the caller, e.g. 'trial' or 'outbound'.
#     spec: TrialSpec | None = None,  # Optional TrialSpec used to validate/resolve segment definitions.
# ) -> tuple[str, str]:  # Returns ``(start_column, end_column)`` in the trial table.
#     """
#     Resolve a segment name into the two trial-table columns that bound it.

#     When a ``TrialSpec`` is available, ``segment_bounds(spec)`` is treated as the
#     source of truth. This validates the requested segment against the exact spec
#     that produced the table. Without a spec, the helper falls back to the generic
#     naming convention emitted by ``parse_trials``:
#     ``{segment}_start_time`` / ``{segment}_end_time``. The special whole-trial
#     segment maps to the base ``start_time`` / ``end_time`` columns.

#     Parameters
#     ----------
#     segment : str
#         Segment name to resolve. ``'trial'`` always denotes the full trial span.
#         Other values are experiment-defined names already represented in the trial
#         table, such as ``'outbound'``.
#     spec : TrialSpec | None
#         Optional TrialSpec corresponding to the trial table. If supplied, the
#         requested segment must occur in ``segment_bounds(spec)``; otherwise a
#         ``ValueError`` is raised instead of guessing a column pair.

#     Returns
#     -------
#     tuple[str, str]
#         Two-element tuple ``(start_column, end_column)`` containing the column
#         names that hold the requested segment's inclusive time bounds.
#     """

#     # === 1| Prefer the TrialSpec When Available ==================================
#     # The spec is the strongest source of truth because a future/experiment-specific
#     # spec could intentionally map a segment differently from the default naming
#     # convention. Validate before returning to catch typos early.

#     if spec is not None:
#         bounds = segment_bounds(spec)  # Build the complete legal segment->column mapping for this spec.

#         if segment not in bounds:  # Caller requested a name the supplied spec does not define.
#             raise ValueError(f"segment must be one of {sorted(bounds)}, got {segment!r}.")

#         return bounds[segment]  # Exact pair defined by the supplied TrialSpec.

#     # === 2| Handle the Whole-Trial Special Case ==================================

#     if segment == "trial":  # Base delimiting columns are always emitted by parse_trials.
#         return "start_time", "end_time"

#     # === 3| Fall Back to the Generic Segment Naming Convention ===================

#     return f"{segment}_start_time", f"{segment}_end_time"  # Matches parse_trials segment-column output convention.


# # ===============================================================================


# ################################################################################
# # Public API
# ################################################################################


# # ===============================================================================
# # 1| Slice One Combined Stream by Source Session and Inclusive Time Window
# # ===============================================================================


# def slice_stream(
#     stream: DataObject,  # Combined DataFrame/DataArray/Dataset carrying session + time identity.
#     *,
#     session: Any,  # Source-session identifier to retain.
#     start: float,  # Inclusive lower time bound within that session.
#     end: float,  # Inclusive upper time bound within that session.
#     session_coord: str = "session",  # Session coordinate/column name on the combined stream.
#     time_coord: str = "Time",  # Time coordinate/column/index name used for interval selection.
# ) -> DataObject:  # Returns same broad object type restricted to requested session/window.
#     """
#     Return one session's stream elements whose time lies within ``[start, end]``.

#     The session constraint is not optional in concept: after cross-session
#     combination, two sessions may both contain ``Time == 5.0``. A time-only slice
#     would therefore mix unrelated sessions. This helper always combines session
#     identity with time bounds in one element-wise mask.

#     Parameters
#     ----------
#     stream : DataObject
#         Combined ``DataArray``, ``Dataset``, or ``DataFrame``. It must carry
#         source-session identity plus time information. For xarray both are
#         coordinates and ``time_coord`` must also be a dimension. For DataFrames,
#         ``session_coord`` must be a column and time may be either a column or an
#         index whose name equals ``time_coord``.
#     session : Any
#         Source-session identifier to retain. The value is compared directly to
#         the stream's per-element session coordinate/column and therefore should
#         have the same value/type used by ``StreamContainer.combine``.
#     start : float
#         Inclusive lower time bound inside the selected session.
#     end : float
#         Inclusive upper time bound inside the selected session.
#     session_coord : str
#         Name of the source-session coordinate/column. Default ``'session'``.
#     time_coord : str
#         Name of the time coordinate, DataFrame column, or named DataFrame index.
#         Default ``'Time'``.

#     Returns
#     -------
#     DataObject
#         Subset of the same broad data-object type containing only elements that
#         simultaneously match the requested session and inclusive time interval.
#         NaN bounds naturally produce an empty subset because numeric comparisons
#         against NaN are False.
#     """

#     # === 1| Handle xarray Streams with an Element-Wise Boolean Mask ==============

#     if isinstance(stream, xr.DataArray | xr.Dataset):
#         # === 1.1| Validate Required Coordinates/Dimension Before Masking =========

#         if session_coord not in stream.coords:  # Combined stream must identify every element's source session.
#             raise KeyError(f"stream has no {session_coord!r} coordinate.")
#         if time_coord not in stream.coords:  # Time values must be available as an xarray coordinate.
#             raise KeyError(f"stream has no {time_coord!r} coordinate.")
#         if time_coord not in stream.dims:  # ``isel`` below selects positions along this same dimension.
#             raise ValueError(f"time coordinate {time_coord!r} is not a stream dimension.")

#         # === 1.2| Build One Joint Session + Time Mask =============================

#         sessions = stream.coords[session_coord].values  # One source-session value per element along the time dimension.
#         times = stream.coords[time_coord].values  # Local-session time values aligned with those same elements.
#         mask = (
#             (sessions == session)  # Never allow matching time values from another session through.
#             & (times >= start)  # Inclusive lower bound.
#             & (times <= end)  # Inclusive upper bound.
#         )

#         # === 1.3| Convert the Boolean Mask to Positional Indices ==================
#         # ``np.flatnonzero`` gives exactly the positions to retain while avoiding
#         # xarray alignment/broadcasting ambiguity from passing a raw NumPy mask.

#         return stream.isel({time_coord: np.flatnonzero(mask)})  # Preserves all other dimensions/coordinates on the selected elements.

#     # === 2| Handle DataFrame Streams =============================================

#     if isinstance(stream, pd.DataFrame):
#         # === 2.1| Require Explicit Per-Row Source-Session Identity ================

#         if session_coord not in stream.columns:
#             raise KeyError(f"stream has no {session_coord!r} column.")

#         # === 2.2| Resolve Time from Either a Column or Explicitly Named Index =====
#         # A named index is intentionally supported because StreamContainer retains
#         # semantically named per-session indexes during concatenation.

#         if time_coord in stream.columns:
#             times = stream[time_coord].to_numpy()  # Explicit time column takes precedence when present.
#         elif stream.index.name == time_coord:
#             times = stream.index.to_numpy()  # Named semantic index, commonly the original per-session Time index.
#         else:
#             raise KeyError(f"stream has neither a {time_coord!r} column nor an index named {time_coord!r}.")

#         # === 2.3| Build and Apply the Same Joint Session + Time Mask ===============

#         sessions = stream[session_coord].to_numpy()  # One source-session id per DataFrame row.
#         mask = (sessions == session) & (times >= start) & (times <= end)

#         return stream.loc[mask].copy()  # Copy prevents accidental mutation of the original combined DataFrame.

#     # === 3| Reject Objects Outside the DataObject Contract =======================

#     raise TypeError(f"cannot slice {type(stream).__name__}; expected DataArray / Dataset / DataFrame.")


# # ===============================================================================


# # ===============================================================================
# # 2| Slice One Combined Stream Using One Trial Row's Segment Bounds
# # ===============================================================================


# def slice_stream_for_trial(
#     stream: DataObject,  # Combined stream to restrict to one trial/segment.
#     trial_row: pd.Series,  # One row from a trial table carrying session + time bounds.
#     *,
#     segment: str = "trial",  # Segment name whose time bounds should be used.
#     spec: TrialSpec | None = None,  # Optional TrialSpec for explicit segment validation/resolution.
#     session_column: str = "session",  # Trial-table column identifying the source session.
#     session_coord: str = "session",  # Stream coordinate/column identifying source session.
#     time_coord: str = "Time",  # Stream time coordinate/column/index name.
# ) -> DataObject:  # Returns stream subset corresponding to this one trial segment.
#     """
#     Slice a combined stream to the time interval represented by one trial row.

#     Parameters
#     ----------
#     stream : DataObject
#         Combined stream to slice. Must satisfy the same session/time requirements
#         documented by ``slice_stream``.
#     trial_row : pd.Series
#         One row of a trial table. It must contain the source-session identifier and
#         the two columns that bound the requested segment.
#     segment : str
#         Trial segment to extract. ``'trial'`` uses the full trial's
#         ``start_time`` / ``end_time``. Other names use the segment mapping from
#         ``spec`` when supplied or the generic ``{segment}_start_time`` /
#         ``{segment}_end_time`` convention otherwise.
#     spec : TrialSpec | None
#         Optional TrialSpec used to validate and resolve the requested segment.
#     session_column : str
#         Column in ``trial_row`` containing the source-session id. Default
#         ``'session'``.
#     session_coord : str
#         Coordinate/column on ``stream`` containing source-session identity.
#         Default ``'session'``.
#     time_coord : str
#         Time coordinate/column/named index on ``stream``. Default ``'Time'``.

#     Returns
#     -------
#     DataObject
#         Stream subset containing only the selected trial row's source session and
#         requested segment interval.
#     """

#     # === 1| Resolve the Segment's Trial-Table Bound Columns =======================

#     start_column, end_column = _segment_columns(segment, spec)  # Either spec-defined pair or generic naming convention.

#     # === 2| Validate that the Trial Row Contains Everything Needed ================
#     # Collect all missing names and report them together; this is more useful than
#     # failing sequentially on the first Series lookup.

#     required = [session_column, start_column, end_column]
#     missing = [column for column in required if column not in trial_row.index]

#     if missing:
#         raise KeyError(f"trial row is missing required column(s): {missing}.")

#     # === 3| Delegate the Actual Session-Aware Interval Selection ==================

#     return slice_stream(
#         stream,
#         session=trial_row[session_column],  # Source session carried by the trial row after cross-session combination.
#         start=trial_row[start_column],  # Inclusive segment start time.
#         end=trial_row[end_column],  # Inclusive segment end time.
#         session_coord=session_coord,  # Stream-side session identity name may differ from trial-table column name.
#         time_coord=time_coord,  # Stream-side time representation.
#     )


# # ===============================================================================



# # ===============================================================================
# # 3| Produce One Session-Aware Stream Slice for Every Trial-Table Row
# # ===============================================================================


# def slice_stream_per_trial(
#     stream: DataObject,  # Combined stream to slice repeatedly.
#     trials: pd.DataFrame,  # Trial table or already-filtered subset, processed in row order.
#     *,
#     segment: str = "trial",  # Trial segment extracted for every row.
#     spec: TrialSpec | None = None,  # Optional TrialSpec for segment validation/resolution.
#     session_column: str = "session",  # Trial-table source-session column.
#     session_coord: str = "session",  # Stream source-session coordinate/column.
#     time_coord: str = "Time",  # Stream time coordinate/column/index name.
# ) -> list[DataObject]:  # Returns one stream subset per input trial row.
#     """
#     Return one stream slice per trial-table row, preserving trial row order.

#     This function intentionally does NOT filter trials by outcome, mouse, day,
#     condition, etc. Callers should filter the trial DataFrame first and pass the
#     resulting subset here. That keeps experiment-specific scientific selection
#     outside the generic datastructure machinery while still centralising the
#     technically important session-aware cross-stream slicing step.

#     Parameters
#     ----------
#     stream : DataObject
#         Combined stream to slice for every supplied trial row.
#     trials : pd.DataFrame
#         Trial table or filtered trial subset. Rows are processed in existing
#         DataFrame order and each row must satisfy ``slice_stream_for_trial``'s
#         required column contract.
#     segment : str
#         Segment to extract for every trial. Default ``'trial'``.
#     spec : TrialSpec | None
#         Optional TrialSpec used to validate/resolve the segment name.
#     session_column : str
#         Trial-table column containing source-session identity. Default
#         ``'session'``.
#     session_coord : str
#         Combined-stream coordinate/column containing source-session identity.
#         Default ``'session'``.
#     time_coord : str
#         Combined-stream time coordinate/column/named index. Default ``'Time'``.

#     Returns
#     -------
#     list[DataObject]
#         One stream subset per input trial row, in exactly the same row order as
#         ``trials``. An empty trial DataFrame naturally returns an empty list.
#     """

#     # === 1| Apply the Single-Trial Primitive Row-by-Row ===========================
#     # Reusing ``slice_stream_for_trial`` keeps all segment resolution, required
#     # column validation, and session-aware slicing semantics in one place rather
#     # than duplicating them in a vectorised-looking wrapper.

#     return [
#         slice_stream_for_trial(
#             stream,  # Same combined stream is queried for every trial row.
#             row,  # One trial row supplies source session + time bounds.
#             segment=segment,
#             spec=spec,
#             session_column=session_column,
#             session_coord=session_coord,
#             time_coord=time_coord,
#         )
#         for _, row in trials.iterrows()  # DataFrame iteration order defines returned slice order.
#     ]


# # ===============================================================================


# ################################################################################
