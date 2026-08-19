"""
Slice combined streams by source session and trial-aware time windows.
---------------------------------------------------------------------

Description:
    ``slicing.py`` contains generic cross-stream slicing operations used after
    streams have been combined across sessions.

    A combined stream can contain repeated time values because each session often
    starts its own local clock at or near zero. Time alone is therefore not a
    globally unique selector. Correct slicing must constrain BOTH:
      * which source session contributed the data; and
      * which time interval inside that session is required.

    ``slice_stream`` is the primitive that performs that joint selection for
    DataFrames and xarray streams. ``slice_stream_for_trial`` obtains the session
    and time bounds from one row of a trial table. ``slice_stream_per_trial``
    repeats the same operation for every row in a trial subset.

    This module is intentionally experiment-agnostic. It understands the generic
    trial-segment column conventions emitted by ``parse_trials`` but does not know
    what experiment-specific names such as "outbound" or "search" mean.

Contents:
--------------------------------
- _segment_columns:        Resolve one segment name to trial-table bound columns.
- slice_stream:            Slice one combined stream by session + [start, end].
- slice_stream_for_trial:  Slice a stream using one trial row's segment bounds.
- slice_stream_per_trial:  Produce one stream slice per trial-table row.
"""


################################################################################
# Imports
################################################################################

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
        names that hold the requested segment's inclusive time bounds.
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


################################################################################
# Public API
################################################################################


# ===============================================================================
# 1| Slice One Combined Stream by Source Session and Inclusive Time Window
# ===============================================================================


def slice_stream(
    stream: DataObject,  # Combined DataFrame/DataArray/Dataset carrying session + time identity.
    *,
    session: Any,  # Source-session identifier to retain.
    start: float,  # Inclusive lower time bound within that session.
    end: float,  # Inclusive upper time bound within that session.
    session_coord: str = "session",  # Session coordinate/column name on the combined stream.
    time_coord: str = "Time",  # Time coordinate/column/index name used for interval selection.
) -> DataObject:  # Returns same broad object type restricted to requested session/window.
    """
    Return one session's stream elements whose time lies within ``[start, end]``.

    The session constraint is not optional in concept: after cross-session
    combination, two sessions may both contain ``Time == 5.0``. A time-only slice
    would therefore mix unrelated sessions. This helper always combines session
    identity with time bounds in one element-wise mask.

    Parameters
    ----------
    stream : DataObject
        Combined ``DataArray``, ``Dataset``, or ``DataFrame``. It must carry
        source-session identity plus time information. For xarray both are
        coordinates and ``time_coord`` must also be a dimension. For DataFrames,
        ``session_coord`` must be a column and time may be either a column or an
        index whose name equals ``time_coord``.
    session : Any
        Source-session identifier to retain. The value is compared directly to
        the stream's per-element session coordinate/column and therefore should
        have the same value/type used by ``StreamContainer.combine``.
    start : float
        Inclusive lower time bound inside the selected session.
    end : float
        Inclusive upper time bound inside the selected session.
    session_coord : str
        Name of the source-session coordinate/column. Default ``'session'``.
    time_coord : str
        Name of the time coordinate, DataFrame column, or named DataFrame index.
        Default ``'Time'``.

    Returns
    -------
    DataObject
        Subset of the same broad data-object type containing only elements that
        simultaneously match the requested session and inclusive time interval.
        NaN bounds naturally produce an empty subset because numeric comparisons
        against NaN are False.
    """

    # === 1| Handle xarray Streams with an Element-Wise Boolean Mask ==============

    if isinstance(stream, xr.DataArray | xr.Dataset):
        # === 1.1| Validate Required Coordinates/Dimension Before Masking =========

        if session_coord not in stream.coords:  # Combined stream must identify every element's source session.
            raise KeyError(f"stream has no {session_coord!r} coordinate.")
        if time_coord not in stream.coords:  # Time values must be available as an xarray coordinate.
            raise KeyError(f"stream has no {time_coord!r} coordinate.")
        if time_coord not in stream.dims:  # ``isel`` below selects positions along this same dimension.
            raise ValueError(f"time coordinate {time_coord!r} is not a stream dimension.")

        # === 1.2| Build One Joint Session + Time Mask =============================

        sessions = stream.coords[session_coord].values  # One source-session value per element along the time dimension.
        times = stream.coords[time_coord].values  # Local-session time values aligned with those same elements.
        mask = (
            (sessions == session)  # Never allow matching time values from another session through.
            & (times >= start)  # Inclusive lower bound.
            & (times <= end)  # Inclusive upper bound.
        )

        # === 1.3| Convert the Boolean Mask to Positional Indices ==================
        # ``np.flatnonzero`` gives exactly the positions to retain while avoiding
        # xarray alignment/broadcasting ambiguity from passing a raw NumPy mask.

        return stream.isel({time_coord: np.flatnonzero(mask)})  # Preserves all other dimensions/coordinates on the selected elements.

    # === 2| Handle DataFrame Streams =============================================

    if isinstance(stream, pd.DataFrame):
        # === 2.1| Require Explicit Per-Row Source-Session Identity ================

        if session_coord not in stream.columns:
            raise KeyError(f"stream has no {session_coord!r} column.")

        # === 2.2| Resolve Time from Either a Column or Explicitly Named Index =====
        # A named index is intentionally supported because StreamContainer retains
        # semantically named per-session indexes during concatenation.

        if time_coord in stream.columns:
            times = stream[time_coord].to_numpy()  # Explicit time column takes precedence when present.
        elif stream.index.name == time_coord:
            times = stream.index.to_numpy()  # Named semantic index, commonly the original per-session Time index.
        else:
            raise KeyError(f"stream has neither a {time_coord!r} column nor an index named {time_coord!r}.")

        # === 2.3| Build and Apply the Same Joint Session + Time Mask ===============

        sessions = stream[session_coord].to_numpy()  # One source-session id per DataFrame row.
        mask = (sessions == session) & (times >= start) & (times <= end)

        return stream.loc[mask].copy()  # Copy prevents accidental mutation of the original combined DataFrame.

    # === 3| Reject Objects Outside the DataObject Contract =======================

    raise TypeError(f"cannot slice {type(stream).__name__}; expected DataArray / Dataset / DataFrame.")


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
        Combined stream to slice. Must satisfy the same session/time requirements
        documented by ``slice_stream``.
    trial_row : pd.Series
        One row of a trial table. It must contain the source-session identifier and
        the two columns that bound the requested segment.
    segment : str
        Trial segment to extract. ``'trial'`` uses the full trial's
        ``start_time`` / ``end_time``. Other names use the segment mapping from
        ``spec`` when supplied or the generic ``{segment}_start_time`` /
        ``{segment}_end_time`` convention otherwise.
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
        requested segment interval.
    """

    # === 1| Resolve the Segment's Trial-Table Bound Columns =======================

    start_column, end_column = _segment_columns(segment, spec)  # Either spec-defined pair or generic naming convention.

    # === 2| Validate that the Trial Row Contains Everything Needed ================
    # Collect all missing names and report them together; this is more useful than
    # failing sequentially on the first Series lookup.

    required = [session_column, start_column, end_column]
    missing = [column for column in required if column not in trial_row.index]

    if missing:
        raise KeyError(f"trial row is missing required column(s): {missing}.")

    # === 3| Delegate the Actual Session-Aware Interval Selection ==================

    return slice_stream(
        stream,
        session=trial_row[session_column],  # Source session carried by the trial row after cross-session combination.
        start=trial_row[start_column],  # Inclusive segment start time.
        end=trial_row[end_column],  # Inclusive segment end time.
        session_coord=session_coord,  # Stream-side session identity name may differ from trial-table column name.
        time_coord=time_coord,  # Stream-side time representation.
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
