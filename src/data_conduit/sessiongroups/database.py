'''
Build SessionGroups and combine sessions into multi-session structures.
-----------------------------------------------------------------------

Description:
    Top of the workflow. Given a select_sessions result, build_sessions
    loads each selected session via the alignment pipeline (see
    alignment.py), attaches the selection's labels as metadata, and
    returns them as one ordered SessionGroup. combine_sessions then
    concatenates one chosen structure (or every structure) across the
    group, with a per-row session tag. add_label_column / drop_columns
    tailor a combined output.

    All time alignment, stacking, and indexing is delegated to the
    alignment and bundle primitives; this module is thin orchestration.
    A combined output keeps its NATIVE type: a Nosepoke stays an
    xr.DataArray queryable via .ulookup, a trial table stays a DataFrame.

Contents:
--------------------------------
- build_sessions:       Load every selected session into one SessionGroup.
- combine_sessions:     Stack one structure type (or all) across sessions.
- add_label_column:     Add a coordinate/column to a combined object.
- drop_columns:         Remove coordinates/columns from a combined object.
'''





################################################################################
# Imports
################################################################################

import numpy as np
import pandas as pd
import xarray as xr

from data_conduit.sessiongroups.alignment import (
    attach_cross_clock,
    build_global_clock,
    build_index_tables,
    load_session,
    normalise_to_zero,
)
from data_conduit.sessiongroups.bundle import combine_bundles
from data_conduit.sessiongroups.sessiongroups_core import Session, SessionGroup

################################################################################


# Constant: xarray container types we treat alike (see bundle.py for the
# rationale: both Datasets and DataArrays expose the same coords / drop /
# assign_coords API we need here).
_XARRAY_TYPES = (xr.DataArray, xr.Dataset)




################################################################################
# Build a SessionGroup From a Selection
################################################################################



#===============================================================================
# 1| build_sessions (Load + Optional Align + Wrap as SessionGroup)
#===============================================================================
def build_sessions(
        selection: dict[str, dict],
        data_structures,
        *,
        label: str | None = None,
        sort_by=('label',),
        normalise: bool = False,
        t0_from: str | float = 'session',
        time_coord: str = 'Time',
        attach_cross_clock_specs: bool = True,
        ttl_sync: dict | None = None,
        global_clock: dict | None = None,
        prefix_data_arrays: bool = True,
        verbose: bool = False,
) -> SessionGroup:
    '''
    Load every session in a selection and return them as one ordered group.

    Each session is read by the alignment pipeline (load_session, then any
    of normalise_to_zero / attach_cross_clock / build_global_clock /
    build_index_tables that have been switched on). Sidecars (global
    clock, per-member index tables, fitted TTL models) are stashed into
    each session's metadata under ``'global_clock'``, ``'index_tables'``,
    ``'ttl_models'``.

    ----------
    Parameters:
        selection (dict):
            Output of ``select_sessions``: ``{session_key: {'path': ...,
            <labels>}}``.
        data_structures (DataStructureCatalog | iterable of DataStructureSpec):
            What to extract from each session (shared across all sessions).
        label (str | None):
            Label for the returned SessionGroup as a whole.
        sort_by (sequence of str):
            Metadata keys to order the sessions by. Default ``('label',)``.
            Session names are typically timestamps, so the default already
            sorts chronologically; use ``('datetime',)`` if you attached a
            DateTimeExtractor.
        normalise (bool):
            If True, run normalise_to_zero on each session. Default False
            (times stay on their original axis).
        t0_from (str | float):
            Passed through to normalise_to_zero when ``normalise=True``.
            ``'session'`` (default) uses the earliest log across all
            time-bearing structures; a stream name uses that stream's
            earliest; a float uses the value verbatim.
        time_coord (str):
            Time coordinate name used by every alignment step.
        attach_cross_clock_specs (bool):
            If True (default), run attach_cross_clock for any spec with
            ``sync`` set. Set False to skip cross-clock structures entirely.
        ttl_sync (dict | None):
            Default kwargs forwarded to the TTL conversion (merged with
            each spec's own ``sync`` dict).
        global_clock (dict | None):
            If given, build a regular clock per session via
            build_global_clock, then per-member index tables via
            build_index_tables. Keys: ``timestep`` (required), ``start``,
            ``end``, ``reference_stream``, ``match_type`` (default
            ``'nearest'``). Both the clock and tables are written into
            ``session.metadata``.
        prefix_data_arrays (bool):
            Prefix each structure's members with its spec name. Default True.
        verbose (bool):
            If True, print per-structure load/skip lines.
    Returns:
        SessionGroup:
            The loaded sessions, ordered by ``sort_by``.
    '''

    # Accumulate one Session per entry in the selection.
    sessions: list[Session] = []
    for entry in selection.values():

        # 0| Everything in the selection entry except the path is per-session
        #    metadata (mouseID, phase, day, label, datetime, ...). Carry it
        #    onto the session for tagging, sorting, and later combine.
        metadata = {key: value for key, value in entry.items() if key != 'path'}

        # 1| Load same-clock structures into a plain Session.
        session = load_session(
            entry['path'],
            data_structures,
            metadata=metadata,
            prefix_data_arrays=prefix_data_arrays,
            verbose=verbose,
        )

        # 2| Optionally normalise to a zero-based timebase.
        if normalise:
            session = normalise_to_zero(session, time_coord=time_coord, t0_from=t0_from)

        # 3| Optionally attach cross-clock structures (Neuropixels, DLC, ...).
        #    The fitted TTL models are stashed onto metadata for provenance.
        if attach_cross_clock_specs:
            session, ttl_models = attach_cross_clock(
                session,
                data_structures,
                ttl_sync=ttl_sync,
                prefix_data_arrays=prefix_data_arrays,
                verbose=verbose,
            )
            if ttl_models:
                session.metadata['ttl_models'] = ttl_models

        # 4| Optionally build a regular clock + per-member index tables.
        #    Both are also stashed onto metadata so a downstream combine
        #    sees them per-session (they don't survive the combine, since
        #    a combined session needs a new clock anyway).
        if global_clock is not None:
            if 'timestep' not in global_clock:
                raise ValueError("global_clock config requires a 'timestep'.")
            clock = build_global_clock(
                session,
                timestep=global_clock['timestep'],
                time_coord=time_coord,
                start=global_clock.get('start'),
                end=global_clock.get('end'),
                reference_stream=global_clock.get('reference_stream', 'events'),
            )
            tables = build_index_tables(
                session,
                clock,
                time_coord=time_coord,
                match_type=global_clock.get('match_type', 'nearest'),
            )
            session.metadata['global_clock'] = clock
            session.metadata['index_tables'] = tables

        sessions.append(session)

    # 5| Order the sessions by the requested metadata keys, then wrap as a
    #    SessionGroup. The order matters: combine_sessions stacks in this
    #    order, so we want chronological by default.
    def order_key(session):
        return tuple(session.metadata.get(key) for key in sort_by)

    group = SessionGroup(sessions, label=label)
    return group.sorted(order_key)

#===============================================================================



################################################################################




################################################################################
# Combine a Data-Structure Type Across Sessions
################################################################################



#===============================================================================
# 1| combine_sessions (Stack One Type or All Types Across Sessions)
#===============================================================================
def combine_sessions(
        group: SessionGroup,
        *,
        dim: str = 'Time',
        type_name: str | None = None,
        label_coord: str = 'label',
        on: str = 'intersection',
):
    '''
    Concatenate sessions into one multi-session structure, tagged by session.

    Sessions are stacked in the group's order (set by ``build_sessions``),
    so the result is sorted first by session label, then by within-session
    time order (the brief's required ordering, achieved for free by
    combining ordered sessions. Each row / sample gains a ``label_coord``
    tag (a column for tables, a coordinate for xarray) identifying its
    source session.

    ----------
    Parameters:
        group (SessionGroup):
            The sessions to combine (already ordered).
        dim (str):
            Alignment axis to stack xarray members along (default
            ``'Time'``). DataFrame members are stacked row-wise regardless.
        type_name (str | None):
            If given, combine only this ONE member (e.g.
            ``'nosepoke:Activations'``) and return that single combined
            object (the "multi-session Nosepoke"). If None, combine every
            member and return a combined Session.
        label_coord (str):
            Name of the per-row / sample session tag. Default ``'label'``.
            Tag values default to each session's ``metadata[label_coord]``.
        on (str):
            Member-name reconciliation across sessions (see
            ``combine_bundles``). One of ``'intersection'`` (default.
            keep members present in every session) or ``'strict'``.
    Returns:
        xr.DataArray | xr.Dataset | pd.DataFrame  (if ``type_name`` given)
        or Session  (if ``type_name`` is None):
            The combined multi-session structure.
    '''

    # 1| Materialise the group (it may be lazy) and reject empty input early.
    sessions = list(group)
    if not sessions:
        raise ValueError('cannot combine an empty SessionGroup.')

    # 2| Default the per-session tag values from metadata (fall back to position
    #    if a session has no value under label_coord). These become the
    #    provenance labels written along the alignment axis.
    keys = [s.metadata.get(label_coord, index) for index, s in enumerate(sessions)]

    # 3| Whole-session combine: stack every member, return a Session. The
    #    SessionGroup.combine method already does exactly this (and records
    #    group-level metadata for split-ability), so we defer to it.
    if type_name is None:
        return group.combine(dim=dim, keys=keys, label_coord=label_coord, on=on)

    # 4| Single-type combine: stack just one member, return that object.
    #    Keep only the sessions that actually have this member (others are
    #    skipped, which matters when an optional structure is missing from
    #    some sessions).
    single_bundles = []
    single_keys = []
    for key, session in zip(keys, sessions, strict=True):
        if type_name in session.data:
            single_bundles.append({type_name: session.data[type_name]})
            single_keys.append(key)

    if not single_bundles:
        raise KeyError(f'no session in the group has a member named {type_name!r}.')

    # 5| Delegate the stacking to combine_bundles, then unpack the one member.
    combined = combine_bundles(
        single_bundles, dim=dim, keys=single_keys, label_coord=label_coord, on=on,
    )
    return combined[type_name]

#===============================================================================



################################################################################




################################################################################
# Small Helpers for Tailoring a Combined Table (Task 4d)
################################################################################



#===============================================================================
# 1| add_label_column (Add a Coordinate / Column to a Combined Object)
#===============================================================================
def add_label_column(
        obj,
        name: str,
        value_or_fn,
        *,
        dim: str = 'Time',
):
    '''
    Add an extra label to a combined object: a coordinate (xarray) or column
    (DataFrame).

    Use this to tag rows / samples with a derived grouping (e.g. a "block"
    or "condition") computed from criteria, on top of the session label.

    ----------
    Parameters:
        obj (xr.DataArray | xr.Dataset | pd.DataFrame):
            The combined object to label.
        name (str):
            Name of the new coordinate / column.
        value_or_fn (scalar | callable):
            Either a single value broadcast to every element, or a
            callable ``fn(obj) -> array`` producing one value per element
            along ``dim`` (a "criterion", e.g.
            ``lambda df: df['outcome'].eq('Success')``).
        dim (str):
            Alignment axis the label runs along, for xarray members.
            Default ``'Time'``.
    Returns:
        Same type as ``obj``:
            A new object with the label attached; ``obj`` is not mutated.
    '''

    # 1| Work out the per-element values: call the criterion to get one value
    #    per element, or keep the scalar as-is (broadcast where needed below).
    values = np.asarray(value_or_fn(obj)) if callable(value_or_fn) else value_or_fn

    # 2| DataFrame branch: assigning a column broadcasts a scalar automatically;
    #    an array must match the number of rows.
    if isinstance(obj, pd.DataFrame):
        labelled = obj.copy()
        labelled[name] = values
        return labelled

    # 3| xarray branch: a scalar becomes a length-``dim`` array so it sits
    #    along the axis (rather than a single scalar coordinate), matching
    #    the DataFrame-column feel.
    if isinstance(obj, _XARRAY_TYPES):
        if not callable(value_or_fn):
            values = np.repeat(value_or_fn, obj.sizes[dim])
        return obj.assign_coords({name: (dim, np.asarray(values))})

    raise TypeError(f'cannot add a label to a {type(obj).__name__}.')

#===============================================================================



#===============================================================================
# 2| drop_columns (Remove Coordinates / Columns From a Combined Object)
#===============================================================================
def drop_columns(
        obj,
        names,
):
    '''
    Drop coordinates / columns from a combined object by name.

    The mirror of ``add_label_column``. Remove tags / columns you don't
    want in a new datatable. Works on a DataFrame (drops columns) or an
    xarray object (drops non-dimension coordinates / data variables).
    Missing names are ignored.

    ----------
    Parameters:
        obj (xr.DataArray | xr.Dataset | pd.DataFrame):
            The combined object.
        names (str | sequence of str):
            Coordinate / column name(s) to drop. A single string is
            accepted as a one-element list.
    Returns:
        Same type as ``obj``:
            A new object without the named coords / columns.
    '''

    # 1| Accept a single name or a list, and work with a list internally.
    name_list = [names] if isinstance(names, str) else list(names)

    # 2| DataFrame branch: drop the named columns; errors='ignore' so missing
    #    names are no-ops rather than KeyError.
    if isinstance(obj, pd.DataFrame):
        return obj.drop(columns=name_list, errors='ignore')

    # 3| xarray branch: drop_vars handles both non-dim coords and data vars,
    #    again ignoring missing names.
    if isinstance(obj, _XARRAY_TYPES):
        return obj.drop_vars(name_list, errors='ignore')

    raise TypeError(f'cannot drop columns from a {type(obj).__name__}.')

#===============================================================================



################################################################################
