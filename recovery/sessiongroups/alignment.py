'''
Alignment pipeline for sessions.
--------------------------------

Description:
    A set of small composable functions that transform a Session: loading from
    disk via a DataStructureCatalog, normalising to a zero-based timebase,
    attaching cross-clock structures via TTL conversion, and building an
    optional regular global clock with per-member index tables.

    Each function returns a new Session (or a sidecar) and is fully optional,
    so a caller assembles only the steps they need. Nothing is stored on a
    session subclass; alignment state lives in Session.metadata or in returned
    sidecars (e.g. fitted TTL models, the global clock vector, index tables).

Contents:
--------------------------------
- load_session:         Read every enabled same-clock spec into a Session.
- normalise_to_zero:    Shift every time-bearing member to start at t=0.
- attach_cross_clock:   Load cross-clock specs, fit TTL, merge into a Session.
- build_global_clock:   Regular time vector spanning the reference stream.
- build_index_tables:   Per-member index tables onto a global clock.
'''





################################################################################
# Imports
################################################################################

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data_conduit.globaltimes.globaltimes_core import create_global_clock, index_map_util
from data_conduit.sessiongroups.builders import _object_to_bundle
from data_conduit.sessiongroups.data_structures import DataStructureCatalog
from data_conduit.sessiongroups.sessiongroups_core import Session

################################################################################


# Constant: xarray container types treated alike when reading or shifting a
# time axis. A Dataset is a collection of DataArrays sharing coordinates; both
# expose .coords / .assign_coords, so the same code path handles them.
_XARRAY_TYPES = (xr.DataArray, xr.Dataset)




################################################################################
# Private Helpers
################################################################################



#===============================================================================
# 1| Read a Member's Time Axis
#===============================================================================
def _member_times(
        obj,
        time_coord: str,
) -> np.ndarray | None:
    '''
    Return a member's 1D time values, or None if it carries no time axis.

    A bundle member can be an xarray object (time lives as a coordinate) or a
    DataFrame (time is usually the index, occasionally a column). Some members
    have no time at all (e.g. a settings table); those return None so callers
    can skip them.

    ----------
    Parameters:
        obj (xr.DataArray | xr.Dataset | pd.DataFrame):
            The bundle member to inspect.
        time_coord (str):
            Name of the time coordinate (for xarray) or the named index/column
            (for DataFrame) to look up.
    Returns:
        np.ndarray | None:
            The 1D time values if the axis is present, otherwise None.
    '''

    # xarray case: time is a named coordinate. If present, return its values.
    if isinstance(obj, _XARRAY_TYPES):
        if time_coord in obj.coords:
            return np.asarray(obj.coords[time_coord].values)
        return None

    # DataFrame case: time is usually the named index, occasionally a column.
    if isinstance(obj, pd.DataFrame):
        if obj.index.name == time_coord:
            return np.asarray(obj.index.values)
        if time_coord in obj.columns:
            return np.asarray(obj[time_coord].values)
        return None

    # Anything else (e.g. a SessionSettings scalar) has no recognisable time axis.
    return None

#===============================================================================



#===============================================================================
# 2| Shift a Member's Time Axis
#===============================================================================
def _shift_time(
        obj,
        offset: float,
        time_coord: str,
):
    '''
    Return a copy of obj with offset subtracted from its time axis.

    Used to normalise a session so its earliest log sits at t = 0 (callers
    pass offset = t0). Members with no time axis are returned unchanged.

    ----------
    Parameters:
        obj (xr.DataArray | xr.Dataset | pd.DataFrame):
            The bundle member whose time axis we are shifting.
        offset (float):
            The value to subtract from every time entry (typically the
            session's t0).
        time_coord (str):
            Name of the time coordinate / index / column.
    Returns:
        xr.DataArray | xr.Dataset | pd.DataFrame:
            A new object with the shifted time axis, or the original object
            unchanged if no time axis was found.
    '''

    # xarray case: rebuild the time coordinate shifted by the offset. The
    # assign_coords call returns a new object so the input is not mutated.
    if isinstance(obj, _XARRAY_TYPES):
        if time_coord in obj.coords:
            return obj.assign_coords({time_coord: obj.coords[time_coord] - offset})
        return obj

    # DataFrame case: copy first, then shift whichever of the index/column is
    # the time axis. The copy ensures we never mutate the caller's frame.
    if isinstance(obj, pd.DataFrame):
        if obj.index.name == time_coord:
            shifted = obj.copy()
            shifted.index = shifted.index - offset
            return shifted
        if time_coord in obj.columns:
            shifted = obj.copy()
            shifted[time_coord] = shifted[time_coord] - offset
            return shifted
        return obj

    # No recognised type: return unchanged.
    return obj

#===============================================================================



#===============================================================================
# 3| Resolve Specs from a Catalog or Iterable
#===============================================================================
def _resolve_specs(
        data_structures,
) -> list:
    '''
    Return the enabled specs from a catalog or a plain iterable of specs.

    Accepts either a DataStructureCatalog (we ask it for ``enabled_specs``) or
    any iterable of DataStructureSpec (we filter on ``enabled`` ourselves), so
    callers can pass whichever is convenient.

    ----------
    Parameters:
        data_structures (DataStructureCatalog | iterable of DataStructureSpec):
            The catalog or list of specs to resolve.
    Returns:
        list:
            The enabled DataStructureSpec objects, in catalog order.
    '''

    # Catalog case: defer to the catalog's own enabled_specs accessor (which
    # respects insertion order and the enabled flag).
    if isinstance(data_structures, DataStructureCatalog):
        return data_structures.enabled_specs()

    # Iterable case: filter on the ``enabled`` attribute, defaulting to True so
    # plain spec lists without an enabled flag are accepted.
    return [spec for spec in data_structures if getattr(spec, 'enabled', True)]

#===============================================================================



#===============================================================================
# 4| Earliest Time Across Members
#===============================================================================
def _earliest_time(
        objs,
        time_coord: str,
) -> float | None:
    '''
    Return the minimum time over the given members, or None if none have one.

    Used to define t0 = the session's earliest log across all time-bearing
    members (so nothing ends up negative after normalisation).

    ----------
    Parameters:
        objs (iterable):
            The bundle members to scan.
        time_coord (str):
            Name of the time coordinate / index / column to look up.
    Returns:
        float | None:
            The minimum time value across all members that carry a time axis,
            or None if no member carries one.
    '''

    # Track the running minimum. Start at None so we can distinguish "no times
    # found anywhere" from "earliest is zero".
    earliest = None

    # Scan each member, skipping those with no time axis or empty times.
    for obj in objs:
        times = _member_times(obj, time_coord)
        if times is not None and times.size:
            candidate = float(np.min(times))
            earliest = candidate if earliest is None else min(earliest, candidate)

    return earliest

#===============================================================================



#===============================================================================
# 5| Reference-Stream Member Names
#===============================================================================
def _reference_members(
        data: dict,
        reference_stream: str,
) -> list[str]:
    '''
    Return the bundle member names belonging to the reference stream.

    A spec named ``'events'`` may appear in the bundle as the bare member
    ``'events'`` (a DataFrame) or as one or more prefixed members like
    ``'events:<sub>'`` (xarray data_arrays), so we match both shapes.

    ----------
    Parameters:
        data (dict):
            The bundle (mapping member name to xarray/DataFrame object).
        reference_stream (str):
            The spec name whose members we want to find.
    Returns:
        list[str]:
            The member names that belong to the named stream, in dict order.
    '''

    # Build the prefix once; iterate the dict and match either form.
    prefix = f'{reference_stream}:'
    return [name for name in data if name == reference_stream or name.startswith(prefix)]

#===============================================================================



#===============================================================================
# 6| Load One Same-Clock Spec into Bundle Members
#===============================================================================
def _load_spec_members(
        spec,
        session_path,
        *,
        prefix_data_arrays: bool,
        verbose: bool,
) -> dict:
    '''
    Load one same-clock spec into bundle members ({} if skipped).

    Honours spec.required: a non-required structure whose data is missing or
    unreadable is skipped with a warning; a required one that fails is
    re-raised. A structure that loaded but produced nothing is treated the
    same way (skipped with a warning, or raised if required).

    ----------
    Parameters:
        spec (DataStructureSpec):
            The spec to load. Its ``reader`` is called with ``session_path``;
            its ``name`` becomes the member prefix; ``required`` controls the
            missing-data behaviour.
        session_path (Path):
            The session directory to pass to the spec's reader.
        prefix_data_arrays (bool):
            If True, prefix each member key with ``spec.name`` (e.g. an
            ``Activations`` array becomes ``nosepoke:Activations``).
        verbose (bool):
            If True, print a line indicating what was loaded or skipped.
    Returns:
        dict:
            The mapping of (prefixed) member name to data object. Empty if
            the spec was skipped.
    '''

    # 1| Call the spec's reader. Wrap so missing/unreadable optional structures
    #    just warn and return {}; required failures re-raise.
    try:
        obj = spec.reader(session_path)
        members = _object_to_bundle(obj, spec.name, prefix_data_arrays=prefix_data_arrays)
    except Exception as exc:
        if spec.required:
            raise
        warnings.warn(
            f'skipping data structure {spec.name!r}: could not load ({exc})',
            stacklevel=2,
        )
        return {}

    # 2| Structure read OK but produced no members (e.g. empty .data_arrays):
    #    raise if required, otherwise skip and (optionally) report.
    if not members:
        if spec.required:
            raise ValueError(
                f'required data structure {spec.name!r} produced no data at {session_path}.'
            )
        if verbose:
            print(f'[alignment] {spec.name!r} produced no members; skipping.')
        return {}

    # 3| Success: optionally report what we loaded, then return the members.
    if verbose:
        print(f'[alignment] loaded {spec.name!r}: {list(members)}')
    return members

#===============================================================================



#===============================================================================
# 7| Apply a Fitted TTL Model to a Stream's Time Axis
#===============================================================================
def _apply_time_model(
        data,
        model,
        time_column: str,
):
    '''
    Return ``data`` with its time axis converted by a fitted TTL ``model``.

    ``model.transform(times)`` maps a structure's own clock onto the reference
    clock. Used by attach_cross_clock to bring a cross-clock structure
    (Neuropixels, DLC) onto the session's reference timebase before merging.

    ----------
    Parameters:
        data (pd.DataFrame | xr.DataArray | xr.Dataset):
            The data object whose time axis we are converting.
        model (object):
            A fitted conversion with a ``.transform(times) -> times`` method.
        time_column (str):
            Name of the time column / index (DataFrame) or coordinate (xarray)
            to convert.
    Returns:
        pd.DataFrame | xr.DataArray | xr.Dataset:
            A new object with the converted time axis, or the original
            unchanged if no recognised time axis was found.
    '''

    # DataFrame case: transform the column or the index in a copy.
    if isinstance(data, pd.DataFrame):
        shifted = data.copy()
        if time_column in shifted.columns:
            shifted[time_column] = model.transform(np.asarray(shifted[time_column].values))
        elif shifted.index.name == time_column:
            shifted.index = model.transform(np.asarray(shifted.index.values))
        return shifted

    # xarray case: transform the named coordinate and re-attach via assign_coords.
    if isinstance(data, _XARRAY_TYPES) and time_column in data.coords:
        converted = model.transform(np.asarray(data.coords[time_column].values))
        return data.assign_coords({time_column: converted})

    # Anything else: pass through unchanged.
    return data

#===============================================================================



################################################################################




################################################################################
# Alignment Pipeline (Public)
################################################################################



#===============================================================================
# 1| Load a Session from Disk
#===============================================================================
def load_session(
        session_path: str | Path,
        data_structures,
        *,
        metadata: dict | None = None,
        prefix_data_arrays: bool = True,
        verbose: bool = False,
) -> Session:
    '''
    Walk a catalog, load every enabled same-clock spec, return a plain Session.

    Performs no time normalisation and no cross-clock conversion: each member
    keeps its original time axis. Use ``normalise_to_zero`` /
    ``attach_cross_clock`` / ``build_global_clock`` afterwards as needed.

    ----------
    Parameters:
        session_path (str | Path):
            The session directory to load from. Passed to each spec's reader.
        data_structures (DataStructureCatalog | iterable of DataStructureSpec):
            What to extract. Only enabled, same-clock specs are loaded here;
            cross-clock specs (those with ``spec.sync`` set) are deferred to
            ``attach_cross_clock``, which reads their config from the same
            catalog.
        metadata (dict | None):
            Extra metadata to attach to the returned Session. The session_path
            is added under ``'session_path'`` automatically.
        prefix_data_arrays (bool):
            If True (default), prefix each structure's members with
            ``spec.name`` so a Nosepoke's ``Activations`` array becomes
            ``'nosepoke:Activations'``. Set False to keep the bare names.
        verbose (bool):
            If True, print a line per loaded / skipped structure.
    Returns:
        Session:
            (data, metadata) pair with the loaded same-clock bundle and the
            caller's metadata extended with ``'session_path'``.
    '''

    # 1| Normalise the path so all downstream code can rely on a Path object.
    session_path = Path(session_path)

    # 2| Resolve specs to the enabled ones, then split off the same-clock specs
    #    (cross-clock specs are loaded later, by attach_cross_clock).
    specs = _resolve_specs(data_structures)
    same_clock_specs = [spec for spec in specs if spec.sync is None]

    # 3| Load each same-clock spec and merge its members into one bundle.
    bundle: dict = {}
    for spec in same_clock_specs:
        bundle.update(
            _load_spec_members(
                spec, session_path,
                prefix_data_arrays=prefix_data_arrays,
                verbose=verbose,
            )
        )

    # 4| Build the metadata (callers' fields + session_path) and return.
    full_metadata = dict(metadata or {})
    full_metadata['session_path'] = str(session_path)
    return Session(bundle, full_metadata)

#===============================================================================



#===============================================================================
# 2| Normalise a Session to a Zero-Based Timebase
#===============================================================================
def normalise_to_zero(
        session: Session,
        *,
        time_coord: str = 'Time',
        t0_from: str | float = 'session',
) -> Session:
    '''
    Return a new Session with ``time_coord`` shifted so the reference starts at 0.

    The chosen t0 is also recorded in ``metadata['t0']`` for provenance and to
    let callers recover absolute times later (``absolute = relative + t0``).

    ----------
    Parameters:
        session (Session):
            The input session. Not mutated; a new Session is returned.
        time_coord (str):
            Name of the time coordinate (for xarray members) or index/column
            (for DataFrame members) to shift. Default ``'Time'``.
        t0_from (str | float):
            Source for t0:
              * a number      -> use it verbatim (no scan over the bundle).
              * 'session'     -> earliest time across all time-bearing members.
              * any other str -> earliest time of that named stream/spec only.
    Returns:
        Session:
            A new Session with every time-bearing member shifted by t0, and
            ``metadata['t0']`` / ``metadata['time_coord']`` populated. If t0
            could not be resolved (no time-bearing members or named stream
            absent) the original session's data is returned unchanged and a
            warning is emitted.
    '''

    # 1| Resolve t0 from the requested source. Three branches: literal number,
    #    'session' (global earliest), or any other string (named stream).
    if isinstance(t0_from, int | float) and not isinstance(t0_from, bool):
        t0 = float(t0_from)
    elif t0_from == 'session':
        t0 = _earliest_time(session.data.values(), time_coord)
        if t0 is None:
            warnings.warn(
                'no time-bearing members found; cannot normalise to t=0.',
                stacklevel=2,
            )
    else:
        # Named stream: pick the members belonging to that stream and find the
        # earliest time among them.
        ref_names = _reference_members(session.data, t0_from)
        if not ref_names:
            warnings.warn(
                f'stream {t0_from!r} not found; cannot normalise to t=0.',
                stacklevel=2,
            )
            t0 = None
        else:
            t0 = _earliest_time((session.data[name] for name in ref_names), time_coord)
            if t0 is None:
                warnings.warn(
                    f'stream {t0_from!r} has no {time_coord!r} axis; cannot normalise.',
                    stacklevel=2,
                )

    # 2| If t0 could not be resolved, return a shallow copy of the input
    #    unchanged (so callers can still chain into the next pipeline step).
    if t0 is None:
        return Session(dict(session.data), dict(session.metadata))

    # 3| Apply the SAME t0 shift to every member so they stay mutually aligned.
    shifted = {name: _shift_time(obj, t0, time_coord) for name, obj in session.data.items()}

    # 4| Record provenance in metadata and return the new Session.
    new_metadata = dict(session.metadata)
    new_metadata['t0'] = t0
    new_metadata['time_coord'] = time_coord
    return Session(shifted, new_metadata)

#===============================================================================



#===============================================================================
# 3| Attach Cross-Clock Structures via TTL Conversion
#===============================================================================
def attach_cross_clock(
        session: Session,
        data_structures,
        *,
        session_path: str | Path | None = None,
        ttl_sync: dict | None = None,
        prefix_data_arrays: bool = True,
        verbose: bool = False,
) -> tuple[Session, dict[str, object]]:
    '''
    Load every cross-clock spec, fit its TTL conversion, return a merged Session.

    A cross-clock spec carries a ``sync`` dict whose ``'reference_pulses'``
    entry is either a pulse-table DataFrame or a ``callable(bundle) ->
    DataFrame``. The spec's reader must return ``{'pulse_table', 'data',
    'time_column'}``. The fitted TTL conversion is applied to the data's time
    axis BEFORE merging; if ``metadata['t0']`` exists (i.e. the session was
    normalised), the converted times are then shifted by t0 so they sit on the
    same zero-based timebase as the rest of the bundle.

    ----------
    Parameters:
        session (Session):
            The session to merge cross-clock members into. Its bundle is also
            passed to any ``reference_pulses`` callable, and its
            ``metadata['t0']`` (if set) is applied to the converted data.
        data_structures (DataStructureCatalog | iterable of DataStructureSpec):
            The catalog or list. Specs with ``spec.sync`` set are processed
            here; same-clock specs are ignored.
        session_path (str | Path | None):
            Disk path passed to each spec's reader. Defaults to
            ``metadata['session_path']`` if not given. Raises if neither is
            available.
        ttl_sync (dict | None):
            Default kwargs forwarded to the TTL conversion, merged with each
            spec's own ``sync`` dict (spec-level keys win on collision).
        prefix_data_arrays (bool):
            If True (default), prefix each cross-clock member key with
            ``spec.name``. Set False to keep the bare names.
        verbose (bool):
            If True, print a line per loaded / skipped cross-clock spec.
    Returns:
        tuple[Session, dict[str, object]]:
            * A new Session whose bundle is the original plus the converted
              cross-clock members; metadata is copied unchanged.
            * A dict mapping each cross-clock spec name to its fitted TTL
              conversion model. Empty if no cross-clock specs were processed.
              The caller decides whether to cache or discard.
    '''

    # 0| Lazy import: only callers with cross-clock specs need the TTL machinery,
    #    so we keep the import out of the module-level dependency chain.
    from data_conduit.sync.ttl.ttlsync_core import get_ttl_timebase_conversion

    # 1| Resolve the session path. The caller can supply one explicitly, or it
    #    can be picked up from the session's metadata; otherwise this is fatal.
    if session_path is None:
        session_path = session.metadata.get('session_path')
        if session_path is None:
            raise ValueError(
                'session_path is required (no session_path in session.metadata).'
            )
    session_path = Path(session_path)

    # 2| Pick out cross-clock specs (those with a sync dict).
    specs = _resolve_specs(data_structures)
    cross_clock_specs = [spec for spec in specs if spec.sync is not None]

    # 3| Cache t0 once: if the session was normalised, we apply the same shift
    #    to the converted cross-clock data so it sits on the same timebase.
    t0 = session.metadata.get('t0')

    # 4| Start from a copy of the current bundle so the input session is not
    #    mutated, and an empty dict for the fitted TTL models we will return.
    new_bundle = dict(session.data)
    ttl_models: dict[str, object] = {}

    # 5| Process each cross-clock spec in turn.
    for spec in cross_clock_specs:

        # 5a| Load the structure: a {'pulse_table', 'data', 'time_column'} dict.
        try:
            result = spec.reader(session_path)
        except Exception as exc:
            if spec.required:
                raise
            warnings.warn(
                f'skipping cross-clock structure {spec.name!r}: could not load ({exc})',
                stacklevel=2,
            )
            continue

        # 5b| Resolve reference pulses. Either supplied directly as a DataFrame
        #     or as a callable that derives it from the already-loaded bundle.
        reference_pulses = spec.sync.get('reference_pulses')
        if callable(reference_pulses):
            reference_pulses = reference_pulses(session.data)
        if reference_pulses is None:
            raise ValueError(
                f"cross-clock structure {spec.name!r} needs sync['reference_pulses']."
            )

        # 5c| Fit the linear time conversion (structure clock -> reference clock).
        #     Spec-level sync kwargs override the global ttl_sync defaults.
        ttl_kwargs = {**(ttl_sync or {})}
        ttl_kwargs.update({k: v for k, v in spec.sync.items() if k != 'reference_pulses'})
        conversion = get_ttl_timebase_conversion(
            reference_pulses=reference_pulses,
            target_pulses=result['pulse_table'],
            return_details=True,
            **ttl_kwargs,
        )
        # get_ttl_timebase_conversion(..., return_details=True) returns the
        # tuple (df, ratio, model, stats); we only need the model object here.
        model = conversion[2]
        ttl_models[spec.name] = model

        # 5d| Convert the data's own times, then apply the session's t0 shift
        #     if one exists, so the new members share the reference timebase.
        data = _apply_time_model(result['data'], model, result['time_column'])
        if t0 is not None:
            data = _shift_time(data, t0, result['time_column'])

        # 5e| Turn the converted data into prefixed bundle members and merge.
        members = _object_to_bundle(data, spec.name, prefix_data_arrays=prefix_data_arrays)
        new_bundle.update(members)
        if verbose:
            print(f'[alignment] loaded cross-clock {spec.name!r}: {list(members)}')

    # 6| Return the merged Session and the dict of fitted models.
    return Session(new_bundle, dict(session.metadata)), ttl_models

#===============================================================================



#===============================================================================
# 4| Build a Regular Global Clock
#===============================================================================
def build_global_clock(
        session: Session,
        *,
        timestep: float,
        time_coord: str = 'Time',
        start: float | None = None,
        end: float | None = None,
        reference_stream: str = 'events',
) -> np.ndarray:
    '''
    Build a regular time vector spanning the reference stream's extent.

    The clock is the shared "ruler" against which per-member sample positions
    can later be resolved by ``build_index_tables``. Defaults are chosen so a
    normalised session (start = 0.0) and a stream whose extent defines the end
    just work out of the box.

    ----------
    Parameters:
        session (Session):
            The session whose reference stream's extent defines the clock end
            when ``end`` is not given.
        timestep (float):
            Spacing between successive clock ticks (in the same units as
            ``time_coord``, e.g. seconds).
        time_coord (str):
            Name of the time coordinate on the reference stream.
        start (float | None):
            Clock start. Defaults to 0.0 (matches a normalised session).
        end (float | None):
            Clock end. Defaults to the latest time across all reference-stream
            members. Raises if no reference-stream times can be found.
        reference_stream (str):
            Spec name whose member(s) define the clock end. Default
            ``'events'``.
    Returns:
        np.ndarray:
            The 1D clock vector, regularly spaced from start to end.
    '''

    # 1| If end was not supplied, infer it from the latest reference-stream time.
    if end is None:
        ref_names = _reference_members(session.data, reference_stream)
        ref_times = [
            _member_times(session.data[name], time_coord)
            for name in ref_names
        ]
        # Keep only members that actually have a non-empty time axis.
        ref_times = [t for t in ref_times if t is not None and t.size]
        if not ref_times:
            raise ValueError(
                f'cannot infer global-clock end: reference stream {reference_stream!r} has no times.'
            )
        end = float(max(np.max(t) for t in ref_times))

    # 2| Delegate the actual clock construction to globaltimes. Default start
    #    is 0.0 (matches a session that was normalised to zero).
    return create_global_clock(start if start is not None else 0.0, end, timestep)

#===============================================================================



#===============================================================================
# 5| Build Per-Member Index Tables onto a Global Clock
#===============================================================================
def build_index_tables(
        session: Session,
        global_clock: np.ndarray,
        *,
        time_coord: str = 'Time',
        match_type: str = 'nearest',
) -> dict[str, pd.DataFrame]:
    '''
    Map each time-bearing member onto a global clock and return the tables.

    For every member with a time axis, ``index_map_util`` returns (among other
    things) a ``'full'`` table linking each global-clock index to that
    member's own sample index. We keep those tables keyed by member name so a
    caller can later answer "at global tick N, which row of stream X applies?"
    without resampling the data.

    ----------
    Parameters:
        session (Session):
            The session whose members are being indexed. Members without a
            ``time_coord`` axis are silently skipped.
        global_clock (np.ndarray):
            The shared 1D time vector (output of ``build_global_clock``, or
            any compatible array).
        time_coord (str):
            Name of the time coordinate on each member.
        match_type (str):
            Match policy forwarded to ``index_map_util`` (e.g. ``'nearest'``,
            ``'before'``). Default ``'nearest'``.
    Returns:
        dict[str, pd.DataFrame]:
            One ``'full'`` index table per time-bearing member, keyed by
            member name. Members with no time axis are omitted.
    '''

    # Walk the bundle, building one table per time-bearing member.
    tables: dict[str, pd.DataFrame] = {}
    for name, obj in session.data.items():
        times = _member_times(obj, time_coord)
        if times is None or not times.size:
            continue  # No time axis on this member, nothing to map.
        # The 'full' table is the one we want: per global tick, the matched
        # member sample index plus any auxiliary columns from index_map_util.
        tables[name] = index_map_util(global_clock, times, match_type=match_type)['full']
    return tables

#===============================================================================



################################################################################
