"""
Normalise named data streams and combine the same stream across sessions.
-------------------------------------------------------------------------

Description:
    ``streams.py`` defines the data representation used at the boundary between
    per-session loading and cross-session combination.

    A STREAM is one named supported analysis object, currently a pandas
    ``DataFrame`` or xarray ``DataArray`` / ``Dataset``. After a
    ``StreamCatalog`` has processed ONE session, its output is represented as a
    ``StreamMap``: ``{stream_name: DataObject}``.

    Reader/configurator outputs are not required to start in that exact form.
    ``object_to_streams`` normalises several common source-object shapes into a
    StreamMap, including wrappers exposing ``.data_arrays`` or ``.df``.

    ``StreamContainer`` then flips the orientation from "all streams belonging to
    one session" to "one stream gathered from every session that provides it".
    Its ``combine`` operation concatenates those like-named objects, tags each
    element with its source session, and broadcasts every explicit
    ``SessionRef.levels`` value using ``attach_level``.

    ``SessionRef.metadata`` is intentionally retained alongside each member but
    is NOT automatically broadcast. That distinction prevents arbitrary session
    metadata from being duplicated across every row/time point unless it has
    explicitly been declared a grouping/identity level.

Contents:
--------------------------------
- DataObject:                    Supported runtime stream object union.
- StreamMap:                     ``{stream_name: DataObject}`` for one session.
- object_to_streams:             Normalise one loaded/configured object into streams.
- attach_level:                  Broadcast one per-session level onto combined data.
- StreamContainer:               One named stream gathered across multiple sessions.
- _stream_container_init:        Initialise/validate a StreamContainer.
- _stream_container_add:         Add one session's member to the container.
- _stream_container_session_ids: Return contributing session ids in order.
- _stream_container_len:         Return the number of contributing sessions.
- _stream_container_combine:     Concatenate members and attach session levels.
"""


################################################################################
# Imports
################################################################################

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .selection import SessionRef

################################################################################


################################################################################
# Types
################################################################################

# One STREAM must be an object for which this module has explicit combination and
# level-attachment semantics. Keeping the union narrow is intentional: arbitrary
# Python objects should be resolved/configured before they cross this boundary.
DataObject = xr.DataArray | xr.Dataset | pd.DataFrame

# A StreamMap is the complete set of named streams produced for ONE session after
# reading, configuration, and object normalisation.
StreamMap = dict[str, DataObject]

# Python's ``isinstance`` needs concrete runtime classes rather than a ``|`` union
# alias. Keep this tuple private so runtime validation uses the exact same three
# supported object types as the public ``DataObject`` annotation.
_DATA_OBJECT_TYPES = (xr.DataArray, xr.Dataset, pd.DataFrame)

################################################################################


################################################################################
# Object -> StreamMap Conversion
################################################################################


# ===============================================================================
# 1| Convert One Reader/Configurator Output into One or More Named Streams
# ===============================================================================


def object_to_streams(
    obj: object,  # Loaded/configured source object to normalise.
    name: str,  # Source name used as stream name or multi-stream prefix.
    *,
    prefix_data_arrays: bool = True,  # Prefix sub-stream names with ``f'{name}:'`` when True.
) -> StreamMap:  # Returns only supported named DataObject values.
    """
    Normalise one reader/configurator output into a ``StreamMap`` fragment.

    Readers and configurators may expose data through different lightweight
    wrappers. This helper converts those shapes into one common dictionary form
    so the rest of the pipeline never needs to care which source wrapper produced
    the data.

    Supported input shapes are checked in this order:
      1. object exposing ``.data_arrays`` as a mapping;
      2. object exposing ``.df`` as a pandas DataFrame;
      3. bare DataArray / Dataset / DataFrame;
      4. plain mapping whose values are supported DataObjects.

    The order matters. A rich source object may itself also satisfy a more generic
    protocol, so source-specific interfaces are inspected before falling back to
    bare/mapping handling.

    Parameters
    ----------
    obj : object
        Loaded/configured object to convert. The accepted runtime shapes are
        listed above. ``object`` is used rather than ``DataObject`` because the
        whole purpose of this function is to unwrap source objects into actual
        ``DataObject`` values.
    name : str
        Name associated with the source object in the catalog's configured object
        dictionary. A single stream uses this name directly. Multi-member objects
        use it as a prefix by default, e.g. ``dlc:position`` and
        ``dlc:confidence``.
    prefix_data_arrays : bool
        Whether keys from ``.data_arrays`` or plain mappings should be prefixed
        with ``f'{name}:'``. True by default because independent sources commonly
        reuse generic sub-names such as ``position`` or ``confidence``; prefixing
        prevents those sub-names from colliding in the session StreamMap.

    Returns
    -------
    StreamMap
        Dictionary mapping produced stream names to supported ``DataObject``
        values. Every value is runtime-validated before return.
    """

    # === 1| Prefer Explicit ``.data_arrays`` Source Interfaces ==================
    # HARP-like and pose-like source wrappers often expose several named arrays
    # through this attribute. Treat the wrapper as a multi-stream source rather
    # than trying to combine/return the wrapper object itself.

    data_arrays = getattr(obj, "data_arrays", None)  # Missing attribute produces None without raising.

    if isinstance(data_arrays, Mapping):
        streams = (
            {f"{name}:{key}": value for key, value in data_arrays.items()}  # Prefix protects generic sub-names from cross-source collisions.
            if prefix_data_arrays
            else dict(data_arrays)  # Explicit opt-out preserves only the sub-stream keys.
        )

    # === 2| Unwrap Source Objects that Expose One DataFrame via ``.df`` ==========

    elif isinstance(getattr(obj, "df", None), pd.DataFrame):
        streams = {name: obj.df}  # Single-member wrapper naturally retains the source's own name.

    # === 3| Pass Through Bare Supported DataObjects ==============================

    elif isinstance(obj, _DATA_OBJECT_TYPES):
        streams = {name: obj}  # No wrapper/sub-name exists, so use the configured object name directly.

    # === 4| Expand a Plain Mapping of DataObjects ================================
    # This supports configurators that naturally return a small named set of
    # streams without requiring a custom wrapper class solely for ``.data_arrays``.

    elif isinstance(obj, Mapping):
        streams = (
            {f"{name}:{key}": value for key, value in obj.items()}  # Same collision-avoidance convention as ``.data_arrays``.
            if prefix_data_arrays
            else dict(obj)
        )

    # === 5| Reject Objects with No Supported Stream Representation ===============

    else:
        raise TypeError(f"object {name!r} produced {type(obj).__name__}; expected an object with .data_arrays/.df, a DataArray/Dataset/DataFrame, or a mapping of those.")

    # === 6| Enforce that Every Produced Value is Actually a DataObject ===========
    # A plain Mapping can contain nested dicts, strings, configuration objects,
    # etc. Allowing those through would postpone the failure until combination,
    # far away from the object that violated the StreamMap contract.

    unsupported = {
        key: type(value).__name__  # Keep error output compact but identify the offending runtime type.
        for key, value in streams.items()
        if not isinstance(value, _DATA_OBJECT_TYPES)
    }

    if unsupported:
        raise TypeError(f"object {name!r} produced unsupported stream values: {unsupported}; expected DataArray / Dataset / DataFrame values.")

    # === 7| Return a Validated StreamMap Fragment =================================

    return streams  # Safe for direct merge into the per-session StreamMap.


# ===============================================================================


################################################################################
# Level Attachment
################################################################################


# ===============================================================================
# 1| Attach One Per-Session Level to a Combined Stream
# ===============================================================================


def attach_level(
    data: DataObject,  # Combined stream already carrying per-element session identity.
    level_name: str,  # Name of the level to broadcast, e.g. mouseID or day.
    session_to_value: Mapping[str, Any],  # Mapping from session id to that session's level value.
    *,
    dim: str = "Time",  # Alignment dimension for xarray streams.
    base: str = "session",  # Existing session coordinate/column used as the mapping key.
) -> DataObject:  # Returns same broad data-object type with the new level attached.
    """
    Broadcast one session-level value onto every element of a combined stream.

    Combination first tags each element with its source session. Because a level
    such as mouseID/day is constant within a session, the level can then be
    attached by mapping that per-element session identity through
    ``session_to_value``.

    Parameters
    ----------
    data : DataObject
        Combined xarray or DataFrame stream. It must already carry source-session
        identity under ``base``. For xarray this is a coordinate; for DataFrames
        it is a column.
    level_name : str
        Name of the level to create, e.g. ``'mouseID'``, ``'phase'``, or
        ``'day'``.
    session_to_value : Mapping[str, Any]
        Mapping ``{session_id: level_value}``. Values may be any metadata type
        representable in an object-valued xarray coordinate/DataFrame column.
        ``dict.get`` semantics are used, so sessions absent from the mapping
        receive ``None`` rather than causing a second lookup error.
    dim : str
        xarray dimension along which the combined stream is aligned and therefore
        along which the new coordinate should be attached. Default ``'Time'``.
        Ignored for DataFrame streams because rows are already the alignment axis.
    base : str
        Name of the existing source-session coordinate/column used to look up
        level values. Default ``'session'``.

    Returns
    -------
    DataObject
        Derived data object carrying ``level_name`` as an xarray coordinate or
        DataFrame column. The input DataFrame is copied rather than modified in
        place; xarray ``assign_coords`` likewise returns a derived object.
    """

    # === 1| xarray: Map the Existing Session Coordinate onto the New Level =======

    if isinstance(data, xr.DataArray | xr.Dataset):
        if base not in data.coords:  # Without source session identity there is nothing to map from.
            raise KeyError(f"combined stream has no {base!r} coordinate.")
        if dim not in data.dims:  # New coordinate must align with an actual dimension.
            raise KeyError(f"combined stream has no {dim!r} dimension.")
        if level_name in data.coords or level_name in data.dims or (isinstance(data, xr.Dataset) and level_name in data.data_vars):
            raise ValueError(f"session level {level_name!r} collides with an existing xarray name.")

        sessions = data.coords[base].values  # One source-session label per element along ``dim``.
        values = np.array(
            [session_to_value.get(session) for session in sessions],  # Broadcast per-session constant to every matching element.
            dtype=object,  # Preserve heterogeneous/string metadata without coercion surprises.
        )

        return data.assign_coords({level_name: (dim, values)})  # Coordinate is aligned element-for-element with the existing stream.

    # === 2| DataFrame: Map the Existing Session Column onto a New Column =========

    if isinstance(data, pd.DataFrame):
        if base not in data.columns:  # Combined DataFrame must already identify each row's source session.
            raise KeyError(f"combined stream has no {base!r} column.")
        if level_name in data.columns or level_name == data.index.name:
            raise ValueError(f"session level {level_name!r} collides with an existing DataFrame name.")

        out = data.copy()  # Avoid mutating the combined object held by another caller/reference.
        out[level_name] = [
            session_to_value.get(session)  # Map each row's session id to the session-level constant.
            for session in out[base].to_numpy()
        ]
        return out

    # === 3| Reject Unsupported Runtime Objects ===================================
    # Typed callers should never reach this branch because ``DataObject`` is
    # narrow, but explicit runtime validation gives untyped use a clear failure.

    raise TypeError(f"cannot attach a level to {type(data).__name__}; expected DataArray / Dataset / DataFrame.")


# ===============================================================================


# ===============================================================================
# 2| Harmonise Along-Dimension Coordinates Before xarray Concatenation
# ===============================================================================


def _harmonise_along_dim(
    objects: list[xr.DataArray | xr.Dataset],
    dim: str,
) -> list[xr.DataArray | xr.Dataset]:
    """Give every xarray object the same set of auxiliary coordinates along ``dim``.

    ``xr.concat`` rejects a coordinate that is present on only some inputs. This
    occurs when a previously combined stream, carrying provenance coordinates,
    is combined with a plain per-session stream. Missing coordinates are filled
    with object-valued ``NaN`` entries so nested combination remains composable.
    """

    along_dim: set[str] = {name for obj in objects for name, coordinate in obj.coords.items() if name != dim and dim in coordinate.dims}

    if not along_dim:
        return objects

    harmonised: list[xr.DataArray | xr.Dataset] = []
    for obj in objects:
        missing = along_dim.difference(obj.coords)
        if missing:
            obj = obj.assign_coords({name: (dim, np.full(obj.sizes[dim], np.nan, dtype=object)) for name in missing})
        harmonised.append(obj)

    return harmonised


# ===============================================================================


################################################################################
# StreamContainer Methods
################################################################################


# ===============================================================================
# 1| Initialise and Validate a StreamContainer
# ===============================================================================


def _stream_container_init(
    self: "StreamContainer",  # Container instance being initialised.
    name: str,  # Single stream name represented by this container.
    streams: dict[str, DataObject] | None = None,  # Optional initial per-session stream mapping.
    sessions: dict[str, SessionRef] | None = None,  # Optional matching per-session SessionRef mapping.
) -> None:  # Stores validated state on ``self`` and returns nothing.
    """
    Initialise one cross-session stream container and validate initial members.

    Parameters
    ----------
    self : StreamContainer
        Container instance being initialised.
    name : str
        Name of the stream represented by every member of this container.
    streams : dict[str, DataObject] | None
        Optional ``{session_id: DataObject}`` mapping. A defensive shallow copy is
        taken so later mutation of the caller's dictionary does not silently alter
        the container's membership.
    sessions : dict[str, SessionRef] | None
        Optional ``{session_id: SessionRef}`` mapping corresponding one-to-one
        with ``streams``. A defensive shallow copy is likewise taken.

    Returns
    -------
    None
        Validated container state is stored on ``self``.
    """

    # === 1| Store Independent Dictionary Copies ==================================

    self.name = name  # Container identity never changes during combination.
    self.streams = {} if streams is None else dict(streams)  # Avoid aliasing caller-owned membership dictionary.
    self.sessions = {} if sessions is None else dict(sessions)  # Keep session descriptions keyed identically to stream members.

    # === 2| Require One SessionRef for Every Initial Stream ======================
    # Combination needs the matching SessionRef to attach levels. Allowing the
    # two mappings to drift would create a partially described member set.

    if set(self.streams) != set(self.sessions):  # Key-set equality enforces a one-to-one correspondence.
        raise ValueError(f"streams and sessions must contain the same session ids; got streams={sorted(self.streams)} and sessions={sorted(self.sessions)}.")

    # === 3| Validate Every Initial Stream Value ==================================

    for session_id, stream in self.streams.items():
        if not isinstance(stream, _DATA_OBJECT_TYPES):  # Reject unsupported types at the container boundary, not during concat.
            raise TypeError(f"stream {self.name!r} for session {session_id!r} is {type(stream).__name__}; expected DataArray / Dataset / DataFrame.")


# ===============================================================================


# ===============================================================================
# 2| Add One Session's Instance of this Stream
# ===============================================================================


def _stream_container_add(
    self: "StreamContainer",  # Container receiving another session member.
    session_id: str,  # Selection key identifying the source session.
    session: SessionRef,  # Source session reference carrying path/levels/metadata.
    stream: DataObject,  # This stream's data object for the source session.
) -> "StreamContainer":  # Returns ``self`` to support incremental/chained construction.
    """
    Add one session's data object to this stream container.

    Parameters
    ----------
    self : StreamContainer
        Container being modified.
    session_id : str
        Session identifier from the selection mapping. This same id is later
        attached to every element contributed by the session during combination.
    session : SessionRef
        Reference describing the selected source session. ``session.levels`` are
        later broadcast onto the combined stream; ``session.metadata`` remains
        available for inspection without automatic broadcasting.
    stream : DataObject
        This container's named stream for the given session. Must be one of the
        supported DataFrame / DataArray / Dataset types.

    Returns
    -------
    StreamContainer
        The same container instance after insertion.
    """

    # === 1| Reject Duplicate Membership for the Same Session =====================

    if session_id in self.streams:  # One container can hold at most one object per session id.
        raise ValueError(f"session {session_id!r} is already present in stream {self.name!r}.")

    # === 2| Enforce the Supported Stream Type Contract ===========================

    if not isinstance(stream, _DATA_OBJECT_TYPES):  # Fail at insertion rather than much later during combination.
        raise TypeError(f"stream {self.name!r} for session {session_id!r} is {type(stream).__name__}; expected DataArray / Dataset / DataFrame.")

    # === 3| Store the DataObject and Matching SessionRef Together ================

    self.streams[session_id] = stream  # Insertion order becomes cross-session combination order.
    self.sessions[session_id] = session  # Same key preserves direct member->SessionRef correspondence.

    return self  # Enables ``container.add(...).add(...)`` if desired.


# ===============================================================================


# ===============================================================================
# 3| Return Contributing Session IDs in Combination Order
# ===============================================================================


def _stream_container_session_ids(
    self: "StreamContainer",  # Container being inspected.
) -> list[str]:  # Returns session ids in dictionary insertion order.
    """
    Return the session ids contributing this stream, in insertion order.

    Parameters
    ----------
    self : StreamContainer
        Container being inspected.

    Returns
    -------
    list[str]
        Ordered copy of session ids. This order is also the order in which the
        corresponding per-session objects are concatenated by ``combine``.
    """

    return list(self.streams)  # Standard dict iteration preserves insertion order.


# ===============================================================================


# ===============================================================================
# 4| Return the Number of Sessions Contributing this Stream
# ===============================================================================


def _stream_container_len(
    self: "StreamContainer",  # Container being inspected.
) -> int:  # Returns number of per-session members.
    """
    Return how many selected sessions contribute this stream.

    Parameters
    ----------
    self : StreamContainer
        Container being inspected.

    Returns
    -------
    int
        Number of stored per-session stream objects.
    """

    return len(self.streams)


# ===============================================================================


# ===============================================================================
# 5| Combine this Named Stream Across All Contributing Sessions
# ===============================================================================


def _stream_container_combine(
    self: "StreamContainer",  # Container whose members should be concatenated.
    *,
    dim: str = "Time",  # Alignment dimension for xarray concatenation.
    session_coord: str = "session",  # Name used to tag source-session identity per element.
) -> DataObject:  # Returns one combined stream with session + explicit levels attached.
    """
    Concatenate this stream across sessions and attach session/level identity.

    DataFrame and xarray streams require different mechanics but produce the same
    conceptual result: one combined object where every row/element can still be
    traced to its source session and hierarchy levels.

    DataFrame indexes receive special treatment. If EVERY per-session DataFrame
    has a named index, the indexes are preserved during concatenation because a
    named index often carries meaningful per-session time values required by
    session-aware slicing. If any index is unnamed, concatenation instead uses a
    fresh RangeIndex to avoid presenting an accidental/unlabelled index as
    semantically meaningful.

    Parameters
    ----------
    self : StreamContainer
        Container holding one named stream and its matching SessionRefs.
    dim : str
        Dimension along which xarray members should be concatenated. Defaults to
        ``'Time'``. Every xarray member must contain this dimension. Ignored for
        DataFrame concatenation, which always concatenates rows.
    session_coord : str
        Name assigned to the per-element source-session identity. For DataFrames
        this becomes a column; for xarray it becomes a coordinate along ``dim``.
        Defaults to ``'session'``.

    Returns
    -------
    DataObject
        One combined DataFrame / DataArray / Dataset carrying source-session
        identity plus every explicit level found in the contributing
        ``SessionRef.levels`` mappings.
    """

    # === 1| Reject Combination of an Empty Container =============================

    if not self.streams:  # An empty result would have no type/dimension semantics to infer safely.
        raise ValueError(f"stream {self.name!r} has no session data to combine.")

    # === 2| Capture Ordered Members and Establish the Stream Family ==============

    session_ids = self.session_ids  # Same order used for data objects, labels, and level mappings.
    objects = list(self.streams.values())  # Ordered per-session instances of this one stream.
    first = objects[0]  # First object determines DataFrame vs xarray combination branch.

    # === 3| Combine DataFrame Members by Rows ====================================

    if isinstance(first, pd.DataFrame):
        # === 3.1| Require Every Member to be a DataFrame =========================
        # Mixing DataFrame/xarray under one stream name would make the stream's
        # semantics inconsistent across sessions and cannot be concatenated safely.

        if not all(isinstance(obj, pd.DataFrame) for obj in objects):
            raise TypeError(f"stream {self.name!r} mixes DataFrame and non-DataFrame objects.")

        # === 3.2| Copy Each Piece and Tag Rows with Source Session ================

        index_names = {obj.index.name for obj in objects}
        if len(index_names) > 1:
            raise ValueError(f"stream {self.name!r} mixes DataFrame index names {sorted(repr(name) for name in index_names)}.")
        if session_coord in first.columns or session_coord == first.index.name:
            raise ValueError(f"session_coord {session_coord!r} collides with an existing DataFrame column or index name in stream {self.name!r}.")
        if any(session_coord in obj.columns or session_coord == obj.index.name for obj in objects[1:]):
            raise ValueError(f"session_coord {session_coord!r} collides with an existing DataFrame column or index name in stream {self.name!r}.")

        pieces: list[pd.DataFrame] = []
        for session_id, obj in zip(session_ids, objects, strict=True):
            piece = obj.copy()  # Do not mutate the per-session object stored in ``self.streams``.
            piece[session_coord] = session_id  # One constant session id broadcast to all rows from this piece.
            pieces.append(piece)

        # === 3.3| Preserve Named Indexes Only When All Pieces Have One ============
        # Named indexes are likely semantic (commonly Time). Unnamed indexes are
        # commonly incidental row numbering, so reset them during concatenation.

        preserve_index = first.index.name is not None
        combined: DataObject = pd.concat(
            pieces,
            ignore_index=not preserve_index,  # Preserve repeated per-session time indexes only when explicitly named.
        )

    # === 4| Combine xarray Members Along the Requested Dimension =================

    elif isinstance(first, xr.DataArray | xr.Dataset):
        # === 4.1| Require the Exact Same xarray Container Type ===================
        # DataArray and Dataset have related APIs but represent different object
        # structures; combining them under one stream name is almost certainly a
        # misconfigured reader/configurator pipeline.

        if not all(type(obj) is type(first) for obj in objects):
            raise TypeError(f"stream {self.name!r} mixes incompatible xarray object types.")

        # === 4.2| Require the Alignment Dimension in Every Session Member =========

        for obj in objects:
            if dim not in obj.dims:
                raise KeyError(f"stream {self.name!r} has no {dim!r} dimension in one or more sessions.")
            if session_coord == dim or session_coord in obj.coords or (isinstance(obj, xr.Dataset) and session_coord in obj.data_vars):
                raise ValueError(f"session_coord {session_coord!r} collides with an existing xarray name in stream {self.name!r}.")

        # === 4.3| Concatenate Data and Build One Parallel Session Label Array =====

        objects = _harmonise_along_dim(objects, dim)  # Preserve auxiliary/provenance coords present on only some sessions.
        combined = xr.concat(objects, dim=dim)  # Session pieces remain in the container's insertion order.
        labels = np.concatenate(
            [
                np.repeat(session_id, obj.sizes[dim])  # Repeat each session id once per element it contributes along ``dim``.
                for session_id, obj in zip(session_ids, objects, strict=True)
            ]
        )
        combined = combined.assign_coords({session_coord: (dim, labels)})  # Session identity now travels element-by-element with the combined data.

    # === 5| Reject Any Runtime Type Outside the DataObject Contract ===============

    else:
        raise TypeError(f"stream {self.name!r} contains {type(first).__name__}; expected DataArray / Dataset / DataFrame.")

    # === 6| Discover Explicit Level Names in Stable First-Appearance Order ========
    # Sessions can technically carry non-identical level dictionaries (e.g. a
    # missing optional level). Build the union without sorting so declared/walk
    # order remains intuitive. Missing values later resolve through ``dict.get``.

    level_names: list[str] = []

    for session in self.sessions.values():
        for level_name in session.levels:
            if level_name not in level_names:  # Keep each level once while preserving first appearance.
                level_names.append(level_name)

    # === 7| Broadcast Each Explicit Session Level onto the Combined Stream =======

    for level_name in level_names:
        # === 7.1| Prevent a Level from Overwriting Source-Session Identity ========

        if level_name == session_coord:
            raise ValueError(f"session level {level_name!r} collides with session_coord.")

        # === 7.2| Build the Per-Session Constant Mapping for this Level ===========

        session_to_value = {
            session_id: session.levels.get(level_name)  # Missing level on one SessionRef intentionally becomes None.
            for session_id, session in self.sessions.items()
        }

        # === 7.3| Broadcast the Mapping Element-by-Element ========================

        combined = attach_level(
            combined,  # Current combined object, including any previously attached levels.
            level_name,  # New coordinate/column to create.
            session_to_value,  # Per-session constant lookup table.
            dim=dim,  # xarray alignment dimension; ignored for DataFrames.
            base=session_coord,  # Existing source-session identity used as mapping key.
        )

    # === 8| Return the Fully Combined Stream =====================================

    return combined  # Metadata remains accessible through ``self.sessions`` but is not broadcast.


# ===============================================================================


################################################################################
# StreamContainer
################################################################################


# ===============================================================================
# 1| StreamContainer (One Named Stream Gathered Across Sessions)
# ===============================================================================


@dataclass(init=False, eq=False)
class StreamContainer:
    """
    Gather one named stream from every selected session that provides it.

    A container represents ONE stream name, not one session. For example, after
    per-session loading there may be ``'trials'`` data from sessions A/B/C but
    ``'dlc:position'`` only from A/C. Those become two separate containers with
    different contributing session sets.

    ``streams`` and ``sessions`` use the same session ids as keys. The former
    stores the actual DataObjects; the latter stores the matching ``SessionRef``
    so combination can access explicit levels and retained metadata.

    ``dataclass(init=False, eq=False)`` is intentional. The generated repr remains
    useful, while identity-based equality avoids applying scalar ``==`` semantics
    to DataFrame/xarray members. The custom ``__init__`` implementation is defined
    immediately above and exposed by the class body so validation still runs at
    construction.

    Parameters
    ----------
    name : str
        Stream name represented by the container, e.g. ``'trials'`` or
        ``'dlc:position'``.
    streams : dict[str, DataObject] | None
        Optional initial mapping from session id to that session's instance of
        this stream.
    sessions : dict[str, SessionRef] | None
        Optional matching mapping from session id to ``SessionRef``. Its key set
        must exactly match ``streams`` so every data object has an associated
        source-session description.
    """

    name: str  # Single stream name represented by every member of this container.
    streams: dict[str, DataObject] = field(default_factory=dict)  # Session id -> this stream's DataObject for that session.
    sessions: dict[str, SessionRef] = field(default_factory=dict)  # Session id -> matching path/levels/metadata reference.

    __init__ = _stream_container_init  # Validated construction using the externally defined implementation.
    add = _stream_container_add  # Add/replace one session's contribution to this stream.
    session_ids = property(_stream_container_session_ids)  # Read-only ordered session-id view.
    __len__ = _stream_container_len  # Number of contributing sessions.
    combine = _stream_container_combine  # Cross-session combination + level attachment.


# ===============================================================================
