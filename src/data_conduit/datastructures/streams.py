"""
Normalise named data streams and combine the same stream across sessions.
-------------------------------------------------------------------------

Description:
    ``streams.py`` defines the data representation used at the boundary between
    per-session loading and cross-session combination.

    A STREAM is one named supported analysis object, currently a pandas
    ``DataFrame`` or xarray ``DataArray`` / ``Dataset``. A ``StreamMap`` is the
    common ``{stream_name: DataObject}`` shape used both for one session's catalog
    output and for the final cross-session combined result.

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
- DataObject:                    Type alias for supported stream objects.
- StreamMap:                     Type alias for a mapping of named streams.
- object_to_streams:             Normalise one loaded/configured object into streams.
- attach_level:                  Broadcast one per-session level onto combined data.
- _harmonise_along_dim:          Harmonise xarray auxiliary coordinates before concat.
- _validate_stream_object:       Validate one per-session stream object.
- _validate_container_members:   Validate initial member/session mappings.
- _combine_dataframe_streams:    Combine DataFrame members and tag source rows.
- _combine_xarray_streams:       Combine xarray members and tag source elements.
- _attach_stream_levels:         Attach explicit SessionRef levels.
- _add_stream_member:            Add one session's member to the container.
- _get_stream_session_ids:       Return contributing session IDs in order.
- _count_stream_members:         Return the number of contributing sessions.
- _combine_stream_container:     Concatenate members and attach session levels.
- StreamContainer:               One named stream gathered across multiple sessions.
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


# A DataObject is one stream value whose combination and level-attachment
# semantics are defined by this module:
#   - ``xr.DataArray`` for one labelled n-dimensional array;
#   - ``xr.Dataset`` for related labelled arrays sharing coordinates;
#   - ``pd.DataFrame`` for labelled tabular data.
#
# This alias documents the supported boundary and helps static type checkers; it
# does not prevent untyped callers from passing other objects at runtime. Public
# entry points therefore validate values explicitly where the boundary is crossed.
DataObject = xr.DataArray | xr.Dataset | pd.DataFrame

# A StreamMap maps stream names to ``DataObject`` values. The catalog uses this
# shape for one session's normalised output, and DataStructure uses it again for
# the final cross-session result after each value has been combined. Like
# ``DataObject``, the alias describes the expected shape but does not enforce it
# at runtime; ``object_to_streams`` validates the values it produces.
StreamMap = dict[str, DataObject]

# PEP 604 unions can be passed to ``isinstance`` on supported Python versions.
# This tuple is retained as the single reusable runtime class collection and also
# makes it straightforward to enumerate the accepted class names in errors.
_DATA_OBJECT_TYPES = (xr.DataArray, xr.Dataset, pd.DataFrame)

################################################################################


################################################################################
# Object -> StreamMap Conversion
################################################################################


# ===============================================================================
# 1| Convert One Reader/Configurator Output into One or More Named Streams
# ===============================================================================

def object_to_streams(
        obj: object,                           # Loaded/configured source object to normalise.
        name: str,                             # Source name used directly or as a prefix.
        *,                                     # Following arguments must be passed by keyword.
        prefix_data_arrays: bool = True,       # Prefix keys from multi-stream objects.
) -> StreamMap:                                # Validated named DataObject values from this source.
    """
    Normalise one reader/configurator output into a ``StreamMap`` fragment.

    Readers and configurators may expose data through several lightweight object
    shapes. This function translates those shapes into the common
    ``{stream_name: DataObject}`` representation consumed by the rest of the
    pipeline.

    Supported input shapes are checked in this order:

      1. an object exposing ``.data_arrays`` as a mapping;
      2. an object exposing ``.df`` as a pandas DataFrame;
      3. a bare DataArray, Dataset, or DataFrame;
      4. a plain mapping whose values are supported ``DataObject`` values.

    The order is intentional: a source-specific wrapper may also satisfy a more
    general protocol, so explicit wrapper interfaces take precedence over bare
    object or mapping handling. Every produced value is runtime-validated before
    the fragment is returned.

    Parameters
    ----------
    obj : object
        Loaded or configured object to convert. ``object`` is intentionally broad
        because this function unwraps source-specific containers into actual
        ``DataObject`` values.
    name : str
        Name assigned to the source in the catalog's configured-object mapping.
        Single-stream inputs use it directly. Multi-stream inputs use it as a
        prefix by default, producing names such as ``'dlc:position'``.
    prefix_data_arrays : bool, optional
        Whether keys from ``.data_arrays`` and plain mappings should be prefixed
        with ``f'{name}:'``. Defaults to True because independent sources often
        reuse generic sub-names such as ``'position'`` or ``'confidence'``.
        Single-stream inputs are unaffected.

    Returns
    -------
    StreamMap
        Dictionary mapping produced stream names to supported ``DataObject``
        values.

    Raises
    ------
    TypeError
        If ``obj`` has no supported representation or any produced value is not a
        DataArray, Dataset, or DataFrame.
    """

    # === 1| Prefer Explicit ``.data_arrays`` Source Interfaces ==================
    # Multi-stream source wrappers expose their named arrays through this
    # attribute. Read it safely so objects without the interface fall through to
    # the next supported representation.

    data_arrays = obj.data_arrays if hasattr(obj, "data_arrays") else None

    if isinstance(data_arrays, Mapping):
        if prefix_data_arrays:                                                          # noqa: SIM108
            streams = {
                f"{name}:{key}": value
                for key, value in data_arrays.items()
            }
        else:
            streams = dict(data_arrays)


    # === 2| Unwrap Objects Exposing One DataFrame through ``.df`` ================

    elif isinstance(getattr(obj, "df", None), pd.DataFrame):
        streams = {name: obj.df}

    # === 3| Pass Through Bare Supported DataObject Values =======================

    elif isinstance(obj, _DATA_OBJECT_TYPES):
        streams = {name: obj}

    # === 4| Expand Plain Mappings of Named DataObject Values ====================
    # Configurators can return a small named group directly without introducing a
    # wrapper class solely to expose ``.data_arrays``.

    elif isinstance(obj, Mapping):
        if prefix_data_arrays:                                                          # noqa: SIM108
            streams = {
                f"{name}:{key}": value
                for key, value in obj.items()
            }
        else:
            streams = dict(obj)

    # === 5| Reject Objects with No Supported Stream Representation ===============

    else:
        raise TypeError(
            f"Unsupported object type for stream '{name}': {type(obj).__name__}"
            f"Supported types are: {', '.join(t.__name__ for t in _DATA_OBJECT_TYPES)}"
        )

    # === 6| Validate Every Produced Stream Value ================================
    # A plain Mapping can contain nested dicts, strings, configuration objects,
    # etc. Allowing those through would postpone the failure until combination,
    # far away from the object that violated the StreamMap contract.
    unsupported = {
        key: type(value).__name__  # Keep error output compact but identify the offending runtime type.
        for key, value in streams.items()
        if not isinstance(value, _DATA_OBJECT_TYPES)    # Collect every invalid value in one pass.
    }

    if unsupported:
        raise TypeError(f"object {name!r} produced unsupported stream values: {unsupported}; "
                        f"Expected stream values to be instances of: {', '.join(t.__name__ for t in _DATA_OBJECT_TYPES)}")

    # === 7| Return the Validated StreamMap Fragment =============================

    return streams


# ===============================================================================




################################################################################
# Level Attachment
################################################################################


# ===============================================================================
# 1| Attach One Per-Session Level to a Combined Stream
# ===============================================================================

def attach_level(
        data: DataObject,                           # Combined stream carrying source-session identity.
        level_name: str,                            # Name of the level to broadcast.
        session_to_value: Mapping[str, Any],        # Session ID -> level value lookup.
        *,                                          # Following arguments must be passed by keyword.
        dim: str = 'Time',                          # Alignment dimension for xarray data.
        base: str = 'session',                      # Existing session coordinate or column.
) -> DataObject:                                    # Derived object with the level attached.
    """
    Broadcast one per-session level onto every element of a combined stream.

    Combination first tags each row or xarray element with its source session.
    Because a level such as mouse ID, phase, or day is constant within a session,
    this function can attach it by mapping the existing source-session identity
    through ``session_to_value``.

    Parameters
    ----------
    data : DataObject
        Combined xarray or DataFrame stream. It must already identify each
        element's source session under ``base``: as a coordinate for xarray or a
        column for DataFrames. For xarray, that coordinate must be one-dimensional
        and aligned with ``dim``.
    level_name : str
        Name of the coordinate or column to add, such as ``'mouseID'``,
        ``'phase'``, or ``'day'``.
    session_to_value : Mapping[str, Any]
        Mapping from session ID to its level value, for example
        ``{"session_1": "mouse_A"}``. Lookup uses ``Mapping.get``, so a session
        absent from the mapping receives ``None``.
    dim : str, optional
        xarray dimension along which the new coordinate is attached. Defaults to
        ``'Time'``. Ignored for DataFrames, whose rows provide the alignment axis.
    base : str, optional
        Existing xarray coordinate or DataFrame column containing the session IDs
        used as lookup keys. Defaults to ``'session'``.

    Returns
    -------
    DataObject
        Derived object with ``level_name`` added as an xarray coordinate or
        DataFrame column. The input is not modified in place.

    Raises
    ------
    KeyError
        If a DataFrame does not contain the ``base`` column.
    ValueError
        If an xarray object lacks the ``base`` coordinate or ``dim``; if
        ``level_name`` matches an xarray coordinate, dimension, or Dataset data
        variable; or if it matches a DataFrame column or ``DataFrame.index.name``.
    TypeError
        If ``data`` is not a supported ``DataObject``.

    Example
    -------
    Add a mouse ID to each row using its source session:

    >>> data = pd.DataFrame({
    ...     "session": ["session_1", "session_1", "session_2"],
    ...     "speed": [2.1, 2.5, 1.8],
    ... })
    >>> session_to_mouse = {
    ...     "session_1": "mouse_A",
    ...     "session_2": "mouse_B",
    ... }
    >>> result = attach_level(data, "mouseID", session_to_mouse)
    >>> result
        session  speed  mouseID
    0  session_1    2.1  mouse_A
    1  session_1    2.5  mouse_A
    2  session_2    1.8  mouse_B
    """

    # === 1| xarray: Map the Session Coordinate onto a New Coordinate ============

    if isinstance(data, (xr.DataArray, xr.Dataset)):

        # === 1.1| Validate the Existing Identity and Target Names ===============

        if base not in data.coords:
            raise ValueError(f"Expected session coordinate '{base}' not found in xarray object.")

        if dim not in data.dims:
            raise ValueError(f"Expected dimension '{dim}' not found in xarray object.")

        if level_name in data.coords or level_name in data.dims or (isinstance(data, xr.Dataset) and level_name in data.data_vars):
            raise ValueError(f"session level {level_name!r} collides with an existing xarray name.")

        # === 1.2| Look Up One Level Value per Element ============================

        sessions = data.coords[base].values                                         # One source-session ID per element along ``dim``.

        level_values = np.array(
                                [session_to_value.get(session)
                                for session in sessions                                   # Broadcast the session-level constant to matching elements.
                                ],
                                dtype=object,                                             # Preserve strings, None, and heterogeneous values.
        )

        # === 1.3| Attach the Values along the Existing Dimension =================

        return data.assign_coords({level_name: (dim, level_values)})                       # Return a derived xarray object; do not mutate ``data``.



    # === 2| DataFrame: Map the Session Column onto a New Column ==================

    if isinstance(data, pd.DataFrame):

        # === 2.1| Validate the Existing Identity and Target Name =================

        if base not in data.columns:
            raise KeyError(f"combined stream has no {base!r} column.")

        if level_name in data.columns or level_name == data.index.name:
            raise ValueError(f"session level {level_name!r} collides with an existing DataFrame name.")

        # === 2.2| Copy and Broadcast the Session-Level Values ====================

        out = data.copy()

        out[level_name] = [
            session_to_value.get(session)
            for session in out[base].to_numpy()
        ]
        return out

    # === 3| Reject Objects Outside the Runtime DataObject Contract ===============
    # Type aliases and annotations do not stop untyped callers from reaching this
    # function, so retain an explicit failure at the public boundary.

    raise TypeError(f"cannot attach a level to {type(data).__name__}; expected DataArray / Dataset / DataFrame.")

# ===============================================================================



# ===============================================================================
# 2| Harmonise Along-Dimension Coordinates Before xarray Concatenation
# ===============================================================================


def _harmonise_along_dim(
    objects: list[xr.DataArray | xr.Dataset],                                       # Ordered xarray members to prepare for concatenation.
    dim: str,                                                                       # Dimension along which members will be concatenated.
) -> list[xr.DataArray | xr.Dataset]:                                               # Members with a shared along-dimension coordinate set.
    """
    Give every xarray member the same auxiliary coordinates along ``dim``.

    ``xr.concat`` cannot consistently carry a coordinate that varies along the
    concatenation dimension but is present on only some inputs. This can occur
    when previously combined data, already carrying provenance coordinates, is
    combined with a plain per-session member. The helper discovers the union of
    those along-dimension coordinates and fills each missing coordinate with one
    object-valued ``NaN`` per element.

    Coordinates unrelated to ``dim`` are left alone. Objects that already have the
    complete along-dimension coordinate set are appended unchanged;
    ``assign_coords`` returns a derived object for members that need placeholders.

    Parameters
    ----------
    objects : list[xr.DataArray | xr.Dataset]
        Ordered DataArrays or Datasets that will be passed to ``xr.concat``.
    dim : str
        Concatenation dimension whose auxiliary coordinates should be harmonised.

    Returns
    -------
    list[xr.DataArray | xr.Dataset]
        Ordered members with a consistent set of coordinates varying along
        ``dim``.

    """

    # === 1| Discover Auxiliary Coordinates that Vary along ``dim`` ==============

    along_dim: set[str] = set()

    for obj in objects:                                     # Inspect each member's coordinate schema.
        for name, coordinate in obj.coords.items():         # Consider dimension and auxiliary coordinates.
            if name != dim and dim in coordinate.dims:      # Exclude the dimension coordinate itself.
                along_dim.add(name)                         # Retain each varying auxiliary coordinate once.


    # === 2| Return Early When No Harmonisation Is Required ======================

    if not along_dim:
        return objects

    # === 3| Fill Missing Coordinates Member by Member ===========================

    harmonised: list[xr.DataArray | xr.Dataset] = []

    for obj in objects:

        # === 3.1| Identify this Member's Missing Coordinate Names ===============

        missing = along_dim.difference(obj.coords)

        # === 3.2| Add One Placeholder per Element along ``dim`` =================

        if missing:
            obj = obj.assign_coords({                                           # Return a derived object; do not mutate the input.
                name: (                                                         # Coordinate specification: dimension plus values.
                    dim,                                                        # Align placeholders with the concatenation dimension.
                    np.full(
                        obj.sizes[dim],                                         # Create one placeholder per element.
                        np.nan,                                                 # The absent coordinate value is unknown.
                        dtype=object,                                           # Accommodate string-valued provenance elsewhere.
                    ),
                )
                for name in missing                                             # Fill every missing along-dimension coordinate.
            })

        # === 3.3| Preserve the Original Member Order ============================

        harmonised.append(obj)

    # === 4| Return Members Ready for Concatenation ==============================

    return harmonised

# ===============================================================================


################################################################################
# StreamContainer Processing Helpers
################################################################################


# ===============================================================================
# 1| Validate One Per-Session Stream Object
# ===============================================================================

def _validate_stream_object(
        stream: object,                                           # Candidate per-session stream member.
        *,                                                        # Context must be supplied by keyword.
        stream_name: str,                                         # Stream name used in validation errors.
        session_id: str,                                          # Source-session ID used in validation errors.
    ) -> None:
    """
    Validate that one per-session member satisfies the ``DataObject`` contract.

    Parameters
    ----------
    stream : object
        Candidate object supplied for one session. Explicit validation is needed
        because the ``DataObject`` alias does not enforce runtime types.
    stream_name : str
        Name of the stream/container being populated.
    session_id : str
        Session identifier associated with this member.

    Returns
    -------
    None
        Returns silently when ``stream`` is a DataFrame, DataArray, or Dataset.

    Raises
    ------
    TypeError
        If ``stream`` is not a supported ``DataObject``.
    """

    # === 1| Enforce the Runtime DataObject Boundary =============================

    if not isinstance(stream, _DATA_OBJECT_TYPES):
        raise TypeError(
            f"stream {stream_name!r} for session {session_id!r} is "
            f"{type(stream).__name__}; expected DataArray / Dataset / DataFrame."
        )

# ===============================================================================


# ===============================================================================
# 2| Validate Initial StreamContainer Members
# ===============================================================================

def _validate_container_members(
        name: str,                                                # Stream name represented by the container.
        streams: Mapping[str, DataObject],                        # Session ID -> stream member.
        sessions: Mapping[str, SessionRef],                       # Session ID -> matching source reference.
    ) -> None:
    """
    Validate the initial member mappings used to construct a container.

    A stream member cannot be combined safely without its matching ``SessionRef``:
    the reference supplies the explicit levels attached after concatenation. The
    two mappings must therefore have identical session ID keys, and every member
    must satisfy the runtime ``DataObject`` boundary.

    Parameters
    ----------
    name : str
        Name of the stream represented by the container.
    streams : Mapping[str, DataObject]
        Initial mapping from session ID to this stream's per-session object.
    sessions : Mapping[str, SessionRef]
        Matching mapping from session ID to ``SessionRef``. Its key set must
        exactly equal the keys in ``streams``.

    Returns
    -------
    None
        Returns silently when the mappings are aligned and every stream is
        supported.

    Raises
    ------
    ValueError
        If ``streams`` and ``sessions`` do not have identical session ID keys.
    TypeError
        If any member of ``streams`` is not a supported ``DataObject``.
    """

    # === 1| Require One SessionRef for Every Stream Member ======================

    if set(streams) != set(sessions):
        raise ValueError(
            "streams and sessions must contain the same session ids; "
            f"got streams={sorted(streams)} and sessions={sorted(sessions)}."
        )

    # === 2| Validate Every Initial Stream Object ================================

    for session_id, stream in streams.items():
        _validate_stream_object(
            stream,
            stream_name=name,
            session_id=session_id,
        )

# ===============================================================================


# ===============================================================================
# 3| Combine DataFrame Stream Members
# ===============================================================================

def _combine_dataframe_streams(
        name: str,                                                # Stream name represented by all members.
        session_ids: list[str],                                  # Session IDs in concatenation order.
        objects: list[DataObject],                               # Ordered per-session members.
        *,                                                        # Following arguments must be passed by keyword.
        session_coord: str,                                      # New source-session column name.
    ) -> pd.DataFrame:
    """
    Concatenate one DataFrame stream across sessions and tag every source row.

    All members must be DataFrames with the same ``DataFrame.index.name``,
    including the case where every index is unnamed. A shared named index is
    retained because its values may be semantically meaningful, such as
    per-session time. When all indexes are unnamed, concatenation creates a fresh
    ``RangeIndex``. Mixing a named index with an unnamed one, or mixing different
    names, is rejected. Per-level ``MultiIndex.names`` are not inspected
    separately.

    Each member is copied before the source-session column is added, so the
    per-session DataFrames stored in the container are not mutated.

    Parameters
    ----------
    name : str
        Stream name used in validation errors.
    session_ids : list[str]
        Session IDs in exactly the same order as ``objects``.
    objects : list[DataObject]
        Ordered members corresponding positionally to ``session_ids``. Every
        member must be a DataFrame.
    session_coord : str
        Column name added to identify each row's source session. It must not
        collide with an existing column or ``DataFrame.index.name``.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame carrying the source-session column.

    Raises
    ------
    TypeError
        If any member is not a DataFrame.
    ValueError
        If ``DataFrame.index.name`` differs between members or ``session_coord``
        collides with an existing column or ``DataFrame.index.name``.
    """

    # === 1| Require a Homogeneous DataFrame Stream ==============================

    if not all(isinstance(obj, pd.DataFrame) for obj in objects):
        raise TypeError(
            f"stream {name!r} mixes DataFrame and non-DataFrame objects."
        )

    dataframes = [
        obj
        for obj in objects
        if isinstance(obj, pd.DataFrame)
    ]  # Runtime check above guarantees that every object is retained.

    # === 2| Validate Index Semantics and Session-Column Collisions ==============
    # The set includes ``None`` for unnamed indexes, so a named/unnamed mixture is
    # rejected alongside a mixture of different explicit names.

    index_names = {obj.index.name for obj in dataframes}
    if len(index_names) > 1:
        raise ValueError(
            f"stream {name!r} mixes DataFrame index names "
            f"{sorted(repr(index_name) for index_name in index_names)}."
        )

    if any(
        session_coord in obj.columns or session_coord == obj.index.name
        for obj in dataframes
    ):
        raise ValueError(
            f"session_coord {session_coord!r} collides with an existing "
            f"DataFrame column or index name in stream {name!r}."
        )

    # === 3| Copy and Tag Every Per-Session Piece ================================

    pieces: list[pd.DataFrame] = []

    for session_id, dataframe in zip(session_ids, dataframes, strict=True):
        piece = dataframe.copy()                                  # Preserve the original per-session member.
        piece[session_coord] = session_id                          # Broadcast one source-session ID across its rows.
        pieces.append(piece)

    # === 4| Preserve a Shared Named Index or Create a Fresh RangeIndex ===========

    preserve_index = dataframes[0].index.name is not None

    return pd.concat(
        pieces,
        ignore_index=not preserve_index,
    )

# ===============================================================================


# ===============================================================================
# 4| Combine xarray Stream Members
# ===============================================================================

def _combine_xarray_streams(
        name: str,                                                # Stream name represented by all members.
        session_ids: list[str],                                  # Session IDs in concatenation order.
        objects: list[DataObject],                               # Ordered per-session members.
        *,                                                        # Following arguments must be passed by keyword.
        dim: str,                                                 # Concatenation dimension.
        session_coord: str,                                      # New source-session coordinate name.
    ) -> xr.DataArray | xr.Dataset:
    """
    Concatenate one xarray stream across sessions and tag source elements.

    Every member must have the same exact xarray container type: DataArrays and
    Datasets are not mixed under one stream name. Each member must contain
    ``dim``. ``session_coord`` must differ from ``dim`` and from every existing
    coordinate, and for a Dataset it must also differ from every data variable.
    Auxiliary coordinates varying along ``dim`` are harmonised before
    concatenation so provenance present on only some members remains composable.

    Parameters
    ----------
    name : str
        Stream name used in validation errors.
    session_ids : list[str]
        Session IDs in exactly the same order as ``objects``.
    objects : list[DataObject]
        Ordered members corresponding positionally to ``session_ids``. Every
        member must be the same exact xarray type.
    dim : str
        Dimension along which members are concatenated.
    session_coord : str
        Coordinate attached along ``dim`` to identify each element's source. It
        must not equal ``dim`` or an existing coordinate, or a Dataset data
        variable when the members are Datasets.

    Returns
    -------
    xr.DataArray | xr.Dataset
        Concatenated xarray object carrying source-session identity.

    Raises
    ------
    TypeError
        If members are not all the same exact xarray container type.
    KeyError
        If any member lacks ``dim``.
    ValueError
        If ``session_coord`` equals ``dim`` or an existing coordinate, or matches
        a Dataset data variable.
    """

    # === 1| Require a Homogeneous xarray Stream =================================

    first = objects[0]

    if not isinstance(first, xr.DataArray | xr.Dataset):
        raise TypeError(
            f"stream {name!r} contains {type(first).__name__}; expected "
            "DataArray / Dataset."
        )

    if not all(type(obj) is type(first) for obj in objects):
        raise TypeError(
            f"stream {name!r} mixes incompatible xarray object types."
        )

    xarray_objects = [
        obj
        for obj in objects
        if isinstance(obj, xr.DataArray | xr.Dataset)
    ]  # Runtime checks above guarantee that every object is retained.

    # === 2| Validate the Concatenation Dimension and Name Collisions ============

    for obj in xarray_objects:
        if dim not in obj.dims:
            raise KeyError(
                f"stream {name!r} has no {dim!r} dimension in one or more sessions."
            )

        if (
            session_coord == dim
            or session_coord in obj.coords
            or (isinstance(obj, xr.Dataset) and session_coord in obj.data_vars)
        ):
            raise ValueError(
                f"session_coord {session_coord!r} collides with an existing "
                f"xarray name in stream {name!r}."
            )

    # === 3| Harmonise Auxiliary Coordinates and Concatenate =====================

    xarray_objects = _harmonise_along_dim(xarray_objects, dim)
    combined = xr.concat(xarray_objects, dim=dim)

    # === 4| Build and Attach Parallel Source-Session Labels =====================

    labels = np.concatenate(
        [
            np.repeat(session_id, obj.sizes[dim])
            for session_id, obj in zip(session_ids, xarray_objects, strict=True)
        ]
    )

    return combined.assign_coords({session_coord: (dim, labels)})

# ===============================================================================


# ===============================================================================
# 5| Attach Explicit Session Levels to a Combined Stream
# ===============================================================================

def _attach_stream_levels(
        data: DataObject,                                         # Combined stream carrying source-session identity.
        sessions: Mapping[str, SessionRef],                       # Session IDs and their source references.
        *,                                                        # Following arguments must be passed by keyword.
        dim: str,                                                 # Alignment dimension for xarray streams.
        session_coord: str,                                      # Existing source-session coordinate or column.
    ) -> DataObject:
    """
    Attach every explicitly declared ``SessionRef.levels`` value to combined data.

    Level names are discovered in stable first-appearance order. Sessions may
    omit a level present on another SessionRef; those elements receive ``None``
    through the mapping used by ``attach_level``. ``SessionRef.metadata`` is not
    inspected or broadcast: only values deliberately declared as levels cross
    into the combined data object.

    Parameters
    ----------
    data : DataObject
        Combined stream already carrying source-session identity.
    sessions : Mapping[str, SessionRef]
        Session-id mapping corresponding to the contributing members. Each
        reference supplies the levels declared for its session.
    dim : str
        Alignment dimension used for xarray coordinate attachment.
    session_coord : str
        Existing source-session coordinate/column used as the mapping base.

    Returns
    -------
    DataObject
        Combined data carrying all explicit session levels.

    Raises
    ------
    ValueError
        If an explicit level name collides with ``session_coord``. Errors from
        ``attach_level`` propagate unchanged.
    """

    # === 1| Discover Level Names in Stable First-Appearance Order ================

    level_names: list[str] = []

    for session in sessions.values():
        for level_name in session.levels:
            if level_name not in level_names:
                level_names.append(level_name)

    # === 2| Attach Each Level through the Existing Session Identity ==============

    for level_name in level_names:
        if level_name == session_coord:
            raise ValueError(
                f"session level {level_name!r} collides with session_coord."
            )

        session_to_value = {
            session_id: session.levels.get(level_name)
            for session_id, session in sessions.items()
        }

        data = attach_level(
            data,
            level_name,
            session_to_value,
            dim=dim,
            base=session_coord,
        )

    # === 3| Return the Fully Labelled Combined Stream ============================

    return data

# ===============================================================================


################################################################################
# StreamContainer Bound Operations
################################################################################


# ===============================================================================
# 1| Add One Session's Instance of this Stream
# ===============================================================================


def _add_stream_member(
    self: "StreamContainer",  # Container receiving another session member.
    session_id: str,  # Selection key identifying the source session.
    session: SessionRef,  # Source reference carrying path, levels, and metadata.
    stream: DataObject,  # This named stream's object for the source session.
) -> "StreamContainer":  # Same container, allowing chained additions.
    """
    Add one session's data object to this stream container.

    A container may hold at most one member for each session ID. Existing members
    are never replaced implicitly: duplicate insertion raises before either
    internal mapping is changed. The object is runtime-validated, then the stream
    and its matching ``SessionRef`` are stored under the same key.

    Parameters
    ----------
    self : StreamContainer
        Container being modified.
    session_id : str
        Session identifier from the selection mapping. This same ID is later
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
        The same container instance after insertion, allowing chained calls.

    Raises
    ------
    ValueError
        If ``session_id`` is already present.
    TypeError
        If ``stream`` is not a supported ``DataObject``.
    """

    # === 1| Reject Duplicate Membership for the Same Session =====================

    if session_id in self.streams:  # Never replace an existing session member implicitly.
        raise ValueError(f"session {session_id!r} is already present in stream {self.name!r}.")

    # === 2| Enforce the Supported Stream Type Contract ===========================

    _validate_stream_object(
        stream,
        stream_name=self.name,
        session_id=session_id,
    )

    # === 3| Store the DataObject and Matching SessionRef Together ================

    self.streams[session_id] = stream  # Insertion order becomes cross-session combination order.
    self.sessions[session_id] = session  # Same key preserves direct member->SessionRef correspondence.

    return self  # Enables ``container.add(...).add(...)`` when convenient.


# ===============================================================================


# ===============================================================================
# 2| Return Contributing Session IDs in Combination Order
# ===============================================================================


def _get_stream_session_ids(
    self: "StreamContainer",  # Container being inspected.
) -> list[str]:  # Returns session IDs in dictionary insertion order.
    """
    Return the session IDs contributing this stream, in insertion order.

    The property exposes a new list rather than the live dictionary view, so a
    caller cannot mutate container membership through the returned value.

    Parameters
    ----------
    self : StreamContainer
        Container being inspected.

    Returns
    -------
    list[str]
        Ordered copy of session IDs. This order is also the order in which the
        corresponding per-session objects are concatenated by ``combine``.
    """

    return list(self.streams)  # Standard dict iteration preserves insertion order.


# ===============================================================================


# ===============================================================================
# 3| Return the Number of Sessions Contributing this Stream
# ===============================================================================


def _count_stream_members(
    self: "StreamContainer",  # Container being inspected.
) -> int:  # Returns number of per-session members.
    """
    Return how many selected sessions contribute this stream.

    This counts stored per-session members, not the number of rows or xarray
    elements those members contain.

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
# 4| Combine this Named Stream Across All Contributing Sessions
# ===============================================================================


def _combine_stream_container(
    self: "StreamContainer",  # Container whose members should be concatenated.
    *,
    dim: str = "Time",  # Concatenation/alignment dimension for xarray members.
    session_coord: str = "session",  # New per-element source-session name.
) -> DataObject:  # Combined stream carrying session identity and explicit levels.
    """
    Concatenate this stream across sessions and attach session/level identity.

    DataFrame and xarray streams require different mechanics but produce the same
    conceptual result: one combined object where every row/element can still be
    traced to its source session and hierarchy levels.

    DataFrame indexes receive special treatment. Every member must have the same
    ``DataFrame.index.name``. A shared named index is preserved because it may
    carry meaningful per-session values required by later slicing. When every
    index is unnamed, concatenation creates a fresh ``RangeIndex``. Mixing named
    and unnamed indexes, or mixing different explicit names, is rejected rather
    than silently choosing one interpretation. Per-level ``MultiIndex.names`` are
    not inspected separately.

    Combination does not mutate the stored members. DataFrames are copied before
    the session column is added; xarray concatenation and coordinate assignment
    create derived objects. Only explicit ``SessionRef.levels`` values are
    broadcast. Arbitrary ``SessionRef.metadata`` remains available through
    ``self.sessions`` but is not duplicated across the data.

    Parameters
    ----------
    self : StreamContainer
        Container holding one named stream and its matching ``SessionRef``
        objects.
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

    Raises
    ------
    ValueError
        If the container is empty or a requested name collides with existing
        data. Type-specific validation errors propagate from the processing
        helpers.
    TypeError
        If members do not form one supported homogeneous stream family.
    KeyError
        If an xarray member lacks ``dim``.
    """

    # === 1| Reject Combination of an Empty Container =============================

    if not self.streams:  # An empty result would have no type/dimension semantics to infer safely.
        raise ValueError(f"stream {self.name!r} has no session data to combine.")

    # === 2| Capture Ordered Members and Establish the Stream Family ==============

    session_ids = self.session_ids  # Use one order for objects, labels, and level mappings.
    objects = list(self.streams.values())  # Per-session instances of this named stream.
    first = objects[0]  # First object determines DataFrame vs xarray combination branch.

    # === 3| Delegate Type-Specific Combination ===================================

    if isinstance(first, pd.DataFrame):
        combined: DataObject = _combine_dataframe_streams(
            self.name,
            session_ids,
            objects,
            session_coord=session_coord,
        )

    elif isinstance(first, xr.DataArray | xr.Dataset):
        combined = _combine_xarray_streams(
            self.name,
            session_ids,
            objects,
            dim=dim,
            session_coord=session_coord,
        )

    else:
        raise TypeError(f"stream {self.name!r} contains {type(first).__name__}; expected DataArray / Dataset / DataFrame.")

    # === 4| Attach Explicit Session Levels and Return =============================

    return _attach_stream_levels(
        combined,
        self.sessions,
        dim=dim,
        session_coord=session_coord,
    )


# ===============================================================================


################################################################################
# StreamContainer
################################################################################


# ===============================================================================
# 1| StreamContainer (One Named Stream Gathered Across Sessions)
# ===============================================================================


@dataclass(init=False)
class StreamContainer:
    """
    Gather one named stream from every selected session that provides it.

    A container represents ONE stream name, not one session. For example, after
    per-session loading there may be ``'trials'`` data from sessions A/B/C but
    ``'dlc:position'`` only from A/C. Those become two separate containers with
    different contributing session sets.

    ``streams`` and ``sessions`` use identical session ID keys. The former stores
    the actual ``DataObject`` values; the latter stores each matching
    ``SessionRef`` so combination can attach explicitly declared levels while
    retaining access to unbroadcast metadata.

    ``dataclass(init=False)`` is intentional. Dataclass-generated representation
    and equality remain useful, while the explicit constructor can defensively
    copy and validate the paired mappings. The other public operations are
    implemented by the focused module-level functions above and bound directly in
    the compact declarations at the end of the class.

    Parameters
    ----------
    name : str
        Stream name represented by the container, e.g. ``'trials'`` or
        ``'dlc:position'``.
    streams : dict[str, DataObject] | None
        Optional initial mapping from session ID to that session's instance of
        this stream.
    sessions : dict[str, SessionRef] | None
        Optional matching mapping from session ID to ``SessionRef``. Its key set
        must exactly match ``streams`` so every data object has an associated
        source-session description.
    """

    name: str  # Single stream name represented by every member of this container.
    streams: dict[str, DataObject] = field(default_factory=dict)  # Session ID -> per-session stream object.
    sessions: dict[str, SessionRef] = field(default_factory=dict)  # Session ID -> matching source reference.

    def __init__(
        self,
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

        Raises
        ------
        ValueError
            If ``streams`` and ``sessions`` do not contain identical session IDs.
        TypeError
            If any initial stream member is not a supported ``DataObject``.
        """

        # === 1| Store Independent Dictionary Copies ==================================

        self.name = name  # Combination preserves this stream identity.
        self.streams = {} if streams is None else dict(streams)  # Avoid aliasing caller-owned membership dictionary.
        self.sessions = {} if sessions is None else dict(sessions)  # Keep session descriptions keyed identically to stream members.

        # === 2| Validate Initial Members ==============================================

        _validate_container_members(
            self.name,
            self.streams,
            self.sessions,
        )

    add = _add_stream_member  # Add one new session's contribution; duplicates raise.
    session_ids = property(_get_stream_session_ids)  # Read-only ordered session ID view.
    __len__ = _count_stream_members  # Number of contributing sessions.
    combine = _combine_stream_container  # Cross-session combination + level attachment.


# ===============================================================================
