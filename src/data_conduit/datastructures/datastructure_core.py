"""
Orchestrate selection, per-session loading, regrouping, and stream combination.
-----------------------------------------------------------------------------

Description:
    ``datastructure_core.py`` contains the top-level ``DataStructure`` object.
    Its responsibility is orchestration rather than low-level data handling.

    The complete pipeline is:

        filesystem
            -> ``select_sessions``
            -> ``dict[str, SessionRef]``
            -> apply one ``StreamCatalog`` independently to each SessionRef.path
            -> ``dict[str, StreamMap]`` conceptually
            -> regroup like-named streams into ``StreamContainer`` objects
            -> combine each container
            -> final ``StreamMap`` of cross-session data objects.

    ``DataStructure`` intentionally delegates each specialised operation to the
    module that owns it. Selection logic remains in ``selection.py``; readers and
    configurators remain in ``catalog_core.py``; stream combination and level
    attachment remain in ``streams.py``.

    Methods are defined as top-level functions and attached to ``DataStructure``
    after the class declaration.

Contents:
--------------------------------
- DataStructure:                  Store configuration + loaded pipeline state.
- _datastructure_init:            Configure selection/catalog without doing I/O.
- _datastructure_select:          Execute only the selection stage.
- _datastructure_load:            Execute the full select/load/regroup/combine flow.
- _datastructure_require_loaded:  Guard dict-like loaded-data access.
- _datastructure_keys:            Return loaded stream names.
- _datastructure_getitem:         Return one combined stream by name.
- _datastructure_iter:            Iterate loaded stream names.
- _datastructure_len:             Return number of combined streams.
"""


################################################################################
# Imports
################################################################################

from collections.abc import Iterator, KeysView, Sequence
from pathlib import Path
from typing import Any

from .catalog_core import StreamCatalog
from .extractors import LabelExtractor
from .selection import SessionRef, select_sessions
from .streams import DataObject, StreamContainer, StreamMap

################################################################################


################################################################################
# DataStructure Methods
################################################################################


# ===============================================================================
# 1| Configure a DataStructure Without Loading Experimental Data
# ===============================================================================


def _datastructure_init(
    self: "DataStructure",  # DataStructure instance being initialised.
    root: str | Path,  # Root directory containing the experiment hierarchy.
    catalog: StreamCatalog,  # Per-session reader/configurator recipe.
    *,
    depth: int = 0,  # Number of intermediate levels before session folders.
    level_names: Sequence[str] | None = None,  # Optional names for those intermediate hierarchy levels.
    include: Sequence[str] | None = None,  # Optional session-folder allow-list.
    exclude: Sequence[str] | None = None,  # Optional session-folder deny-list.
    extractors: LabelExtractor | Sequence[LabelExtractor] | None = None,  # Optional session metadata extractor(s).
    **level_selectors: Any,  # Optional l{n}_selector filters for intermediate hierarchy levels.
) -> None:  # Stores configuration/state on ``self`` and returns nothing.
    """
    Store the datastructure configuration and initialise empty runtime state.

    No selection, reader, configurator, or combination work occurs during
    construction. Deferring I/O makes a configured ``DataStructure`` cheap to
    create, inspect, and reuse with repeated ``select``/``load`` calls.

    Parameters
    ----------
    self : DataStructure
        Instance being initialised.
    root : str | Path
        Root directory from which session selection will begin. Converted to
        ``Path`` immediately so downstream code uses one filesystem type.
    catalog : StreamCatalog
        Per-session loading recipe. The same catalog is applied independently to
        every selected session path.
    depth : int
        Number of intermediate directory levels between ``root`` and session
        folders. Passed unchanged to ``select_sessions``. Default 0.
    level_names : Sequence[str] | None
        Names assigned to intermediate hierarchy positions. Stored here as a
        tuple for stable repeated forwarding to ``select_sessions``. None means
        selection should generate default ``level_{i}`` names.
    include : Sequence[str] | None
        Session-folder include selector passed to ``select_sessions``.
    exclude : Sequence[str] | None
        Session-folder exclude selector passed as ``exclude_names`` to
        ``select_sessions``. Mutually exclusive with ``include`` at selection
        time.
    extractors : LabelExtractor | Sequence[LabelExtractor] | None
        Session metadata extraction configuration forwarded unchanged to
        ``select_sessions``.
    **level_selectors : Any
        Intermediate hierarchy selectors such as ``l0_selector`` or
        ``l1_selector``. Stored unchanged and parsed later by selection.

    Returns
    -------
    None
        Configuration plus empty ``sessions``, ``containers``, and ``data`` state
        are stored on ``self``.
    """

    # === 1| Normalise and Store Selection/Catalog Configuration ==================

    self.root = Path(root)  # Normalise root path once at object construction.
    self.catalog = catalog  # Reused for every selected session during each load.
    self.depth = depth  # Selection depth semantics are owned by select_sessions.
    self.level_names = tuple(level_names) if level_names is not None else ()  # Immutable/stable copy of caller-supplied level-name order.
    self.include = include  # Retain caller's session include rule for future selections.
    self.exclude = exclude  # Retain caller's session exclude rule for future selections.
    self.extractors = extractors  # Extractor objects/configuration remain reusable across selections.
    self.level_selectors = level_selectors  # Raw kwargs are forwarded on every selection call.

    # === 2| Initialise Runtime State Separately from Configuration ================
    # These attributes represent the most recent pipeline execution. Keeping them
    # explicit makes debugging possible without adding separate "load sessions"
    # public methods solely to expose intermediate state.

    self.sessions: dict[str, SessionRef] = {}  # Most recent selected-session mapping.
    self.containers: dict[str, StreamContainer] = {}  # Most recent per-stream cross-session groupings.
    self.data: StreamMap | None = None  # Most recent final combined data; None means load has not completed.


# ===============================================================================


# ===============================================================================
# 2| Execute Only the Session Selection Stage
# ===============================================================================


def _datastructure_select(
    self: "DataStructure",  # Configured DataStructure whose selection rules should be applied.
) -> dict[str, SessionRef]:  # Returns and stores selected SessionRefs in deterministic walk order.
    """
    Select session directories using the configuration stored on ``self``.

    This is a thin orchestration wrapper around ``select_sessions``. It exists so
    callers can inspect what would be loaded without executing readers, while the
    standalone selection function remains independently usable/testable.

    Parameters
    ----------
    self : DataStructure
        Configured datastructure supplying root, depth, include/exclude rules,
        level names/selectors, and extractor configuration.

    Returns
    -------
    dict[str, SessionRef]
        Ordered selection mapping. The same mapping is also stored in
        ``self.sessions`` for later inspection.
    """

    # === 1| Forward Stored Configuration to the Standalone Selection Function ====

    sessions = select_sessions(
        self.root,  # Validated/normalised root stored during construction.
        depth=self.depth,  # Number of intermediate hierarchy levels.
        include=self.include,  # Session-folder allow-list or None.
        exclude_names=self.exclude,  # Selection's public argument uses the explicit ``exclude_names`` name.
        extractors=self.extractors,  # Metadata extractor configuration.
        level_names=self.level_names or None,  # Empty tuple means "no explicit names" -> selection defaults.
        **self.level_selectors,  # Intermediate hierarchy filters.
    )

    # === 2| Commit Selection and Invalidate Results from a Previous Load =========

    self.sessions = sessions
    self.containers = {}
    self.data = None

    # === 3| Return the Same Mapping Stored for Introspection ======================

    return self.sessions


# ===============================================================================


# ===============================================================================
# 3| Execute Selection -> Per-Session Loading -> Regrouping -> Combination
# ===============================================================================


def _datastructure_load(
    self: "DataStructure",  # Configured DataStructure whose full pipeline should execute.
    *,
    dim: str = "Time",  # Alignment dimension used by xarray StreamContainers.
    session_coord: str = "session",  # Coordinate/column name used for source-session identity.
) -> StreamMap:  # Returns final cross-session combined stream mapping.
    """
    Run the complete datastructure pipeline and return the combined streams.

    The method intentionally regroups while loading rather than first storing a
    second permanent ``dict[str, StreamMap]`` object. Each per-session StreamMap
    is transiently available as ``stream_map``; its members are immediately added
    to the appropriate ``StreamContainer``. This preserves the conceptual
    session->streams stage without creating another long-lived wrapper solely for
    orchestration.

    Parameters
    ----------
    self : DataStructure
        Configured datastructure containing selection rules and one
        ``StreamCatalog``.
    dim : str
        xarray dimension used when each ``StreamContainer`` concatenates its
        contributing sessions. Defaults to ``'Time'``. DataFrame streams combine
        by rows and do not use this parameter for concatenation.
    session_coord : str
        Name attached to each combined element/row to identify its source
        session. Defaults to ``'session'``. The same name becomes the base used by
        ``attach_level`` for broadcasting explicit SessionRef levels.

    Returns
    -------
    StreamMap
        Final mapping ``{stream_name: combined DataObject}``. Different stream
        names may represent different subsets of sessions because optional sources
        are allowed to be absent from individual sessions.
    """

    # === 1| Select the Session Directories =======================================
    # ``load`` always performs a fresh selection so changes to the filesystem or
    # stored selection configuration are reflected consistently in the loaded
    # result rather than reusing potentially stale ``self.sessions`` state.

    sessions = select_sessions(
        self.root,
        depth=self.depth,
        include=self.include,
        exclude_names=self.exclude,
        extractors=self.extractors,
        level_names=self.level_names or None,
        **self.level_selectors,
    )

    if not sessions:  # A full load with no sessions almost always indicates bad configuration.
        raise ValueError(f"no sessions selected under {self.root} (check depth / include / exclude / level selectors).")

    # === 2| Apply the Catalog Independently to Every Selected Session =============
    # As each session returns a StreamMap, transpose orientation immediately from
    # ``session -> streams`` into ``stream -> contributing sessions`` by adding
    # each member to the corresponding StreamContainer.

    containers: dict[str, StreamContainer] = {}

    for session_id, session in sessions.items():
        stream_map = self.catalog.read_session(session.path)  # ONE session -> its named configured/normalised streams.

        # === 2.1| Regroup Like-Named Streams into Cross-Session Containers ========

        for stream_name, stream in stream_map.items():
            if stream_name not in containers:  # First occurrence of this stream creates its cross-session group.
                containers[stream_name] = StreamContainer(stream_name)

            containers[stream_name].add(
                session_id,  # Stable selection key becomes eventual per-element session identity.
                session,  # SessionRef provides levels + retained metadata.
                stream,  # This session's instance of the like-named stream.
            )

    # === 3| Combine Every StreamContainer Independently ===========================
    # Missing optional streams are naturally handled because a container contains
    # only the sessions that actually produced that stream. No global strict set
    # intersection is required.

    data: StreamMap = {
        name: container.combine(
            dim=dim,  # Forward xarray alignment dimension.
            session_coord=session_coord,  # Forward source-session identity name.
        )
        for name, container in containers.items()
    }

    # === 4| Store Intermediate/Final State for Inspection =========================

    self.sessions = sessions  # Commit all runtime state together only after every stage succeeds.
    self.containers = containers  # Allows debugging how one particular stream was assembled.
    self.data = data  # Marks the DataStructure as successfully loaded for dict-like access.

    # === 5| Return the Final Combined StreamMap ===================================

    return data


# ===============================================================================


# ===============================================================================
# 4| Guard Access that Requires a Completed ``load``
# ===============================================================================


def _datastructure_require_loaded(
    self: "DataStructure",  # DataStructure whose loaded state is being checked.
) -> None:  # Raises before load; otherwise returns nothing.
    """
    Raise a clear error when dict-like data access occurs before ``load``.

    Parameters
    ----------
    self : DataStructure
        Datastructure being checked.

    Returns
    -------
    None
        Returns silently when ``self.data`` contains a completed StreamMap.

    Raises
    ------
    RuntimeError
        If ``self.data`` is still None, meaning no successful full load has
        populated the final combined result.
    """

    if self.data is None:  # Empty StreamMap is distinct from never-loaded None.
        raise RuntimeError("call .load() before accessing data.")


# ===============================================================================


# ===============================================================================
# 5| Return the Names of Loaded Combined Streams
# ===============================================================================


def _datastructure_keys(
    self: "DataStructure",  # Loaded DataStructure being inspected.
) -> KeysView[str]:  # Returns dict keys view over the loaded StreamMap.
    """
    Return a dictionary-style view of loaded stream names.

    Parameters
    ----------
    self : DataStructure
        Datastructure whose final ``StreamMap`` should already be populated.

    Returns
    -------
    KeysView[str]
        Live keys view over ``self.data``, e.g. ``'trials'`` or
        ``'dlc:position'`` names.
    """

    self._require_loaded()  # Give a targeted error instead of ``NoneType`` attribute failures.
    return self.data.keys()


# ===============================================================================


# ===============================================================================
# 6| Return One Loaded Combined Stream by Name
# ===============================================================================


def _datastructure_getitem(
    self: "DataStructure",  # Loaded DataStructure being indexed.
    key: str,  # Stream name to retrieve from the final StreamMap.
) -> DataObject:  # Returns the combined DataObject stored under ``key``.
    """
    Return one loaded combined stream using dictionary-style indexing.

    Parameters
    ----------
    self : DataStructure
        Datastructure whose final data has been loaded.
    key : str
        Exact stream name in the final StreamMap, e.g. ``'trials'`` or
        ``'dlc:position'``.

    Returns
    -------
    DataObject
        Combined DataFrame / DataArray / Dataset stored under ``key``.
    """

    self._require_loaded()  # Explicitly distinguish "not loaded" from a missing stream key.
    return self.data[key]  # Ordinary dictionary KeyError remains appropriate for unknown stream names.


# ===============================================================================


# ===============================================================================
# 7| Iterate over Loaded Stream Names
# ===============================================================================


def _datastructure_iter(
    self: "DataStructure",  # Loaded DataStructure being iterated.
) -> Iterator[str]:  # Returns iterator over final StreamMap keys.
    """
    Iterate over loaded stream names in final StreamMap insertion order.

    Parameters
    ----------
    self : DataStructure
        Datastructure whose final data has been loaded.

    Returns
    -------
    Iterator[str]
        Iterator over stream-name keys.
    """

    self._require_loaded()
    return iter(self.data)


# ===============================================================================


# ===============================================================================
# 8| Return the Number of Loaded Combined Streams
# ===============================================================================


def _datastructure_len(
    self: "DataStructure",  # Loaded DataStructure whose stream count is requested.
) -> int:  # Returns number of entries in final StreamMap.
    """
    Return the number of combined streams in the loaded result.

    Parameters
    ----------
    self : DataStructure
        Datastructure whose final data has been loaded.

    Returns
    -------
    int
        Number of stream-name/DataObject entries in ``self.data``.
    """

    self._require_loaded()
    return len(self.data)


# ===============================================================================


################################################################################
# DataStructure
################################################################################


# ===============================================================================
# 1| DataStructure (Selection -> Catalog -> Containers -> Combined StreamMap)
# ===============================================================================


class DataStructure:
    """
    Apply one ``StreamCatalog`` across a configured selection of session folders.

    A ``DataStructure`` stores both configuration and the most recent pipeline
    result. Construction performs no experimental-data I/O. ``select`` populates
    ``sessions`` only; ``load`` reruns selection, loads each session, constructs
    cross-session ``StreamContainer`` objects, combines them, and stores the final
    ``StreamMap`` in ``data``.

    Method implementations are defined as documented top-level functions
    immediately above this class. The class body then exposes those functions
    under the normal public method names.

    Parameters
    ----------
    root : str | Path
        Root directory containing the experimental hierarchy.
    catalog : StreamCatalog
        Per-session loading/configuration recipe applied independently to every
        selected ``SessionRef.path``.
    depth : int
        Number of intermediate hierarchy levels between ``root`` and session
        directories. Default 0.
    level_names : Sequence[str] | None
        Optional names assigned positionally to those intermediate hierarchy
        levels and stored in each ``SessionRef.levels`` mapping.
    include : Sequence[str] | None
        Optional session-folder allow-list.
    exclude : Sequence[str] | None
        Optional session-folder deny-list, mutually exclusive with ``include``.
    extractors : LabelExtractor | Sequence[LabelExtractor] | None
        Session metadata extractors forwarded to ``select_sessions``.
    **level_selectors : Any
        Intermediate hierarchy selectors following the ``l{n}_selector`` naming
        convention understood by ``select_sessions``.
    """

    __init__ = _datastructure_init  # Configuration + empty runtime state.
    select = _datastructure_select  # Selection-only public operation.
    load = _datastructure_load  # Complete orchestration pipeline.
    _require_loaded = _datastructure_require_loaded  # Internal guard shared by dict-like accessors.
    keys = _datastructure_keys  # Dict-like loaded stream-name view.
    __getitem__ = _datastructure_getitem  # Dict-like ``ds['stream_name']`` access.
    __iter__ = _datastructure_iter  # Iteration over loaded stream names.
    __len__ = _datastructure_len  # Number of loaded streams.


# ===============================================================================
