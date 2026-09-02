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

    Per-session ``StreamMap`` objects are transient during ``load``. Their members
    are regrouped immediately into per-stream containers rather than retained in
    a second permanent cache.

    Processing helpers and callable implementations of the public operations are
    defined outside ``DataStructure``. The constructor remains inside the class
    because it establishes configuration and runtime state; the other operations
    are bound directly in the compact class body.

Contents:
--------------------------------
- _build_stream_containers:       Load sessions and group like-named streams.
- _combine_stream_containers:     Combine every stream-specific container.
- _require_loaded_data:           Validate and return the loaded StreamMap.
- _select_datastructure_sessions: Execute only the selection stage.
- _load_datastructure_streams:    Execute the full orchestration pipeline.
- _get_loaded_stream_keys:        Return loaded stream names.
- _get_loaded_stream:             Return one combined stream by name.
- _iterate_loaded_stream_names:   Iterate loaded stream names.
- _count_loaded_streams:          Return the number of combined streams.
- DataStructure:                  Store configuration and loaded pipeline state.
"""


################################################################################
# Imports
################################################################################

from collections.abc import Iterator, KeysView, Mapping, Sequence
from pathlib import Path
from typing import Any

from .catalog_core import StreamCatalog
from .extractors import LabelExtractor
from .selection import SessionRef, select_sessions
from .streams import DataObject, StreamContainer, StreamMap

################################################################################


################################################################################
# DataStructure Processing Helpers
################################################################################


# ===============================================================================
# 1| Load Sessions and Group Like-Named Streams
# ===============================================================================


def _build_stream_containers(
    sessions: Mapping[str, SessionRef],  # Selected session IDs and references.
    catalog: StreamCatalog,  # Per-session loading and configuration recipe.
) -> dict[str, StreamContainer]:  # One cross-session container per stream name.
    """
    Load every selected session and group like-named streams together.

    ``StreamCatalog.read_session`` returns ``{stream_name: DataObject}`` for one
    session. By walking the surrounding session mapping, this helper transposes
    the conceptual ``session -> streams`` orientation into
    ``stream -> contributing sessions`` without retaining a second per-session
    cache.

    A stream may not be produced for every session, for example when an optional
    reader has no source file. Only that stream's container omits the session;
    other streams from the same session remain unaffected.

    Parameters
    ----------
    sessions : Mapping[str, SessionRef]
        Selected sessions in loading order. Mapping keys become the source-session
        identifiers attached to combined data.
    catalog : StreamCatalog
        Reader/configurator recipe applied independently to each session path.

    Returns
    -------
    dict[str, StreamContainer]
        One container per encountered stream name. Stream order follows first
        appearance across the ordered session walk; member order follows session
        order.
    """

    # === 1| Initialise the Stream-Oriented Grouping =============================

    containers: dict[str, StreamContainer] = {}

    # === 2| Load Each Session Independently =====================================

    for session_id, session in sessions.items():
        stream_map = catalog.read_session(session.path)

        # === 2.1| Add Each Result to its Like-Named Container ====================
        # The first occurrence establishes the stream's position in the output;
        # later occurrences extend that container in session order.

        for stream_name, stream in stream_map.items():
            if stream_name not in containers:
                containers[stream_name] = StreamContainer(stream_name)

            containers[stream_name].add(
                session_id,  # Stable selection key becomes source-session identity.
                session,  # SessionRef supplies explicit levels and retained metadata.
                stream,  # This session's instance of the named stream.
            )

    # === 3| Return the Stream-Oriented Grouping =================================

    return containers


# ===============================================================================


# ===============================================================================
# 2| Combine Every Stream-Specific Container
# ===============================================================================


def _combine_stream_containers(
    containers: Mapping[str, StreamContainer],  # Stream name -> contributing sessions.
    *,
    dim: str,  # Concatenation dimension for xarray streams.
    session_coord: str,  # Source-session coordinate or column name.
) -> StreamMap:  # Final cross-session stream mapping.
    """
    Combine every stream-specific container independently.

    Different streams may contain different session subsets because optional
    sources can be absent. Independent combination preserves those subsets rather
    than requiring every selected session to expose the same stream names.

    Parameters
    ----------
    containers : Mapping[str, StreamContainer]
        Stream names mapped to containers holding only the sessions that produced
        each stream.
    dim : str
        Dimension used for xarray concatenation and level attachment. DataFrame
        containers concatenate rows but still receive this value when levels are
        attached through the shared container API.
    session_coord : str
        Coordinate or column name used for per-element source-session identity.

    Returns
    -------
    StreamMap
        Final ``{stream_name: combined DataObject}`` mapping in container insertion
        order.
    """

    # === 1| Combine Each Stream Across its Contributing Sessions =================

    return {
        name: container.combine(
            dim=dim,
            session_coord=session_coord,
        )
        for name, container in containers.items()
    }


# ===============================================================================


# ===============================================================================
# 3| Validate Loaded Data Before Dict-Like Access
# ===============================================================================


def _require_loaded_data(
    data: StreamMap | None,  # Final combined data, or None before successful load.
) -> StreamMap:  # Validated loaded mapping with Optional removed from its type.
    """
    Return loaded data or raise when no full load has completed.

    ``None`` means the datastructure has not completed ``load`` since construction
    or its latest standalone selection. An empty dictionary is different: it is a
    valid loaded result containing no streams and must remain accessible.

    Parameters
    ----------
    data : StreamMap | None
        Current final-data state.

    Returns
    -------
    StreamMap
        The same loaded mapping passed by the caller.

    Raises
    ------
    RuntimeError
        If ``data`` is None.
    """

    if data is None:
        raise RuntimeError("call .load() before accessing data.")

    return data


# ===============================================================================


################################################################################
# DataStructure Bound Operations
################################################################################


# ===============================================================================
# 1| Execute Only the Session Selection Stage
# ===============================================================================


def _select_datastructure_sessions(
    self: "DataStructure",  # Configured DataStructure whose selection rules should be applied.
) -> dict[str, SessionRef]:  # Returns and stores selected SessionRefs in deterministic walk order.
    """
    Select session directories using the configuration stored on ``self``.

    Selection performs no reader, configurator, or combination work. On success,
    the selected mapping is stored in ``self.sessions`` and any previous
    ``containers`` and ``data`` are invalidated because they may describe a
    different selection.

    Parameters
    ----------
    self : DataStructure
        Configured datastructure supplying root, depth, include/exclude rules,
        level names/selectors, and extractor configuration.

    Returns
    -------
    dict[str, SessionRef]
        Ordered selection mapping. This is the same mapping stored in
        ``self.sessions``.
    """

    # === 1| Forward Stored Configuration to the Standalone Selection Function ====

    sessions = select_sessions(
        self.root,  # Path-normalised root stored during construction.
        depth=self.depth,  # Number of intermediate hierarchy levels.
        include=self.include,  # Session-folder allow-list or None.
        exclude_names=self.exclude,  # Selection's public argument uses the explicit ``exclude_names`` name.
        extractors=self.extractors,  # Metadata extractor configuration.
        level_names=self.level_names or None,  # Empty tuple means "no explicit names" -> selection defaults.
        **self.level_selectors,  # Intermediate hierarchy filters.
    )

    # === 2| Commit Selection and Invalidate Results from a Previous Load =========
    # State changes only after selection succeeds, so a selection error leaves the
    # preceding state intact.

    self.sessions = sessions
    self.containers = {}
    self.data = None

    # === 3| Return the Same Mapping Stored for Introspection ======================
 
    return self.sessions


# ===============================================================================


# ===============================================================================
# 2| Execute Selection -> Per-Session Loading -> Regrouping -> Combination
# ===============================================================================


def _load_datastructure_streams(
    self: "DataStructure",  # Configured DataStructure whose full pipeline should execute.
    *,
    dim: str = "Time",  # Alignment dimension used by xarray StreamContainers.
    session_coord: str = "session",  # Coordinate/column name used for source-session identity.
) -> StreamMap:  # Returns final cross-session combined stream mapping.
    """
    Run the complete datastructure pipeline and return the combined streams.

    This operation performs a fresh selection, builds stream-oriented containers,
    combines those containers independently, and commits the resulting runtime
    state only after every stage succeeds. A failure therefore leaves the most
    recent successful ``sessions``, ``containers``, and ``data`` intact.

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

    Raises
    ------
    ValueError
        If the fresh selection contains no sessions.
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

    # === 2| Load Sessions and Regroup Like-Named Streams =========================

    containers = _build_stream_containers(
        sessions,
        self.catalog,
    )

    # === 3| Combine Every Stream Independently ==================================

    data = _combine_stream_containers(
        containers,
        dim=dim,
        session_coord=session_coord,
    )

    # === 4| Commit the Complete Runtime State ===================================

    # These assignments deliberately occur after every reader, configurator, and
    # combination has succeeded, preserving the previous complete result on error.
    self.sessions = sessions
    self.containers = containers
    self.data = data

    # === 5| Return the Final Combined StreamMap ===================================

    return data


# ===============================================================================


# ===============================================================================
# 3| Return the Names of Loaded Combined Streams
# ===============================================================================


def _get_loaded_stream_keys(
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
        Live dictionary keys view over the current loaded mapping, preserving
        final stream insertion order. A later ``load`` replaces ``self.data``;
        an earlier view remains attached to the mapping from which it was created.

    Raises
    ------
    RuntimeError
        If no full load has completed since construction or the latest standalone
        selection.
    """

    data = _require_loaded_data(self.data)
    return data.keys()


# ===============================================================================


# ===============================================================================
# 4| Return One Loaded Combined Stream by Name
# ===============================================================================


def _get_loaded_stream(
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

    Raises
    ------
    RuntimeError
        If no full load has completed since construction or the latest standalone
        selection.
    KeyError
        If data is loaded but ``key`` is not a final stream name.
    """

    data = _require_loaded_data(self.data)
    return data[key]  # Preserve the ordinary dictionary error for an unknown stream.


# ===============================================================================


# ===============================================================================
# 5| Iterate over Loaded Stream Names
# ===============================================================================


def _iterate_loaded_stream_names(
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
        Iterator over stream-name keys in final insertion order.

    Raises
    ------
    RuntimeError
        If no full load has completed since construction or the latest standalone
        selection.
    """

    data = _require_loaded_data(self.data)
    return iter(data)


# ===============================================================================


# ===============================================================================
# 6| Return the Number of Loaded Combined Streams
# ===============================================================================


def _count_loaded_streams(
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

    Raises
    ------
    RuntimeError
        If no full load has completed since construction or the latest standalone
        selection.
    """

    data = _require_loaded_data(self.data)
    return len(data)


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
    result. Construction performs no experimental-data I/O. ``select`` updates
    the selected sessions and invalidates any older loaded result. ``load``
    performs a fresh selection and commits sessions, containers, and combined data
    only after the complete pipeline succeeds.

    Construction is defined directly in the class because it establishes
    configuration and runtime state. Processing helpers and the callable
    implementations of the remaining operations are defined above, then bound
    directly under their public names in the compact class body.

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

    def __init__(
        self,
        root: str | Path,  # Root directory containing the experiment hierarchy.
        catalog: StreamCatalog,  # Per-session loading and configuration recipe.
        *,
        depth: int = 0,  # Number of intermediate levels before session folders.
        level_names: Sequence[str] | None = None,  # Optional names for hierarchy levels.
        include: Sequence[str] | None = None,  # Optional session-folder allow-list.
        exclude: Sequence[str] | None = None,  # Optional session-folder deny-list.
        extractors: LabelExtractor | Sequence[LabelExtractor] | None = None,  # Optional metadata extractor(s).
        **level_selectors: Any,  # Optional l{n}_selector hierarchy filters.
    ) -> None:
        """
        Store configuration and initialise empty runtime state without doing I/O.

        Parameters
        ----------
        self : DataStructure
            Instance being initialised.
        root : str | Path
            Root directory from which session selection begins. It is normalised
            to ``Path`` once so every later selection uses one filesystem type.
        catalog : StreamCatalog
            Per-session loading recipe applied independently to every selected
            session path.
        depth : int
            Number of intermediate directory levels between ``root`` and session
            folders. Passed unchanged to ``select_sessions``. Default 0.
        level_names : Sequence[str] | None
            Names assigned positionally to intermediate hierarchy levels. A
            defensive tuple copy preserves order across repeated selections. None
            requests the default ``level_{i}`` names from selection.
        include : Sequence[str] | None
            Session-folder allow-list forwarded to ``select_sessions``.
        exclude : Sequence[str] | None
            Session-folder deny-list forwarded as ``exclude_names``. It is
            mutually exclusive with ``include`` when selection runs.
        extractors : LabelExtractor | Sequence[LabelExtractor] | None
            Session metadata extraction configuration forwarded unchanged to
            ``select_sessions``.
        **level_selectors : Any
            Intermediate hierarchy selectors such as ``l0_selector`` and
            ``l1_selector``. They are retained for every later selection.

        Returns
        -------
        None
            Configuration and empty ``sessions``, ``containers``, and ``data``
            state are stored on ``self``.
        """

        # === 1| Normalise and Store Reusable Configuration =======================
        # ``root`` and ``level_names`` receive stable representations here; the
        # remaining objects are retained for repeated forwarding to selection.

        self.root = Path(root)
        self.catalog = catalog
        self.depth = depth
        self.level_names = tuple(level_names) if level_names is not None else ()
        self.include = include
        self.exclude = exclude
        self.extractors = extractors
        self.level_selectors = level_selectors

        # === 2| Initialise Runtime State Separately from Configuration ============
        # These attributes describe only the latest successful selection/load.
        # ``None`` means there is no current loaded result; ``{}`` is valid data.

        self.sessions: dict[str, SessionRef] = {}
        self.containers: dict[str, StreamContainer] = {}
        self.data: StreamMap | None = None

    select = _select_datastructure_sessions  # Select sessions and invalidate older loaded state.
    load = _load_datastructure_streams  # Run the complete pipeline and commit a successful result.
    keys = _get_loaded_stream_keys  # Dictionary-style view of loaded stream names.
    __getitem__ = _get_loaded_stream  # Retrieve one combined stream by name.
    __iter__ = _iterate_loaded_stream_names  # Iterate loaded names in insertion order.
    __len__ = _count_loaded_streams  # Count combined streams in the loaded result.


# ===============================================================================
