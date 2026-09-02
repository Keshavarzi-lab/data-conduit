"""
Define the per-session loading recipe: readers + configurators -> StreamMap.
----------------------------------------------------------------------------

Description:
    ``catalog_core.py`` defines the reusable recipe that turns ONE selected
    session directory into the named streams used by the rest of the
    datastructure pipeline.

    The process has three explicit stages:
      1. READERS load independent source objects. Each reader's registered name
         becomes that object's initial key in the per-session mapping.
      2. CONFIGURATORS receive the complete mapping and may align or derive
         objects, replace values, and add, remove, or rename keys using
         information from any loaded source.
      3. ``object_to_streams`` uses the final object keys as stream names or
         prefixes while normalising each value into one or more supported data
         objects, producing a ``StreamMap`` for that session.

    ``ReaderSpec`` exists because a missing optional source is not equivalent to
    an arbitrary reader failure. Optional readers may skip a genuine
    ``FileNotFoundError``; malformed data, programming errors, and failures from
    required readers propagate normally rather than being silently interpreted as
    "this source was absent".

    Substantive processing helpers and callable implementations of the public
    operations are defined outside ``StreamCatalog``. The constructor remains
    inside the class because it establishes the catalog's state; the remaining
    operations are bound directly in the compact class body.

Contents:
--------------------------------
- Reader:                           Type alias for a session-path reader callable.
- Configurator:                     Type alias for a cross-object transform callable.
- ReaderSpec:                       Store a reader together with its optionality.
- _read_session_objects:            Run every reader for one selected session.
- _configure_session_objects:       Apply configurators to the session object mapping.
- _build_stream_map:                Normalise configured objects into named streams.
- _add_catalog_reader:              Register one named reader.
- _add_catalog_configurator:        Register one ordered configurator.
- _read_catalog_session:            Execute the complete recipe for one session.
- _get_catalog_reader_names:        Return registered reader names in order.
- _get_catalog_configurator_names:  Return configurator names in order.
- _represent_stream_catalog:        Compact introspection representation.
- StreamCatalog:                    Ordered per-session reader/configurator recipe.
"""


################################################################################
# Imports
################################################################################

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .streams import StreamMap, object_to_streams

################################################################################


################################################################################
# Types
################################################################################

# A reader receives ONE selected session directory and returns one source object.
# That object may be replaced during configuration or normalise into several
# streams, so ``object`` is intentionally broader than ``DataObject``.
Reader = Callable[[Path], object]

# A configurator sees and returns the ENTIRE named-object dictionary for one
# session. It may therefore align values across sources, derive new objects, or
# alter the keys that later become stream names or prefixes.
Configurator = Callable[[dict[str, object]], dict[str, object]]

################################################################################


################################################################################
# Reader Specification
################################################################################



# ===============================================================================
# 1| ReaderSpec (Reader Callable + Optionality)
# ===============================================================================


@dataclass(frozen=True)
class ReaderSpec:
    """
    Store one reader together with the rule for handling a missing source.

    ``ReaderSpec`` does not wrap or alter reader execution. It records only
    whether a ``FileNotFoundError`` means that the source is legitimately absent
    from this session or that loading should fail.

    Parameters
    ----------
    reader : Reader
        Callable with signature ``reader(session_path: Path) -> object``. The
        reader is responsible for locating and reading its source within the
        session directory and returning the corresponding source object.
    optional : bool
        Whether ``FileNotFoundError`` from this reader may be interpreted as
        "source not present for this session". False by default. Other exception
        types are never suppressed by ``ReaderSpec.optional``.
    """

    reader: Reader  # Path-to-object callable executed for each selected session.
    optional: bool = False  # Whether genuine source absence may be skipped.


# ===============================================================================



################################################################################
# StreamCatalog Processing Helpers
################################################################################


# ===============================================================================
# 1| Run Readers for One Session
# ===============================================================================


def _read_session_objects(
    session_path: Path,  # Selected session directory supplied to every reader.
    readers: Mapping[str, ReaderSpec],  # Ordered reader specifications.
) -> dict[str, object]:  # Loaded objects under their registered reader names.
    """
    Run every registered reader for one selected session directory.

    Readers are independent source loaders. Their registered names become the
    keys in the initial object mapping, which is completed before configuration
    begins so every configurator can work across the available sources.

    Optionality applies only to genuine source absence represented by
    ``FileNotFoundError``. Malformed data, programming errors, and every other
    exception propagate immediately so they are not silently misclassified as a
    missing optional source.

    Parameters
    ----------
    session_path : Path
        Selected session directory passed unchanged to every reader.
    readers : Mapping[str, ReaderSpec]
        Ordered mapping from reader names to ``ReaderSpec`` instances. Mapping
        iteration order is the execution order.

    Returns
    -------
    dict[str, object]
        Successfully loaded source objects under their registered reader names.
        An optional reader whose source is absent contributes no entry.
    """

    # === 1| Create the Per-Session Object Mapping ================================

    objects: dict[str, object] = {}

    # === 2| Run Every Reader in Registration Order ===============================
    # Retain each registered name as the initial object key; configurators can
    # subsequently preserve, remove, or rename it.

    for name, spec in readers.items():
        try:
            # Optionality changes only missing-source handling; successful
            # results from required and optional readers are stored identically.
            objects[name] = spec.reader(session_path)

        except FileNotFoundError as error:
            if not spec.optional:
                # Missing input from a required reader is a genuine
                # session-loading failure. Preserve the original exception and
                # traceback rather than replacing it with a catalog-level error.
                raise

            warnings.warn(
                f"optional reader {name!r} absent for session {session_path.name!r}: {error}",
                # Account for this processing-helper frame so attribution still
                # points at the external caller of ``read_session``.
                stacklevel=3,
            )

    # === 3| Return the Successfully Loaded Objects ===============================

    return objects


# ===============================================================================


# ===============================================================================
# 2| Apply Configurators to One Session's Object Mapping
# ===============================================================================


def _configure_session_objects(
    objects: dict[str, object],  # Current object mapping for this session.
    configurators: Mapping[str, Configurator],  # Ordered mapping transforms.
) -> dict[str, object]:  # Complete mapping after the final transform.
    """
    Apply every configurator sequentially to one session's objects.

    The first configurator receives the reader-output mapping; each later
    configurator receives the dictionary returned by the preceding stage. This
    makes dependencies between cross-source transformations explicit in the
    registration order.

    A configurator controls the complete mapping passed forward. It may mutate
    the input dictionary or return a replacement; in either case, it may retain,
    add, remove, or rename keys and replace their values. The final keys become
    the source names or prefixes used by ``object_to_streams``. Each return value
    is validated immediately so an invalid stage fails at its source rather than
    later during stream normalisation.

    Parameters
    ----------
    objects : dict[str, object]
        Initial mapping of registered reader names to successfully loaded source
        objects. The first configurator receives this exact dictionary.
    configurators : Mapping[str, Configurator]
        Ordered mapping of configurator names to callables. Names identify the
        stage in validation errors; mapping order is execution order.

    Returns
    -------
    dict[str, object]
        Object mapping returned by the final configurator. With no configurators,
        the original mapping is returned unchanged.

    Raises
    ------
    TypeError
        If any configurator returns something other than a dictionary.
    """

    # === 1| Apply Configurators Sequentially to the Whole Object Mapping =========
    # Keep the registered name alongside the callable so a return-contract error
    # identifies the exact stage that produced an invalid result.

    for name, configurator in configurators.items():
        # Pass the returned dictionary forward exactly as supplied: a
        # configurator is allowed to replace the mapping rather than mutate it.
        configured = configurator(objects)

        # === 1.1| Enforce the Configurator Return Contract ========================
        # Validate immediately rather than allowing an invalid value to fail later
        # during stream normalisation with a less useful error.

        if not isinstance(configured, dict):
            raise TypeError(f"configurator {name!r} must return a dict[str, object]; got {type(configured).__name__}.")

        objects = configured

    # === 2| Return the Fully Configured Mapping ==================================

    return objects


# ===============================================================================


# ===============================================================================
# 3| Convert Configured Objects into One Session StreamMap
# ===============================================================================


def _build_stream_map(
    objects: Mapping[str, object],  # Final named objects for one session.
    *,
    session_name: str,  # Session label used only in collision errors.
) -> StreamMap:  # Named supported streams for this one session.
    """
    Normalise every configured object and merge the resulting streams.

    Each mapping key is passed to ``object_to_streams`` as the source name for its
    value, so it becomes either a stream name or a prefix for several sub-streams.
    ``object_to_streams`` owns the supported object-shape rules; this helper owns
    only the session-wide merge and collision check.

    A stream-name collision is rejected rather than silently overwritten because
    two objects claiming the same stream name make the catalog recipe ambiguous.

    Parameters
    ----------
    objects : Mapping[str, object]
        Fully configured objects keyed by their final source names. These keys
        may originate from reader registrations or from configurator changes.
    session_name : str
        Session label included in duplicate-stream errors. It does not affect the
        names or contents of the produced streams.

    Returns
    -------
    StreamMap
        Dictionary ``{stream_name: DataObject}`` containing the configured and
        normalised streams for this single session.

    Raises
    ------
    TypeError
        If ``object_to_streams`` rejects a configured object or produced value.
    ValueError
        If two configured objects produce the same final stream name.
    """

    # === 1| Initialise the Per-Session StreamMap =================================

    streams: StreamMap = {}

    # === 2| Convert Every Configured Object into Named Supported Streams =========
    # Delegate all supported wrapper and data-object shapes to
    # ``object_to_streams`` so conversion rules have one authoritative home.

    for name, obj in objects.items():
        produced = object_to_streams(
            obj,
            name,
            prefix_data_arrays=True,  # Isolate generic sub-names by source.
        )

        # === 2.1| Reject Duplicate Stream Names ==================================
        # ``dict.update`` would silently overwrite an earlier stream. A collision
        # means the catalog recipe is ambiguous and should be fixed explicitly.

        collisions = set(streams).intersection(produced)

        if collisions:
            raise ValueError(f"duplicate stream names produced in session {session_name!r}: {sorted(collisions)}.")

        # === 2.2| Merge the Non-Colliding Stream Fragment =========================

        streams.update(produced)

    # === 3| Return the Per-Session StreamMap =====================================

    # Preserve the per-session orientation here. ``DataStructure`` later regroups
    # like-named streams across all selected sessions.
    return streams


# ===============================================================================


################################################################################
# StreamCatalog Bound Operations
################################################################################


# ===============================================================================
# 1| Register One Named Reader
# ===============================================================================


def _add_catalog_reader(
    self: "StreamCatalog",  # Catalog receiving the reader.
    name: str,  # Initial key for this reader's loaded object.
    reader: Reader,  # Session-path-to-object callable.
    *,
    optional: bool = False,  # Whether FileNotFoundError may be skipped.
) -> "StreamCatalog":  # Returns ``self`` to support chained registration.
    """
    Register one reader under a unique source name.

    Parameters
    ----------
    self : StreamCatalog
        Catalog being modified.
    name : str
        Unique source name used as the initial key in the reader-output mapping.
        If configurators preserve that key, it later becomes the stream name or
        the prefix for any sub-streams produced from its value. Configurators may
        instead remove or rename it before normalisation.
    reader : Reader
        Callable accepting a ``Path`` to the session directory and returning one
        source object.
    optional : bool
        If True, a ``FileNotFoundError`` raised by this reader is interpreted as
        a legitimately absent source for that session and skipped with a warning.
        False means the error propagates. No other exception type is suppressed.

    Returns
    -------
    StreamCatalog
        The same catalog instance, allowing chained calls such as
        ``catalog.add_reader(...).add_reader(...)``.

    Raises
    ------
    ValueError
        If ``name`` is already registered as a reader.
    """

    # === 1| Prevent Ambiguous Reader Replacement ================================
    # Configurators may depend on registered object keys, so silently replacing a
    # reader under an existing name could change the meaning of later stages.

    if name in self._readers:
        raise ValueError(f"a reader named {name!r} is already in the catalog.")

    # === 2| Store the Callable Together with Optionality =========================

    self._readers[name] = ReaderSpec(
        reader=reader,  # Preserve the supplied loading operation unchanged.
        optional=optional,  # Keep absence policy local to this source.
    )

    return self  # Support fluent catalog declarations.


# ===============================================================================


# ===============================================================================
# 2| Register One Ordered Configurator
# ===============================================================================


def _add_catalog_configurator(
    self: "StreamCatalog",  # Catalog receiving the configurator.
    name: str,  # Unique stage name used in order and diagnostics.
    configurator: Configurator,  # Whole-mapping transformation.
) -> "StreamCatalog":  # Returns ``self`` to support chained registration.
    """
    Register one configurator to run after all readers have completed.

    A configurator's registered name identifies the processing stage for
    ordering, introspection, and validation errors. It does not determine an
    output object or stream name. The keys in the dictionary returned by the
    configurator do: a stage may retain, add, remove, or rename those keys and
    may replace any associated values.

    Parameters
    ----------
    self : StreamCatalog
        Catalog being modified.
    name : str
        Unique configurator identifier. Dictionary insertion order determines
        execution order, so registration order is semantically meaningful. The
        name also identifies the stage if it violates the return contract.
    configurator : Configurator
        Callable with shape ``dict[str, object] -> dict[str, object]``. It sees
        every object currently available for the session, which enables
        cross-source alignment and derived-object construction.

    Returns
    -------
    StreamCatalog
        The same catalog instance for optional chained registration.

    Raises
    ------
    ValueError
        If ``name`` is already registered as a configurator.
    """

    # === 1| Prevent Silent Replacement of an Ordered Processing Stage ============
    # Reusing a name would discard one transformation while retaining its
    # position, obscuring the declared processing recipe.

    if name in self._configurators:
        raise ValueError(f"a configurator named {name!r} is already in the catalog.")

    # === 2| Append the Configurator to the Ordered Recipe ========================

    self._configurators[name] = configurator
    return self  # Support fluent catalog declarations.


# ===============================================================================


# ===============================================================================
# 3| Apply Readers + Configurators + Stream Normalisation to One Session
# ===============================================================================


def _read_catalog_session(
    self: "StreamCatalog",  # Catalog recipe being applied.
    session_path: str | Path,  # Selected session directory supplied to every reader.
) -> StreamMap:  # Named supported streams produced for this session.
    """
    Execute the complete catalog recipe for one selected session directory.

    The operation deliberately separates three failure domains. Required reader
    errors propagate. Optional readers skip only ``FileNotFoundError`` because
    that exception can legitimately mean "this source is absent in this session".
    Configurator errors propagate because a failed transformation is not the same
    as absent data. Finally, stream-name collisions are rejected rather than
    silently overwriting one produced stream with another.

    Parameters
    ----------
    self : StreamCatalog
        Configured catalog containing the ordered readers and configurators.
    session_path : str | Path
        Session directory to process. Strings are normalised to ``Path`` once at
        the start and the same ``Path`` object is passed to every reader.

    Returns
    -------
    StreamMap
        Dictionary ``{stream_name: DataObject}`` containing the fully configured
        and normalised streams for this single session. An optional reader whose
        source is absent contributes no object and therefore no streams.
    """

    # === 1| Normalise the Session Path ===========================================

    path = Path(session_path)

    # === 2| Load Every Available Independent Source ==============================
    # Keep optional-reader mechanics in the processing helper so this bound
    # operation remains a readable description of the complete pipeline.

    objects = _read_session_objects(
        path,
        self._readers,
    )

    # === 3| Apply Ordered Cross-Object Configuration =============================
    # Each configurator sees the mapping returned by the preceding stage.

    objects = _configure_session_objects(
        objects,
        self._configurators,
    )

    # === 4| Normalise the Configured Objects into Named Streams ==================
    # The keys that remain after configuration now determine stream names and
    # multi-stream prefixes.

    return _build_stream_map(
        objects,
        session_name=path.name,
    )


# ===============================================================================


# ===============================================================================
# 4| Return Registered Reader Names in Execution Order
# ===============================================================================


def _get_catalog_reader_names(
    self: "StreamCatalog",  # Catalog being inspected.
) -> list[str]:  # Snapshot of reader keys in execution order.
    """
    Return the registered reader names in the order they will execute.

    Parameters
    ----------
    self : StreamCatalog
        Catalog being inspected.

    Returns
    -------
    list[str]
        Snapshot of the reader-name sequence. It is independent of the internal
        dictionary, so later registration does not alter an earlier result.
    """

    return list(self._readers)


# ===============================================================================


# ===============================================================================
# 5| Return Registered Configurator Names in Execution Order
# ===============================================================================


def _get_catalog_configurator_names(
    self: "StreamCatalog",  # Catalog being inspected.
) -> list[str]:  # Snapshot of configurator keys in execution order.
    """
    Return the registered configurator names in the order they will execute.

    Parameters
    ----------
    self : StreamCatalog
        Catalog being inspected.

    Returns
    -------
    list[str]
        Snapshot of the configurator-name sequence. It is independent of the
        internal dictionary and retains registration order.
    """

    return list(self._configurators)


# ===============================================================================


# ===============================================================================
# 6| Build a Compact Catalog Representation for Introspection
# ===============================================================================


def _represent_stream_catalog(
    self: "StreamCatalog",  # Catalog being represented.
) -> str:  # Compact summary of the registered stage names.
    """
    Return a compact representation of the catalog's registered processing steps.

    Parameters
    ----------
    self : StreamCatalog
        Catalog being represented.

    Returns
    -------
    str
        Human-readable representation containing ordered reader and configurator
        names without printing implementation details of the callables themselves.
    """

    return (
        f"<StreamCatalog readers={self.reader_names} "
        f"configurators={self.configurator_names}>"
    )


# ===============================================================================


################################################################################
# StreamCatalog
################################################################################


# ===============================================================================
# 1| StreamCatalog (Ordered Readers + Ordered Configurators)
# ===============================================================================


class StreamCatalog:
    """
    Define the reusable per-session recipe that produces a ``StreamMap``.

    One ``StreamCatalog`` is normally applied independently to each selected
    ``SessionRef.path``. Readers run before configurators. Reader order provides
    deterministic loading and introspection; configurator order is semantically
    significant because later stages may depend on values or keys produced by
    earlier ones.

    Construction is defined directly in the class because it establishes the
    catalog's state and registration invariants. All other public operations are
    implemented as documented module-level functions above and bound directly
    under their public names in the compact class body.

    Parameters
    ----------
    readers : Mapping[str, Reader | ReaderSpec] | None
        Optional initial mapping from source names such as ``'events'`` or
        ``'dlc'`` to readers. Each key becomes the initial name of that reader's
        object. Plain ``Reader`` callables are registered as required sources;
        explicit ``ReaderSpec`` values retain their optional-source policy.
    configurators : Mapping[str, Configurator] | None
        Optional initial configurator mapping. Mapping iteration order is
        preserved and therefore defines execution order.
    """

    def __init__(
        self,
        readers: Mapping[str, Reader | ReaderSpec] | None = None,
        configurators: Mapping[str, Configurator] | None = None,
    ) -> None:
        """
        Initialise the catalog registries and optionally seed them with entries.

        Parameters
        ----------
        self : StreamCatalog
            Catalog instance being initialised.
        readers : Mapping[str, Reader | ReaderSpec] | None
            Optional mapping from initial object names to either plain reader
            callables or ``ReaderSpec`` instances. Plain callables are registered
            as required sources. Mapping order becomes reader execution order.
        configurators : Mapping[str, Configurator] | None
            Optional mapping of configurator names to configurator callables.
            Mapping order becomes configurator execution order.

        Returns
        -------
        None
            Reader and configurator registries are stored on ``self``.
        """

        # === 1| Create Empty Ordered Registries ==================================
        # Standard dictionaries retain the declaration order needed for reader
        # execution, configurator execution, and stable introspection.

        self._readers: dict[str, ReaderSpec] = {}
        self._configurators: dict[str, Configurator] = {}

        # === 2| Register Seed Readers Through the Public Operation ================
        # Constructor-provided readers use the same callable normalisation and
        # optional-source handling as readers added incrementally.

        for name, reader in (readers or {}).items():
            if isinstance(reader, ReaderSpec):
                self.add_reader(
                    name,
                    reader.reader,
                    optional=reader.optional,
                )
            else:
                self.add_reader(name, reader)

        # === 3| Register Seed Configurators in Mapping Order =====================
        # Reuse the public operation so seeded and incremental stages follow the
        # same validation and ordering rules.

        for name, configurator in (configurators or {}).items():
            self.add_configurator(name, configurator)

    add_reader = _add_catalog_reader  # Register one uniquely named source reader.
    add_configurator = _add_catalog_configurator  # Register one ordered transform.
    read_session = _read_catalog_session  # Execute this recipe for one session.
    reader_names = property(_get_catalog_reader_names)  # Read-only ordered reader names.
    configurator_names = property(_get_catalog_configurator_names)  # Read-only ordered stage names.
    __repr__ = _represent_stream_catalog  # Compact summary of the registered recipe.


# ===============================================================================
