"""
Define the per-session loading recipe: readers + configurators -> StreamMap.
----------------------------------------------------------------------------

Description:
    ``catalog_core.py`` defines how ONE selected session directory becomes the
    named streams used by the rest of the datastructure pipeline.

    The process has three explicit stages:
      1. READERS load independent source objects from a session directory.
      2. CONFIGURATORS receive the complete object mapping and may align, derive,
         replace, or remove objects using information from other loaded sources.
      3. ``object_to_streams`` normalises each configured object into one or more
         supported named streams, producing a ``StreamMap`` for that session.

    ``ReaderSpec`` exists because a missing optional source is not equivalent to
    an arbitrary reader failure. Optional readers may skip a genuine
    ``FileNotFoundError``; malformed data, programming errors, and failures from
    required readers propagate normally rather than being silently interpreted as
    "this stream was absent".

    Methods are defined outside ``StreamCatalog`` and attached after the class
    declaration, matching the module style used across this refactor.

Contents:
--------------------------------
- Reader:                           Type alias for a session-path reader callable.
- Configurator:                     Type alias for a cross-object transform callable.
- ReaderSpec:                       Store a reader together with its optionality.
- StreamCatalog:                    Ordered per-session reader/configurator recipe.
- _stream_catalog_init:             Initialise reader/configurator registries.
- _stream_catalog_add_reader:       Register one named reader.
- _stream_catalog_add_configurator: Register one ordered configurator.
- _stream_catalog_read_session:     Execute the complete recipe for one session.
- _stream_catalog_reader_names:     Return registered reader names in order.
- _stream_catalog_configurator_names:Return configurator names in order.
- _stream_catalog_repr:             Compact introspection representation.
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
# The returned object may later expose several actual streams after configuration
# and normalisation, so ``object`` is intentionally broader than ``DataObject``.
Reader = Callable[[Path], object]

# A configurator sees the ENTIRE object dictionary for one session. This is what
# allows cross-source operations such as aligning DLC with video timestamps or
# deriving a trials table from a separately loaded events object.
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

    ``ReaderSpec`` is intentionally small. It does not wrap or change reader
    execution; it simply records whether a ``FileNotFoundError`` means the source
    is legitimately absent for this session or should be treated as an error.

    Parameters
    ----------
    reader : Reader
        Callable with signature ``reader(session_path: Path) -> object``. The
        reader is responsible for locating/reading its source within the session
        directory and returning the corresponding source object.
    optional : bool
        Whether ``FileNotFoundError`` from this reader may be interpreted as
        "source not present for this session". False by default. Other exception
        types are never suppressed by ``ReaderSpec.optional``.
    """

    reader: Reader  # Actual path->object callable executed by StreamCatalog.
    optional: bool = False  # True only when genuine source absence is permitted.


# ===============================================================================


################################################################################
# StreamCatalog Methods
################################################################################


# ===============================================================================
# 1| Initialise an Empty/Seeded StreamCatalog
# ===============================================================================


def _stream_catalog_init(
    self: "StreamCatalog",  # StreamCatalog instance being initialised.
    readers: Mapping[str, Reader | ReaderSpec] | None = None,  # Optional initial named readers.
    configurators: Mapping[str, Configurator] | None = None,  # Optional initial ordered configurators.
) -> None:  # Stores registries on ``self``; returns nothing.
    """
    Initialise the catalog registries and optionally seed them with entries.

    Parameters
    ----------
    self : StreamCatalog
        Catalog instance being initialised.
    readers : Mapping[str, Reader | ReaderSpec] | None
        Optional mapping of reader names to either plain reader callables or
        ``ReaderSpec`` instances. Plain callables are registered as required.
        Mapping order is retained.
    configurators : Mapping[str, Configurator] | None
        Optional mapping of configurator names to configurator callables. Mapping
        order becomes configurator execution order.

    Returns
    -------
    None
        Reader/configurator registries are stored on ``self``.
    """

    # === 1| Create Empty Ordered Registries ======================================
    # Standard dictionaries preserve insertion order in supported Python versions,
    # so no additional OrderedDict abstraction is necessary.

    self._readers: dict[str, ReaderSpec] = {}  # Name -> reader + optionality.
    self._configurators: dict[str, Configurator] = {}  # Name -> cross-object transform.

    # === 2| Register Any Seed Readers Through the Public Validation Path =========
    # Use ``add_reader`` rather than assigning directly so constructor-provided
    # readers obey the exact same duplicate/normalisation rules as later additions.

    for name, reader in (readers or {}).items():
        if isinstance(reader, ReaderSpec):  # Explicit spec already contains optionality.
            self.add_reader(name, reader.reader, optional=reader.optional)
        else:
            self.add_reader(name, reader)  # Plain callable is required by default.

    # === 3| Register Any Seed Configurators in Mapping Order =====================

    for name, configurator in (configurators or {}).items():
        self.add_configurator(name, configurator)  # Reuse duplicate checks and preserve insertion order.


# ===============================================================================


# ===============================================================================
# 2| Register One Named Reader
# ===============================================================================


def _stream_catalog_add_reader(
    self: "StreamCatalog",  # Catalog receiving the reader.
    name: str,  # Unique source/object name within this catalog.
    reader: Reader,  # Callable loading one source from a session path.
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
        Unique source name used as the initial key in the per-session object
        mapping. It also becomes the base stream name/prefix if the object
        survives configuration unchanged.
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
    """

    # === 1| Prevent Ambiguous Reader Replacement ================================
    # Silent replacement could change loading semantics because later
    # configurators refer to reader/object names. Require explicit unique names.

    if name in self._readers:  # Existing registration would otherwise be overwritten silently.
        raise ValueError(f"a reader named {name!r} is already in the catalog.")

    # === 2| Store the Callable Together with Optionality =========================

    self._readers[name] = ReaderSpec(
        reader=reader,  # Preserve the caller's reader callable unchanged.
        optional=optional,  # Store missing-source policy beside the reader itself.
    )

    return self  # Chaining is convenient when declaring a catalog recipe.


# ===============================================================================


# ===============================================================================
# 3| Register One Ordered Configurator
# ===============================================================================


def _stream_catalog_add_configurator(
    self: "StreamCatalog",  # Catalog receiving the configurator.
    name: str,  # Unique identifier used for ordering/introspection.
    configurator: Configurator,  # Callable transforming the whole session object mapping.
) -> "StreamCatalog":  # Returns ``self`` to support chained registration.
    """
    Register one configurator to run after all readers have completed.

    Configurator names identify/sequence the transform itself; they do not impose
    any output object/stream name. The configurator controls the mapping it
    returns and may therefore add, remove, or replace object keys.

    Parameters
    ----------
    self : StreamCatalog
        Catalog being modified.
    name : str
        Unique configurator identifier. Dictionary insertion order determines
        execution order, so registration order is semantically meaningful.
    configurator : Configurator
        Callable with shape ``dict[str, object] -> dict[str, object]``. It sees
        every object currently available for the session, which enables
        cross-source alignment and derived-object construction.

    Returns
    -------
    StreamCatalog
        The same catalog instance for optional chained registration.
    """

    # === 1| Prevent Silent Replacement of an Ordered Processing Stage ============

    if name in self._configurators:  # Duplicate key would otherwise overwrite a stage and alter pipeline order.
        raise ValueError(f"a configurator named {name!r} is already in the catalog.")

    # === 2| Append the Configurator to the Ordered Recipe ========================

    self._configurators[name] = configurator  # Standard dict insertion order becomes execution order.
    return self


# ===============================================================================


# ===============================================================================
# 4| Apply Readers + Configurators + Stream Normalisation to One Session
# ===============================================================================


def _stream_catalog_read_session(
    self: "StreamCatalog",  # Catalog recipe being applied.
    session_path: str | Path,  # Selected session directory supplied to every reader.
) -> StreamMap:  # Returns all named supported streams for this one session.
    """
    Execute the complete catalog recipe for one selected session directory.

    The method deliberately separates three failure domains. Required reader
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
        and normalised streams for this single session. Sources absent through an
        optional reader simply contribute no streams.
    """

    # === 1| Normalise the Session Path ===========================================

    path = Path(session_path)  # Keep reader inputs consistent regardless of caller path type.

    # === 2| Run Every Reader in Registration Order ===============================
    # Readers are independent source loaders. Build the initial object mapping
    # first so configurators can later see all successfully loaded sources at once.

    objects: dict[str, object] = {}

    for name, spec in self._readers.items():
        try:
            objects[name] = spec.reader(path)  # Required/optional readers execute identically when the source exists.

        except FileNotFoundError as error:
            if not spec.optional:  # Missing required source is a genuine session-loading failure.
                raise

            warnings.warn(
                f"optional reader {name!r} absent for session {path.name!r}: {error}",
                stacklevel=2,  # Point warning attribution at the caller rather than this internal helper.
            )

    # === 3| Apply Configurators Sequentially to the Whole Object Mapping =========
    # Each configurator receives the mapping returned by the preceding stage. This
    # supports pipelines where one configurator creates information needed by a
    # later configurator while keeping all dependencies explicit in order.

    for configurator in self._configurators.values():
        objects = configurator(objects)  # Configurator may add/remove/replace object entries.

        if not isinstance(objects, dict):  # Enforce the contract immediately at the stage that violated it.
            raise TypeError(f"configurators must return a dict[str, object]; got {type(objects).__name__}.")

    # === 4| Convert Every Configured Object into Named Supported Streams =========
    # One object may yield one stream or several sub-streams. ``object_to_streams``
    # owns that object-shape normalisation so the catalog does not duplicate type
    # handling for HARP-like ``.data_arrays``, ``.df`` wrappers, xarray, etc.

    streams: StreamMap = {}

    for name, obj in objects.items():
        produced = object_to_streams(
            obj,  # Configured object to normalise.
            name,  # Object name becomes stream name or prefix.
            prefix_data_arrays=True,  # Prefix multi-member outputs to avoid common sub-name collisions.
        )

        # === 4.1| Reject Duplicate Stream Names ==================================
        # ``dict.update`` would silently overwrite an earlier stream. A collision
        # means the catalog recipe is ambiguous and should be fixed explicitly.

        collisions = set(streams).intersection(produced)  # Names already produced by another configured object.
        if collisions:
            raise ValueError(f"duplicate stream names produced in session {path.name!r}: {sorted(collisions)}.")

        streams.update(produced)  # Safe after collision check: no existing stream can be overwritten.

    # === 5| Return the Per-Session StreamMap =====================================

    return streams  # This is ONE session's orientation; DataStructure later regroups by stream.


# ===============================================================================


# ===============================================================================
# 5| Return Registered Reader Names in Execution Order
# ===============================================================================


def _stream_catalog_reader_names(
    self: "StreamCatalog",  # Catalog being inspected.
) -> list[str]:  # Returns reader keys in dictionary insertion order.
    """
    Return the registered reader names in the order they will execute.

    Parameters
    ----------
    self : StreamCatalog
        Catalog being inspected.

    Returns
    -------
    list[str]
        Copy of the reader-name sequence. Returning a list rather than the live
        dictionary view prevents callers from depending on/mutating internals.
    """

    return list(self._readers)  # ``dict`` iteration order is the reader execution order.


# ===============================================================================


# ===============================================================================
# 6| Return Registered Configurator Names in Execution Order
# ===============================================================================


def _stream_catalog_configurator_names(
    self: "StreamCatalog",  # Catalog being inspected.
) -> list[str]:  # Returns configurator keys in execution order.
    """
    Return the registered configurator names in the order they will execute.

    Parameters
    ----------
    self : StreamCatalog
        Catalog being inspected.

    Returns
    -------
    list[str]
        Copy of the ordered configurator-name sequence.
    """

    return list(self._configurators)  # Registration order and execution order are deliberately identical.


# ===============================================================================


# ===============================================================================
# 7| Build a Compact Catalog Representation for Introspection
# ===============================================================================


def _stream_catalog_repr(
    self: "StreamCatalog",  # Catalog being represented.
) -> str:  # Returns reader/configurator names in one compact string.
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
        f"<StreamCatalog readers={self.reader_names} "  # Reader property returns names in execution order.
        f"configurators={self.configurator_names}>"  # Configurator property mirrors the later processing order.
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

    One ``StreamCatalog`` is normally applied independently to every selected
    ``SessionRef.path``. Reader and configurator insertion order is preserved
    because some configurators may depend on changes made by earlier ones.

    Method implementations are defined as documented top-level functions
    immediately above this class. The class body then exposes those functions
    under the normal public method names.

    Parameters
    ----------
    readers : Mapping[str, Reader | ReaderSpec] | None
        Optional initial reader mapping. Keys are source names such as
        ``'events'`` or ``'dlc'``. Plain ``Reader`` callables are treated as
        required; explicit ``ReaderSpec`` values preserve their optional flag.
    configurators : Mapping[str, Configurator] | None
        Optional initial configurator mapping. Mapping iteration order is
        preserved and therefore defines execution order.
    """

    __init__ = _stream_catalog_init  # Construction / optional seeding.
    add_reader = _stream_catalog_add_reader  # Public reader registration API.
    add_configurator = _stream_catalog_add_configurator  # Public configurator registration API.
    read_session = _stream_catalog_read_session  # Per-session catalog execution API.
    reader_names = property(_stream_catalog_reader_names)  # Read-only ordered reader-name property.
    configurator_names = property(_stream_catalog_configurator_names)  # Read-only ordered configurator-name property.
    __repr__ = _stream_catalog_repr  # Compact introspection representation.


# ===============================================================================
