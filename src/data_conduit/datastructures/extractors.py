"""
Extract session metadata from session directory names.
------------------------------------------------------

Description:
    ``extractors.py`` contains deliberately small objects used by the selection
    stage to derive additional metadata from a session path. An extractor does
    one thing: given a session directory, return one metadata value under a
    known key.

    Extractors are separate from hierarchy ``levels``. The directory walk itself
    knows which intermediate folders represent mouse / phase / day and stores
    those values in ``SessionRef.levels``. Extractors instead describe properties
    of the session directory itself, such as its folder-name label or a datetime
    encoded in that name; these values are stored in ``SessionRef.metadata``.

    ``LabelExtractor`` defines the common contract. Concrete subclasses implement
    only ``extract``; the shared ``label_for`` helper then pairs the result with
    the extractor's metadata key. Method implementations are defined as top-level
    functions immediately before the class that exposes them, keeping the
    implementation separate while making the class API explicit in its body.

Contents:
--------------------------------
- LabelExtractor:             Abstract contract for one session-metadata extractor.
- NameExtractor:              Extract the final folder name as a string.
- DateTimeExtractor:          Parse a datetime from the session folder name.
- _label_extractor_extract:   Abstract value-extraction method.
- _label_extractor_label_for: Pair an extractor key with its extracted value.
- _name_extractor_init:       Configure the metadata key for NameExtractor.
- _name_extractor_extract:    Return ``Path.name``.
- _datetime_extractor_init:   Build the configured name->datetime reader.
- _datetime_extractor_extract:Apply that reader to a session path.
"""


################################################################################
# Imports
################################################################################

from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
from pathlib import Path

from ..core.datetime import (
    name_datetime_reader_for,
    read_datetime_from_name,
)

################################################################################


################################################################################
# LabelExtractor Methods
################################################################################


# ===============================================================================
# 1| Extract One Metadata Value (Abstract)
# ===============================================================================


@abstractmethod
def _label_extractor_extract(
    self: "LabelExtractor",  # Extractor instance being applied.
    session_path: str | Path,  # Selected session directory to inspect.
) -> object:  # Concrete subclasses return their metadata value.
    """
    Derive one metadata value from ``session_path``.

    This is the only operation a concrete ``LabelExtractor`` subclass must
    specialise. The base implementation deliberately raises because there is no
    meaningful generic rule for turning an arbitrary session path into metadata.

    Parameters
    ----------
    self : LabelExtractor
        Extractor instance being applied. Concrete subclasses may carry parsing
        configuration set during their externally defined ``__init__`` methods.
    session_path : str | Path
        Path to the selected session directory. Callers may supply either a
        string or ``Path``; concrete extractors should normalise as needed.

    Returns
    -------
    object
        Metadata value derived from the session path. The type is intentionally
        broad because different extractors may return strings, datetimes, numeric
        identifiers, or other metadata values.
    """

    # === 1| Enforce the Abstract Extractor Contract =============================
    # The decorator normally prevents direct construction of the abstract base.
    # Raising here as well makes the failure explicit if the method is invoked
    # through unusual/manual class manipulation.

    raise NotImplementedError  # No generic path->metadata conversion exists at the base level.


# ===============================================================================


# ===============================================================================
# 2| Pair an Extractor's Metadata Key with its Extracted Value
# ===============================================================================


def _label_extractor_label_for(
    self: "LabelExtractor",  # Extractor instance providing both key and extraction rule.
    session_path: str | Path,  # Selected session directory from which metadata is derived.
) -> tuple[str, object]:  # Returns ``(metadata_key, extracted_value)``.
    """
    Return the ``(key, value)`` pair contributed by this extractor.

    ``select_sessions`` stores extractor outputs in ``SessionRef.metadata``. By
    centralising the key/value pairing here, the selection loop can treat every
    extractor identically and does not need to know which concrete extractor is
    responsible for the value.

    Parameters
    ----------
    self : LabelExtractor
        Extractor being applied. ``self.key`` supplies the metadata dictionary
        key and ``self.extract`` supplies the corresponding value.
    session_path : str | Path
        Selected session directory passed unchanged to ``self.extract``.

    Returns
    -------
    tuple[str, object]
        Two-element tuple ``(self.key, self.extract(session_path))`` ready to be
        inserted into the session's metadata dictionary.
    """

    # === 1| Extract the Value and Pair it with the Declared Metadata Key =========

    return self.key, self.extract(session_path)  # Keeps key ownership inside the extractor rather than in selection.py.


# ===============================================================================


################################################################################
# LabelExtractor
################################################################################


# ===============================================================================
# 1| LabelExtractor (Abstract Session-Metadata Extractor Contract)
# ===============================================================================


class LabelExtractor(ABC):  # noqa: B024 - abstract method is attached below from a decorated function.
    """
    Define the common interface for deriving one metadata value from a session.

    Concrete extractors implement ``extract(session_path)`` and expose a string
    ``key``. The inherited ``label_for`` operation combines those two pieces into
    the ``(key, value)`` pair expected by ``select_sessions``.

    Attributes
    ----------
    key : str
        Metadata key under which this extractor's value should be stored in
        ``SessionRef.metadata``. The abstract base defaults to ``'label'`` so
        simple name-like extractors inherit a sensible convention.
    """

    key: str = "label"  # Metadata dictionary key paired with the extracted value.

    extract = _label_extractor_extract  # Abstract path -> metadata-value contract.
    label_for = _label_extractor_label_for  # Shared ``(key, value)`` convenience method.


# ===============================================================================


################################################################################
# NameExtractor Methods
################################################################################


# ===============================================================================
# 1| Configure the Metadata Key for NameExtractor
# ===============================================================================


def _name_extractor_init(
    self: "NameExtractor",  # NameExtractor instance being initialised.
    *,
    key: str = "label",  # SessionRef.metadata key used for the folder-name value.
) -> None:  # Initialisation mutates the extractor and returns nothing.
    """
    Configure which metadata key receives the session folder name.

    Parameters
    ----------
    self : NameExtractor
        Extractor instance being initialised.
    key : str
        Metadata dictionary key under which the folder name should be stored.
        Defaults to ``'label'`` to preserve the historical selection behaviour.

    Returns
    -------
    None
        The configured key is stored on ``self``; no separate value is returned.
    """

    # === 1| Store the Caller-Visible Metadata Key ================================

    self.key = key  # Extraction behaviour is fixed; only the output key is configurable.


# ===============================================================================


# ===============================================================================
# 2| Extract the Final Folder Name
# ===============================================================================


def _name_extractor_extract(
    self: "NameExtractor",  # NameExtractor instance being applied.
    session_path: str | Path,  # Selected session directory whose final component is required.
) -> str:  # Returns the session directory's basename.
    """
    Return the final path component of ``session_path`` as a string.

    Parameters
    ----------
    self : NameExtractor
        Extractor instance being applied. ``self`` is not otherwise used because
        the folder-name extraction rule has no additional configuration.
    session_path : str | Path
        Path to the selected session directory. Strings are normalised to
        ``Path`` so platform-specific path handling remains delegated to pathlib.

    Returns
    -------
    str
        Final directory name (``Path(session_path).name``), without any parent
        path components.
    """

    # === 1| Normalise to Path and Return the Basename ============================

    return Path(session_path).name  # ``Path.name`` gives the final folder component without string parsing.


# ===============================================================================


################################################################################
# NameExtractor
################################################################################


# ===============================================================================
# 2| NameExtractor (Session Folder Name -> String Metadata)
# ===============================================================================


class NameExtractor(LabelExtractor):
    """
    Extract the selected session directory's final folder name as a string.

    ``NameExtractor`` is the default extractor used by ``select_sessions``. It
    ensures every selected session receives a simple human-readable label even
    when the caller does not configure any custom metadata extraction.

    Parameters
    ----------
    key : str
        Metadata key used to store the extracted folder name. Defaults to
        ``'label'``. Changing the key changes only where the value is stored; it
        does not change how the folder name itself is extracted.
    """

    __init__ = _name_extractor_init  # Configure the metadata key used for the folder name.
    extract = _name_extractor_extract  # Return the selected session directory's final folder name.


# ===============================================================================


################################################################################
# DateTimeExtractor Methods
################################################################################


# ===============================================================================
# 1| Configure the Session-Name Datetime Parser
# ===============================================================================


def _datetime_extractor_init(
    self: "DateTimeExtractor",  # DateTimeExtractor instance being initialised.
    *,
    convention: str | None = None,  # Optional named parser convention.
    formats: str | list[str] | None = None,  # Optional explicit datetime format(s).
    pattern: str | None = None,  # Optional regex isolating the datetime token.
    dayfirst: bool = False,  # Ambiguous-date hint used by automatic parsing.
    yearfirst: bool = False,  # Ambiguous-date hint used by automatic parsing.
    key: str = "datetime",  # SessionRef.metadata key for the parsed datetime.
) -> None:  # Stores parser configuration on ``self`` and returns nothing.
    """
    Build and store the name-to-datetime reader used by this extractor.

    The parsing rule is resolved once during construction rather than rebuilt for
    every session. A named ``convention`` uses ``name_datetime_reader_for``;
    otherwise a partially configured ``read_datetime_from_name`` callable is
    stored. The two approaches deliberately share the same later ``extract``
    method.

    Parameters
    ----------
    self : DateTimeExtractor
        Extractor instance being initialised.
    convention : str | None
        Named datetime convention accepted by ``name_datetime_reader_for``.
        Mutually exclusive with ``formats`` because supplying both would create
        two competing parsing specifications.
    formats : str | list[str] | None
        Explicit datetime format or ordered list of formats passed to
        ``read_datetime_from_name`` when no named convention is requested.
    pattern : str | None
        Optional regex passed to the underlying parser to isolate a datetime
        token from the session folder name before parsing.
    dayfirst : bool
        Day-first hint passed to automatic parsing when explicit formats are not
        sufficient to determine ordering. Default False.
    yearfirst : bool
        Year-first hint passed to automatic parsing when explicit formats are not
        sufficient to determine ordering. Default False.
    key : str
        Metadata key under which the parsed datetime should be stored. Default
        ``'datetime'``.

    Returns
    -------
    None
        The configured metadata key and parsing callable are stored on ``self``.
    """

    # === 1| Reject Competing Parser Specifications ===============================
    # A named convention already determines a parsing rule. Allowing explicit
    # formats at the same time would make precedence unclear, so require callers
    # to choose exactly one explicit strategy.

    if convention is not None and formats is not None:  # Both arguments independently specify datetime parsing behaviour.
        raise ValueError("pass either convention or formats, not both.")

    # === 2| Store the Metadata Key ===============================================

    self.key = key  # Key is independent of how the datetime value itself is parsed.

    # === 3| Build the Appropriate Reusable Reader ================================
    # Resolve the parser once here. ``extract`` can then remain a tiny, uniform
    # operation that simply applies ``self._reader`` to each session path.

    if convention is not None:
        self._reader = name_datetime_reader_for(
            convention,  # Named global-time parsing convention.
            pattern=pattern,  # Optional token-isolation rule is still honoured.
        )
    else:
        self._reader = partial(
            read_datetime_from_name,  # General parser used when no named convention is selected.
            formats=formats,  # Explicit format(s), or None for the parser's automatic behaviour.
            pattern=pattern,  # Optional token-isolation regex.
            dayfirst=dayfirst,  # Preserve caller's ambiguity hint in the stored partial.
            yearfirst=yearfirst,  # Preserve caller's ambiguity hint in the stored partial.
        )


# ===============================================================================


# ===============================================================================
# 2| Parse a Datetime from the Session Directory Name
# ===============================================================================


def _datetime_extractor_extract(
    self: "DateTimeExtractor",  # Configured DateTimeExtractor containing ``self._reader``.
    session_path: str | Path,  # Selected session directory whose name encodes a datetime.
) -> datetime:  # Returns the parsed Python datetime object.
    """
    Parse and return the datetime encoded in ``session_path``.

    Parameters
    ----------
    self : DateTimeExtractor
        Configured extractor. Construction has already resolved the parsing rule
        into ``self._reader``.
    session_path : str | Path
        Selected session directory passed to the configured datetime-name reader.
        The underlying utility is responsible for extracting the relevant name
        token and parsing it according to the configured convention/formats.

    Returns
    -------
    datetime
        Parsed Python ``datetime`` represented by the session directory name.
    """

    # === 1| Apply the Reader Resolved During Construction ========================

    return self._reader(session_path)  # Reusing one configured reader avoids reconstructing parsing rules per session.


# ===============================================================================


################################################################################
# DateTimeExtractor
################################################################################


# ===============================================================================
# 3| DateTimeExtractor (Session Folder Name -> datetime Metadata)
# ===============================================================================


class DateTimeExtractor(LabelExtractor):
    """
    Parse a ``datetime`` from a selected session directory name.

    The actual parsing logic is delegated to the existing global-time datetime
    utilities. The extractor simply stores the chosen parsing configuration once
    and exposes the result through the same ``LabelExtractor`` interface as any
    other session metadata source.

    Parameters
    ----------
    convention : str | None
        Optional named datetime convention understood by
        ``name_datetime_reader_for``. This is mutually exclusive with explicit
        ``formats`` because both arguments independently specify how the name
        should be parsed.
    formats : str | list[str] | None
        Optional explicit ``datetime.strptime`` format or ordered list of formats
        to try. If None and ``convention`` is also None, the lower-level datetime
        reader performs its normal automatic parsing behaviour.
    pattern : str | None
        Optional regular-expression pattern used by the datetime utilities to
        isolate the datetime-containing token from a longer/noisier folder name.
    dayfirst : bool
        Hint for ambiguous automatically parsed dates. ``True`` means interpret
        the day before the month where the underlying parser needs disambiguation.
        Default False.
    yearfirst : bool
        Hint for ambiguous automatically parsed dates. ``True`` means interpret
        the year first where the underlying parser needs disambiguation. Default
        False.
    key : str
        Metadata key under which the parsed ``datetime`` is stored. Defaults to
        ``'datetime'``.
    """

    key: str = "datetime"  # Default SessionRef.metadata key for the parsed datetime.

    __init__ = _datetime_extractor_init  # Configure and cache the session-name datetime parser.
    extract = _datetime_extractor_extract  # Apply the configured parser to one selected session path.


# ===============================================================================
