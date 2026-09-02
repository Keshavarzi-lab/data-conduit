'''
Extract session metadata from session directory names.
------------------------------------------------------

Description:
    ``extractors.py`` contains small configuration objects used by selection to
    derive additional metadata from a session path.

    The actual path-reading/parsing operations are ordinary module-level helper
    functions. ``LabelExtractor`` stores one metadata key and one configured
    reader callable; ``label_for`` applies that reader and returns the
    ``(key, value)`` pair consumed by ``select_sessions``.

    ``NameExtractor`` and ``DateTimeExtractor`` do not implement redundant
    ``extract`` methods. They only configure the common ``LabelExtractor`` with
    the appropriate reader.

Contents:
--------------------------------
- MetadataReader:          Type alias for a path -> metadata-value callable.
- _read_session_name:      Return the final folder name from a session path.
- _build_datetime_reader:  Configure a reusable session-name datetime reader.
- LabelExtractor:          Pair one metadata key with one reader callable.
- NameExtractor:           Configure LabelExtractor for folder-name metadata.
- DateTimeExtractor:       Configure LabelExtractor for parsed datetime metadata.
'''


################################################################################
# Imports
################################################################################

from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

from ..core.datetime import (
    name_datetime_reader_for,
    read_datetime_from_name,
)

################################################################################




################################################################################
# Types
################################################################################

# A metadata reader receives a selected session path and derives one value from
# it. Different readers may return different value types, hence ``object``.
MetadataReader = Callable[[str | Path], object]

################################################################################




################################################################################
# Private Helper Functions
################################################################################



#===============================================================================
# 1| Read Session Folder Name
#===============================================================================

def _read_session_name(
        session_path: str | Path,                         # Selected session directory whose final folder name is required.
    ) -> str:                                              # Returns the final path component as a string.
    '''
    Return the final folder name from a selected session path.

    Parameters
    ----------
    session_path : str | Path
        Path to the selected session directory. Strings are converted to
        ``Path`` so path handling is delegated to ``pathlib`` rather than manual
        string splitting.

    Returns
    -------
    str
        Final path component, e.g. ``'/data/mouse/day/session_1'`` becomes
        ``'session_1'``.
    '''

    return Path(session_path).name                           # ``Path.name`` returns only the final path component.

#===============================================================================



#===============================================================================
# 2| Build Session-Name DateTime Reader
#===============================================================================

def _build_datetime_reader(
        *,
        convention: str | None = None,                    # Optional named datetime convention.
        formats: str | list[str] | None = None,           # Optional explicit datetime format or ordered formats.
        pattern: str | None = None,                       # Optional regex isolating the datetime token.
        dayfirst: bool = False,                           # Ambiguous-date hint used during automatic parsing.
        yearfirst: bool = False,                          # Ambiguous-date hint used during automatic parsing.
    ) -> Callable[[str | Path], datetime]:                # Returns one configured path -> datetime callable.
    '''
    Build the reader used to parse datetimes from session directory names.

    A named ``convention`` and explicit ``formats`` are alternative ways of
    specifying the parser, so callers must choose at most one. The parsing rule
    is resolved once when ``DateTimeExtractor`` is constructed rather than once
    per selected session.

    Parameters
    ----------
    convention : str | None
        Optional named convention understood by ``name_datetime_reader_for``.
        Mutually exclusive with ``formats``.
    formats : str | list[str] | None
        Explicit ``datetime.strptime`` format or ordered list of formats passed
        to ``read_datetime_from_name`` when no named convention is supplied.
    pattern : str | None
        Optional regular-expression pattern used to isolate a datetime token
        from a longer session folder name.
    dayfirst : bool
        Hint passed to automatic datetime parsing when date ordering is
        ambiguous. Default False.
    yearfirst : bool
        Hint passed to automatic datetime parsing when date ordering is
        ambiguous. Default False.

    Returns
    -------
    Callable[[str | Path], datetime]
        Single-argument callable that parses a datetime from a session path.
    '''

    #=== 1| Reject Competing Parser Specifications ==============================

    if convention is not None and formats is not None:      # Both independently define how the datetime should be parsed.
        raise ValueError('pass either convention or formats, not both.')

    #=== 2| Resolve a Named Convention ==========================================

    if convention is not None:
        return name_datetime_reader_for(
            convention,
            pattern=pattern,
        )

    #=== 3| Configure the General Name Reader ===================================

    return partial(
        read_datetime_from_name,
        formats=formats,
        pattern=pattern,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
    )

#===============================================================================




################################################################################
# Extractor Classes
################################################################################



#===============================================================================
# 1| LabelExtractor
#===============================================================================

class LabelExtractor:
    '''
    Pair one session-metadata key with one path reader.

    ``LabelExtractor`` is a concrete configuration object rather than an
    abstract base class. It does not require subclasses to implement a redundant
    ``extract`` method. Any path-to-value callable can be supplied directly, and
    specialised extractors simply configure this common behaviour.

    Parameters
    ----------
    reader : MetadataReader
        Callable of the form ``reader(session_path) -> metadata_value``.
    key : str
        Key under which the value should be stored in ``SessionRef.metadata``.

    Attributes
    ----------
    key : str
        Metadata key paired with the reader's result.
    _reader : MetadataReader
        Configured path-to-value callable used by ``label_for``.
    '''

    def __init__(
            self,
            reader: MetadataReader,                         # Callable deriving one metadata value from a session path.
            *,
            key: str = 'label',                            # SessionRef.metadata key paired with the derived value.
        ) -> None:                                          # Stores the extractor configuration and returns nothing.
        '''
        Store the metadata key and reader used by this extractor.

        Parameters
        ----------
        reader : MetadataReader
            Callable that derives one metadata value from a session path.
        key : str
            Metadata key paired with the returned value. Default ``'label'``.

        Returns
        -------
        None
            Configuration is stored on ``self``.
        '''

        if not callable(reader):                             # Fail immediately rather than when selection later attempts to use the extractor.
            raise TypeError('reader must be callable.')
        if not isinstance(key, str) or not key:              # Metadata dictionaries require a meaningful non-empty string key.
            raise ValueError('key must be a non-empty string.')

        self.key = key
        self._reader = reader

    def label_for(
            self,
            session_path: str | Path,                       # Selected session directory from which metadata is derived.
        ) -> tuple[str, object]:                             # Returns the metadata key and derived value together.
        '''
        Return the metadata key and value produced for ``session_path``.

        Parameters
        ----------
        session_path : str | Path
            Selected session directory passed to the configured reader.

        Returns
        -------
        tuple[str, object]
            ``(self.key, self._reader(session_path))`` ready to insert into
            ``SessionRef.metadata``.
        '''

        return self.key, self._reader(session_path)

#===============================================================================




#===============================================================================
# 2| NameExtractor
#===============================================================================

class NameExtractor(LabelExtractor):
    '''
    Configure ``LabelExtractor`` to return the session folder name.

    Parameters
    ----------
    key : str
        Metadata key used to store the folder name. Default ``'label'``.
    '''

    def __init__(
            self,
            *,
            key: str = 'label',                            # SessionRef.metadata key used for the folder-name value.
        ) -> None:                                          # Configures the inherited reader and returns nothing.
        '''
        Configure folder-name metadata extraction.

        Parameters
        ----------
        key : str
            Metadata key used to store the final folder name. Default
            ``'label'``.

        Returns
        -------
        None
            Configuration is passed to ``LabelExtractor``.
        '''

        super().__init__(
            _read_session_name,
            key=key,
        )

#===============================================================================



#===============================================================================
# 3| DateTimeExtractor
#===============================================================================

class DateTimeExtractor(LabelExtractor):
    '''
    Configure ``LabelExtractor`` to parse a datetime from the session name.

    Parameters
    ----------
    convention : str | None
        Optional named datetime convention. Mutually exclusive with ``formats``.
    formats : str | list[str] | None
        Optional explicit datetime format or ordered list of formats.
    pattern : str | None
        Optional regex isolating the datetime token in a longer folder name.
    dayfirst : bool
        Hint for ambiguous automatic parsing. Default False.
    yearfirst : bool
        Hint for ambiguous automatic parsing. Default False.
    key : str
        Metadata key used to store the parsed datetime. Default ``'datetime'``.
    '''

    def __init__(
            self,
            *,
            convention: str | None = None,                 # Optional named parser convention.
            formats: str | list[str] | None = None,        # Optional explicit datetime format(s).
            pattern: str | None = None,                    # Optional regex isolating the datetime token.
            dayfirst: bool = False,                        # Ambiguous-date hint for automatic parsing.
            yearfirst: bool = False,                       # Ambiguous-date hint for automatic parsing.
            key: str = 'datetime',                         # SessionRef.metadata key for the parsed datetime.
        ) -> None:                                          # Configures the inherited reader and returns nothing.
        '''
        Configure datetime metadata extraction from session folder names.

        The actual parser construction is handled by
        ``_build_datetime_reader``. This constructor only coordinates that helper
        with the common ``LabelExtractor`` configuration.

        Parameters
        ----------
        convention : str | None
            Optional named convention understood by
            ``name_datetime_reader_for``. Mutually exclusive with ``formats``.
        formats : str | list[str] | None
            Explicit format or formats passed to ``read_datetime_from_name``.
        pattern : str | None
            Optional regex isolating the datetime token.
        dayfirst : bool
            Day-first hint for ambiguous automatic parsing. Default False.
        yearfirst : bool
            Year-first hint for ambiguous automatic parsing. Default False.
        key : str
            Metadata key used to store the parsed datetime. Default
            ``'datetime'``.

        Returns
        -------
        None
            Configured reader/key are passed to ``LabelExtractor``.
        '''

        reader = _build_datetime_reader(
            convention=convention,
            formats=formats,
            pattern=pattern,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
        )

        super().__init__(
            reader,
            key=key,
        )

#===============================================================================




################################################################################
# Public API
################################################################################

__all__ = [
    'LabelExtractor',
    'NameExtractor',
    'DateTimeExtractor',
]

################################################################################
