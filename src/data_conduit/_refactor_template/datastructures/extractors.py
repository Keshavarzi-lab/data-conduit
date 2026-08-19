'''
Label extractors: turn a session subfolder into a label to attach to it.
------------------------------------------------------------------------

Description:
    When we select experimental sessions from a directory (see
    selection.py), we want to tag each one with a short LABEL (a value
    we can later put in a column / coordinate so we can tell rows from
    different sessions apart, sort them, group them, etc.

    By default that label is simply the session subfolder's NAME, kept as
    a plain string. Sometimes you want a richer label derived from the
    name). For example, parsing a timestamp-style folder name into a real
    datetime so sessions can be ordered chronologically. An EXTRACTOR is
    the small, swappable object that does that derivation. Keeping it
    separate means new label sources can be added later without touching
    the selection code.

    You can hand selection several extractors at once; each writes its
    value under its own metadata key, so one session can carry several
    labels.

Contents:
--------------------------------
- LabelExtractor:       Abstract base for any extractor.
- NameExtractor:        Default. Label = the folder name, as a string.
- DateTimeExtractor:    Parses the folder name into a datetime.
'''





################################################################################
# Imports
################################################################################

# ABC = Abstract Base Class. It lets us define a template/interface that
# other classes must follow; abstractmethod marks a method that subclasses
# MUST fill in.
from abc import ABC, abstractmethod
from datetime import datetime

# functools.partial takes a function and pre-fills some of its arguments,
# handing back a new function that only needs the remaining ones. We use
# it to bake the chosen datetime-parsing rules into a one-argument reader.
from functools import partial
from pathlib import Path

# Already-built helpers do the actual datetime parsing of a name string:
#   read_datetime_from_name(name, formats=..., ...) -> datetime
#   name_datetime_reader_for(convention) -> a ready-made one-argument parser.
from data_conduit._refactor_template.core.datetime import (
    name_datetime_reader_for,
    read_datetime_from_name,
)

################################################################################




################################################################################
# Base
################################################################################



#===============================================================================
# 1| LabelExtractor (Abstract Base)
#===============================================================================
class LabelExtractor(ABC):
    '''
    Abstract base for deriving a label from a session subfolder.

    "Abstract" means you never use ``LabelExtractor`` directly. You use
    one of its concrete subclasses (NameExtractor, DateTimeExtractor, ...).
    The base only fixes the shape every extractor shares:

      * a ``key``: the name of the metadata field/column the label is
        stored under;
      * ``extract(path)``: compute the label value from a session path;
      * ``label_for(path)``: convenience that returns ``(key, value)``
        together.

    To add a new kind of label later, subclass this and implement
    ``extract``.

    ----------
    Attributes:
        key (str):
            The metadata key the extracted value is stored under. Each
            subclass sets its own (e.g. 'label' for a name, 'datetime'
            for a parsed timestamp).
    '''

    # Default key. Subclasses (or constructor args) override this.
    key: str = 'label'

    @abstractmethod
    def extract(
            self,
            session_path: str | Path,
    ) -> object:
        '''
        Return the label value derived from ``session_path``.

        Subclasses implement this. It receives the session directory path
        and returns whatever the label should be (a string, a datetime,
        ...).

        ----------
        Parameters:
            session_path (str | Path):
                Path to the session subfolder.
        Returns:
            object:
                The label value (subclass-defined type).
        '''

    def label_for(
            self,
            session_path: str | Path,
    ) -> tuple[str, object]:
        '''
        Return ``(key, value)`` for ``session_path``.

        This is the method selection actually calls. It simply pairs
        this extractor's ``key`` with the value that ``extract``
        computes, so the caller can drop ``{key: value}`` straight into
        a session's metadata.

        ----------
        Parameters:
            session_path (str | Path):
                Path to the session subfolder.
        Returns:
            tuple[str, object]:
                ``(self.key, self.extract(session_path))``.
        '''
        return self.key, self.extract(session_path)

#===============================================================================



################################################################################




################################################################################
# Built-in Extractors
################################################################################



#===============================================================================
# 1| NameExtractor (Default: Folder Name as String)
#===============================================================================
class NameExtractor(LabelExtractor):
    '''
    Default extractor: the subfolder name, as a plain string.

    This is what selection uses when you don't ask for anything fancier.
    The name is returned exactly as-is, deliberately NOT parsed into a
    datetime, so a folder called ``'2026-04-30T170040Z'`` gives you the
    STRING ``'2026-04-30T170040Z'`` under the key ``'label'``.

    ----------
    Parameters:
        key (str):
            Metadata key to store the name under. Default ``'label'``.
    '''

    def __init__(
            self,
            *,
            key: str = 'label',
    ) -> None:
        '''
        Remember which metadata key the folder name should be stored under.

        ----------
        Parameters:
            key (str):
                Metadata key for the extracted name. Default ``'label'``.
        Returns:
            None.
        '''
        # The key is customisable so you could, say, store the name under
        # 'session_name' instead of the generic 'label'.
        self.key = key

    def extract(
            self,
            session_path: str | Path,
    ) -> str:
        '''
        Return the final part of the path (the folder name) as a string.

        ``Path(...).name`` is the last path component, e.g.
        ``Path('/data/mouseA/2026-04-30T170040Z').name ==
        '2026-04-30T170040Z'``.

        ----------
        Parameters:
            session_path (str | Path):
                Path to the session subfolder.
        Returns:
            str:
                The folder name.
        '''
        return Path(session_path).name

#===============================================================================



#===============================================================================
# 2| DateTimeExtractor (Parse Folder Name into datetime)
#===============================================================================
class DateTimeExtractor(LabelExtractor):
    '''
    Extractor that parses the subfolder name into a datetime.

    Use this when the folder name encodes a timestamp and you want a real
    datetime (so sessions can be ordered in time, etc.). It does no
    parsing of its own. It delegates to the shared helpers in
    globaltimes.datetime:

      * give it a named ``convention`` (e.g. ``'iso_date'``) and it uses
        the matching format;
      * give it explicit ``formats`` (one or a list of ``strptime``
        patterns) and it tries those;
      * give it neither and the format is AUTO-DETECTED from the name.

    A ``pattern`` regex can pull the timestamp out of a noisier name
    (e.g. ``'Testing_2026-04-30'``).

    ----------
    Parameters:
        convention (str | None):
            Named convention from ``COMMON_DATETIME_CONVENTIONS``
            (e.g. ``'iso_date'``). Cannot be combined with ``formats``.
        formats (str | list[str] | None):
            Explicit ``strptime`` format(s) to try. If both ``convention``
            and ``formats`` are None, the format is auto-detected.
        pattern (str | None):
            Regex isolating the datetime token within the name.
        dayfirst (bool):
            Hint for auto-detection when day/month order is ambiguous.
        yearfirst (bool):
            Hint for auto-detection when year position is ambiguous.
        key (str):
            Metadata key to store the datetime under. Default
            ``'datetime'``.
    '''

    # Default key for this extractor (overridable via the constructor).
    key = 'datetime'

    def __init__(
            self,
            *,
            convention: str | None = None,
            formats: str | list[str] | None = None,
            pattern: str | None = None,
            dayfirst: bool = False,
            yearfirst: bool = False,
            key: str = 'datetime',
    ) -> None:
        '''
        Build a one-argument name->datetime reader from the chosen rules.

        ----------
        Parameters:
            convention (str | None):
                Named datetime convention (mutually exclusive with formats).
            formats (str | list[str] | None):
                Explicit strptime formats to try.
            pattern (str | None):
                Regex isolating the datetime token from the folder name.
            dayfirst (bool):
                Day-first hint for ambiguous auto-detection.
            yearfirst (bool):
                Year-first hint for ambiguous auto-detection.
            key (str):
                Metadata key under which to store the parsed datetime.
        Returns:
            None.
        '''

        # 1| convention and explicit formats are two different ways to say
        #    the same thing, so allowing both would be ambiguous: reject early.
        if convention is not None and formats is not None:
            raise ValueError('pass either convention or formats, not both.')

        self.key = key

        # 2| Pre-build the parser and stash it on the instance, so that every
        #    later extract() call is just "run the parser on this name".
        if convention is not None:
            # name_datetime_reader_for already returns a ready one-argument
            # parser bound to the named convention's format.
            self._reader = name_datetime_reader_for(convention, pattern=pattern)
        else:
            # Otherwise pre-fill read_datetime_from_name's parsing options
            # with functools.partial, leaving only the path/name to be
            # supplied later. formats=None here means "auto-detect".
            self._reader = partial(
                read_datetime_from_name,
                formats=formats,
                pattern=pattern,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
            )

    def extract(
            self,
            session_path: str | Path,
    ) -> datetime:
        '''
        Parse the folder name into a datetime using the bound reader.

        ----------
        Parameters:
            session_path (str | Path):
                Path to the session subfolder.
        Returns:
            datetime:
                The parsed datetime.
        '''
        # The reader pulls the name out of the path itself, so we can pass
        # the whole path straight through.
        return self._reader(session_path)

#===============================================================================



################################################################################
