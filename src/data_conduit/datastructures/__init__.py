"""Select, load, combine, parse, and slice experimental data streams."""

from .catalog_core import Configurator, Reader, ReaderSpec, StreamCatalog
from .datastructure_core import DataStructure
from .extractors import DateTimeExtractor, LabelExtractor, NameExtractor
from .selection import SessionRef, select_sessions
from .slicing import slice_stream, slice_stream_for_trial, slice_stream_per_trial
from .streams import (
    DataObject,
    StreamContainer,
    StreamMap,
    attach_level,
    object_to_streams,
)
from .trials import TrialSpec, first_matching, parse_trials, segment_bounds, within

__all__ = [
    "Configurator",
    "DataObject",
    "DataStructure",
    "DateTimeExtractor",
    "LabelExtractor",
    "NameExtractor",
    "Reader",
    "ReaderSpec",
    "SessionRef",
    "StreamCatalog",
    "StreamContainer",
    "StreamMap",
    "TrialSpec",
    "attach_level",
    "first_matching",
    "object_to_streams",
    "parse_trials",
    "segment_bounds",
    "select_sessions",
    "slice_stream",
    "slice_stream_for_trial",
    "slice_stream_per_trial",
    "within",
]
