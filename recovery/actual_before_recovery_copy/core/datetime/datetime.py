"""Extract datetimes from filesystem locations and directory names."""

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Named layouts are opt-in conveniences; no convention is applied implicitly.
COMMON_DATETIME_CONVENTIONS: dict[str, str] = {
    "iso_date": "%Y-%m-%d",
    "iso_datetime": "%Y-%m-%dT%H:%M:%S",
    "compact_date": "%Y%m%d",
    "compact_datetime": "%Y%m%d_%H%M%S",
    "day_first": "%d-%m-%Y",
    "month_first": "%m-%d-%Y",
}


def _extract_token(text: str, pattern: str) -> str:
    """Return the datetime-bearing part of ``text`` matched by ``pattern``."""
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"pattern {pattern!r} did not match {text!r}.")
    return match.group(1) if match.groups() else match.group(0)


def _parse_datetime_string(
    text: str,
    *,
    formats: str | list[str] | None = None,
    dayfirst: bool = False,
    yearfirst: bool = False,
) -> datetime:
    """Convert a datetime string using explicit formats or auto-detection."""
    text = text.strip()

    if formats is None:
        try:
            timestamp = pd.to_datetime(text, dayfirst=dayfirst, yearfirst=yearfirst)
        except Exception as exc:  # noqa: BLE001 - add parsing context to any backend error.
            raise ValueError(f"Could not auto-detect a datetime in {text!r}.") from exc
        if pd.isna(timestamp):
            raise ValueError(f"Could not auto-detect a datetime in {text!r}.")
        return timestamp.to_pydatetime()

    candidate_formats = [formats] if isinstance(formats, str) else formats
    for fmt in candidate_formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"None of the formats {list(candidate_formats)!r} matched {text!r}.")


def _resolve_convention(convention: str) -> str:
    """Return the ``strptime`` format registered for ``convention``."""
    if convention not in COMMON_DATETIME_CONVENTIONS:
        available = sorted(COMMON_DATETIME_CONVENTIONS)
        raise KeyError(f"Unknown convention {convention!r}. Available: {available}.")
    return COMMON_DATETIME_CONVENTIONS[convention]


def extract_datetime(
    location: str | Path,
    reader: Callable[..., datetime],
    **reader_kwargs: Any,
) -> datetime:
    """Extract a datetime by applying ``reader`` to a filesystem location.

    ``location`` is normalised to a :class:`~pathlib.Path`; additional keyword
    arguments are passed directly to the reader.
    """
    return reader(Path(location), **reader_kwargs)


def read_datetime_from_name(
    location: str | Path,
    *,
    formats: str | list[str] | None = None,
    pattern: str | None = None,
    dayfirst: bool = False,
    yearfirst: bool = False,
) -> datetime:
    """Parse a datetime from the final component of ``location``.

    If ``pattern`` contains a capture group, its first group is parsed;
    otherwise the whole match is parsed. Without a pattern, the complete name
    is used. Explicit ``formats`` are tried in order. When they are omitted,
    :func:`pandas.to_datetime` performs auto-detection.
    """
    name = Path(location).name
    text = _extract_token(name, pattern) if pattern is not None else name
    return _parse_datetime_string(
        text,
        formats=formats,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
    )


def name_datetime_reader_for(
    convention: str,
    *,
    pattern: str | None = None,
) -> Callable[[str | Path], datetime]:
    """Build a name reader bound to a named datetime convention."""
    fmt = _resolve_convention(convention)

    def reader(location: str | Path) -> datetime:
        return read_datetime_from_name(location, formats=fmt, pattern=pattern)

    reader.__name__ = f"read_datetime_from_name[{convention}]"
    reader.__qualname__ = reader.__name__
    return reader
