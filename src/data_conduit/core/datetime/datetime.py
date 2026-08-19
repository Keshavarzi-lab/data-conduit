'''
Modular functions for extracting session/subdirectory DateTime information from directory contents.

Description:
	A small, composable toolkit for turning a filesystem location into a `datetime`.

	`extract_datetime` is the base dispatcher: it coerces a location to a `Path`
	and delegates the actual extraction to a pluggable "reader" callable, so new
	sources (file mtime, sidecar metadata, etc.) can be added later without
	touching the dispatcher.

	`read_datetime_from_name` is a reader that parses the datetime out of a
	session/subdirectory name. It supports either auto-detection (via
	`pandas.to_datetime`) or explicit user-supplied formats (via
	`datetime.strptime`), and can isolate a date token from a noisy name with a
	regular-expression `pattern`.

	`name_datetime_reader_for` is the simple wrapper: it binds one of the named
	`COMMON_DATETIME_CONVENTIONS` and returns a ready-to-use reader that can be
	handed straight to `extract_datetime`.
'''


################################################################################
# Imports
################################################################################

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

################################################################################


# Named conventions for simple-to-obtain datetime layouts. Each maps a friendly
# name to a `datetime.strptime` format string. Users opt in by name; nothing
# here is applied as a hidden default.
COMMON_DATETIME_CONVENTIONS: dict[str, str] = {
	'iso_date':         '%Y-%m-%d',
	'iso_datetime':     '%Y-%m-%dT%H:%M:%S',
	'compact_date':     '%Y%m%d',
	'compact_datetime': '%Y%m%d_%H%M%S',
	'day_first':        '%d-%m-%Y',
	'month_first':      '%m-%d-%Y',
}




################################################################################
# Private Helpers
################################################################################

def _extract_token(
	text: str,
	pattern: str,
) -> str:
	'''
	Pull the datetime-bearing substring out of `text` using a regular expression.

	Parameters
	----------
	text : str
		String to search (typically a session/subdirectory name).
	pattern : str
		Regular expression locating the datetime. If it defines a capture group,
		group 1 is returned; otherwise the whole match is returned.

	Returns
	-------
	str
		The matched datetime substring.
	'''
	match = re.search(pattern, text)
	if match is None:
		raise ValueError(f'pattern {pattern!r} did not match {text!r}.')
	return match.group(1) if match.groups() else match.group(0)


def _parse_datetime_string(
	text: str,
	*,
	formats: str | list[str] | None = None,
	dayfirst: bool = False,
	yearfirst: bool = False,
) -> datetime:
	'''
	Convert a datetime string into a `datetime`, auto-detecting or using formats.

	Parameters
	----------
	text : str
		The datetime string to parse.
	formats : str or list of str or None, optional
		One or more `datetime.strptime` format strings to try in order. If None
		(default), the format is auto-detected with `pandas.to_datetime`.
	dayfirst, yearfirst : bool, optional
		Passed to `pandas.to_datetime` to disambiguate ambiguous orderings during
		auto-detection. Ignored when explicit `formats` are supplied.

	Returns
	-------
	datetime
		The parsed datetime.
	'''
	text = text.strip()

	if formats is None:
		try:
			timestamp = pd.to_datetime(text, dayfirst=dayfirst, yearfirst=yearfirst)
		except Exception as exc:  # noqa: BLE001 - re-raised with clearer context below.
			raise ValueError(f'Could not auto-detect a datetime in {text!r}.') from exc
		if pd.isna(timestamp):
			raise ValueError(f'Could not auto-detect a datetime in {text!r}.')
		return timestamp.to_pydatetime()

	if isinstance(formats, str):
		formats = [formats]

	for fmt in formats:
		try:
			return datetime.strptime(text, fmt)
		except ValueError:
			continue
	raise ValueError(f'None of the formats {list(formats)!r} matched {text!r}.')


def _resolve_convention(
	convention: str,
) -> str:
	'''
	Look up the `strptime` format for a named convention.

	Parameters
	----------
	convention : str
		Key into `COMMON_DATETIME_CONVENTIONS`.

	Returns
	-------
	str
		The corresponding `datetime.strptime` format string.
	'''
	if convention not in COMMON_DATETIME_CONVENTIONS:
		available = sorted(COMMON_DATETIME_CONVENTIONS)
		raise KeyError(f'Unknown convention {convention!r}. Available: {available}.')
	return COMMON_DATETIME_CONVENTIONS[convention]


################################################################################




################################################################################
# Public API
################################################################################

def extract_datetime(
	location: str | Path,
	reader: Callable[..., datetime],
	**reader_kwargs,
) -> datetime:
	'''
	Extract a datetime from `location` by applying a pluggable `reader`.

	This is a thin dispatcher: it coerces `location` to a `Path` and forwards it,
	with any extra keyword arguments, to `reader`. The reader decides where the
	datetime comes from (e.g. the directory name, file mtime, a sidecar file).

	Parameters
	----------
	location : str or Path
		Filesystem location to read the datetime from.
	reader : callable
		Callable of the form ``reader(location, **reader_kwargs) -> datetime``.
		Use `read_datetime_from_name` (with keyword arguments) for ad-hoc reads,
		or a pre-bound reader from `name_datetime_reader_for`.
	**reader_kwargs
		Forwarded to `reader`.

	Returns
	-------
	datetime
		The datetime produced by `reader`.
	'''
	return reader(Path(location), **reader_kwargs)


def read_datetime_from_name(
	location: str | Path,
	*,
	formats: str | list[str] | None = None,
	pattern: str | None = None,
	dayfirst: bool = False,
	yearfirst: bool = False,
) -> datetime:
	'''
	Reader that parses a datetime from a session/subdirectory name.

	The final path component (``Path(location).name``) is used. If `pattern` is
	given, the datetime token is first isolated from that name; otherwise the
	whole name is parsed.

	Parameters
	----------
	location : str or Path
		Location whose name encodes the datetime. Only the name is inspected; the
		path need not exist.
	formats : str or list of str or None, optional
		One or more `datetime.strptime` format strings to try in order (user
		specification). If None (default), the format is auto-detected with
		`pandas.to_datetime`.
	pattern : str or None, optional
		Regular expression isolating the datetime token within the name (useful
		for noisy names such as ``'Testing_all_trials_2026-05-12'``). If it has a
		capture group, group 1 is used; otherwise the whole match is used. If None
		(default), the entire name is parsed.
	dayfirst, yearfirst : bool, optional
		Forwarded to auto-detection to disambiguate ambiguous orderings. Ignored
		when explicit `formats` are supplied.

	Returns
	-------
	datetime
		The datetime parsed from the name.
	'''
	name = Path(location).name
	text = _extract_token(name, pattern) if pattern is not None else name
	return _parse_datetime_string(text, formats=formats, dayfirst=dayfirst, yearfirst=yearfirst)


def name_datetime_reader_for(
	convention: str,
	*,
	pattern: str | None = None,
) -> Callable[[str | Path], datetime]:
	'''
	Build a name reader bound to a named convention (the simple wrapper).

	Resolves `convention` to its `strptime` format and returns a single-argument
	reader suitable for `extract_datetime`, so common cases need no format string.

	Parameters
	----------
	convention : str
		Key into `COMMON_DATETIME_CONVENTIONS` (e.g. ``'iso_date'``).
	pattern : str or None, optional
		Regular expression isolating the datetime token within the name, forwarded
		to `read_datetime_from_name`.

	Returns
	-------
	callable
		Reader of the form ``reader(location) -> datetime`` bound to `convention`.
	'''
	fmt = _resolve_convention(convention)

	def reader(location: str | Path) -> datetime:
		return read_datetime_from_name(location, formats=fmt, pattern=pattern)

	reader.__name__ = f'read_datetime_from_name[{convention}]'
	reader.__qualname__ = reader.__name__
	return reader


################################################################################
