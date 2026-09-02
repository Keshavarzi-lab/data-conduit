'''
Small selection helpers for the Pose_Analysis_workspace notebook.
----------------------------------------------------------------

Description:
    ``data_conduit.sessiongroups.select_sessions`` filters the SESSION level
    by exact folder name (include / exclude) and the INTERMEDIATE levels (e.g.
    ``day``) by ``l{n}_selector``. For pose analysis we often want a looser,
    one-shot filter: "keep every session whose ``{day}/{session name}`` path
    contains this substring / matches this regex". ``filter_sessions`` is that
    post-filter -- it runs on the dict ``select_sessions`` returns, so it stays
    a thin convenience on top of data-conduit rather than a fork of it.

    ``require_dirs`` additionally drops sessions that lack a given subfolder
    (e.g. ``'DLC'``). That matters before a whole-session ``combine_sessions``:
    the default ``on='intersection'`` keeps only members present in EVERY
    session, so one DLC-less session would silently drop pose from the combined
    result. Filtering to DLC-bearing sessions first avoids that surprise.

    Nothing here reads experiment data; it only inspects the selection dict and
    the on-disk folder layout. If this proves broadly useful it could graduate
    into ``select_sessions`` as a ``session_selector`` argument.

Contents:
--------------------------------
- session_path_string:  Flatten one selection entry to its "{day}/{name}" string.
- filter_sessions:      Keep sessions by substring/regex match and/or required subfolders.
'''

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def session_path_string(
        name: str,
        entry: dict,
        *,
        fields: Sequence[str] = ('day', 'label'),
        sep: str = '/',
) -> str:
    '''
    Flatten one selection entry into a path-like string for matching.

    E.g. with ``fields=('day', 'label')`` an entry
    ``{'day': 'Day1', 'label': '2026-03-26T13-29-40', ...}`` becomes
    ``'Day1/2026-03-26T13-29-40'``. Missing fields fall back to the session
    key ``name`` for the ``label`` slot and are otherwise skipped, so this
    never raises on a partial entry.

    ----------
    Parameters:
        name (str):
            The session key (folder name) from ``select_sessions``.
        entry (dict):
            The selection entry (its ``path``, level metadata, and labels).
        fields (sequence of str):
            Metadata keys to join, in order. Default ``('day', 'label')``.
        sep (str):
            Separator between fields. Default ``'/'``.
    Returns:
        str:
            The flattened path string.
    '''

    parts: list[str] = []
    for field in fields:
        if field in entry:
            parts.append(str(entry[field]))
        elif field == 'label':
            parts.append(str(name))
    return sep.join(parts)


def filter_sessions(
        sessions: dict[str, dict],
        pattern: str | None = None,
        *,
        fields: Sequence[str] = ('day', 'label'),
        sep: str = '/',
        regex: bool = False,
        case_sensitive: bool = False,
        require_dirs: Iterable[str] | None = None,
) -> dict[str, dict]:
    '''
    Filter a ``select_sessions`` dict by path-string match and/or subfolders.

    ----------
    Parameters:
        sessions (dict[str, dict]):
            The output of ``select_sessions`` (``{name: {'path', ...}}``).
        pattern (str | None):
            Match applied to each session's flattened ``{day}/{name}`` string
            (see ``session_path_string``). ``None`` keeps everything (so
            ``require_dirs`` can be used on its own).
        fields (sequence of str):
            Which metadata keys make up the flattened string. Default
            ``('day', 'label')``.
        sep (str):
            Separator used when flattening. Default ``'/'``.
        regex (bool):
            If True, ``pattern`` is a regular expression matched with
            ``re.search``; if False (default), it is a plain substring
            ("contains") test.
        case_sensitive (bool):
            Whether matching respects case. Default False.
        require_dirs (iterable of str | None):
            If given, keep only sessions whose folder contains ALL of these
            subdirectories (e.g. ``('DLC',)`` to keep pose-bearing sessions).
    Returns:
        dict[str, dict]:
            The kept entries, in their original order.
    '''

    required = tuple(require_dirs or ())
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = None
    if pattern is not None:
        compiled = re.compile(pattern if regex else re.escape(pattern), flags)

    kept: dict[str, dict] = {}
    for name, entry in sessions.items():

        # Subfolder gate (e.g. must have a DLC/ folder).
        if required:
            path = Path(entry['path'])
            if not all((path / sub).is_dir() for sub in required):
                continue

        # Path-string match.
        if compiled is not None:
            haystack = session_path_string(name, entry, fields=fields, sep=sep)
            if compiled.search(haystack) is None:
                continue

        kept[name] = entry

    return kept
