'''
Walk a Bonsai experiment tree and combine SessionSettings across sessions.

Contents
--------
- make_usid
    Build a Unique Session Identifier from (mouse, day, session).
- make_utid
    Build a Unique Trial Identifier from a USID and 1-based trial index.
- collect_session_settings_directory
    Walk a top-level directory shaped as ``{mouse}/{day}/{session}/...`` and
    return ``{'sessions': df, 'trials': df}``. Each DataFrame is tagged with
    USID (and UTID for trials) plus mouse/day/session identifier columns so
    the rows from many sessions can be filtered or grouped together.
'''

'''Directory walker that turns a Bonsai tree into two long-form DataFrames.

make_usid(mouse, day, session) -> str — builds '{mouse}__{day}__{session}'. Deterministic, joinable across notebooks.
make_utid(usid, n) -> str — builds '{usid}__trial_{n}'.
collect_session_settings_directory(base_dir, *, mouse_pattern='*', day_pattern='Day*', session_pattern='*', expected_columns=None, strict=False, verbose=False) — walks {base}/{mouse}/{day}/{session}/..., instantiates the existing data_conduit.SessionSettings reader on each session directory, returns {'sessions': df, 'trials': df}. The sessions frame is one row per discovered JSONL file with USID + identifier columns + every metadata field. The trials frame is one row per trial across the whole tree with UTID + USID + identifier columns + every per-trial setting.Directory walker that turns a Bonsai tree into two long-form DataFrames.

make_usid(mouse, day, session) -> str — builds '{mouse}__{day}__{session}'. Deterministic, joinable across notebooks.
make_utid(usid, n) -> str — builds '{usid}__trial_{n}'.
collect_session_settings_directory(base_dir, *, mouse_pattern='*', day_pattern='Day*', session_pattern='*', expected_columns=None, strict=False, verbose=False) — walks {base}/{mouse}/{day}/{session}/..., instantiates the existing data_conduit.SessionSettings reader on each session directory, returns {'sessions': df, 'trials': df}. The sessions frame is one row per discovered JSONL file with USID + identifier columns + every metadata field. The trials frame is one row per trial across the whole tree with UTID + USID + identifier columns + every per-trial setting.'''

#===== Imports

from pathlib import Path

import pandas as pd

from data_conduit.datasources.monosource import SessionSettings


def make_usid(mouse: str, day: str, session: str) -> str:
    '''
    Build a Unique Session Identifier from the directory components.

    The format is deterministic so the same (mouse, day, session) triple
    always reconstructs the same USID — useful when joining tables across
    notebooks or reloading partial data.

    Parameters
    ----------
    mouse : str
        Mouse-level directory name (e.g. ``'FbR_M01569522'``).
    day : str
        Day-level directory name (e.g. ``'Day 01'``).
    session : str
        Session-level directory name (e.g. ``'2026-03-26T13-29-40'``).

    Returns
    -------
    str
        ``'{mouse}__{day}__{session}'``.
    '''
    return f'{mouse}__{day}__{session}'


def make_utid(usid: str, trial_index: int) -> str:
    '''
    Build a Unique Trial Identifier for a 1-based trial index within a USID.

    Parameters
    ----------
    usid : str
        Output of :func:`make_usid` for the parent session.
    trial_index : int
        1-based index of the trial within the session.

    Returns
    -------
    str
        ``'{usid}__trial_{trial_index}'``.
    '''
    return f'{usid}__trial_{trial_index}'


def _iter_session_directories(
        base: Path,
        mouse_pattern: str,
        day_pattern: str,
        session_pattern: str,
):
    '''
    Yield ``(mouse_name, day_name, session_name, session_path)`` for each
    session-level directory under ``base`` that matches the supplied glob
    patterns. Sorted alphabetically at every level so output is stable.
    '''
    for mouse_path in sorted(p for p in base.glob(mouse_pattern) if p.is_dir()):
        for day_path in sorted(p for p in mouse_path.glob(day_pattern) if p.is_dir()):
            for session_path in sorted(
                p for p in day_path.glob(session_pattern) if p.is_dir()
            ):
                yield mouse_path.name, day_path.name, session_path.name, session_path


def _flatten_session_settings_leaf(leaf) -> list[tuple[str, pd.DataFrame | None, pd.DataFrame | None]]:
    '''
    Normalise the ``SessionSettings.df`` shape into ``[(stem, metadata, trials), ...]``.

    A session directory typically yields ``{'metadata': df, 'trials': df}``
    (one JSONL file). When more than one JSONL is present, the shape becomes
    ``{stem: {'metadata': df, 'trials': df}, ...}``, possibly nested under a
    leading ``'SessionSettings'`` key. This helper walks both shapes.
    '''
    if not isinstance(leaf, dict) or not leaf:
        return []

    if {'metadata', 'trials'}.issubset(leaf.keys()):
        return [('', leaf.get('metadata'), leaf.get('trials'))]

    pairs: list[tuple[str, pd.DataFrame | None, pd.DataFrame | None]] = []
    for key, value in leaf.items():
        if isinstance(value, dict) and {'metadata', 'trials'}.issubset(value.keys()):
            pairs.append((key, value.get('metadata'), value.get('trials')))
        elif isinstance(value, dict):
            pairs.extend(
                (f'{key}/{sub_key}' if sub_key else key, meta, trials)
                for sub_key, meta, trials in _flatten_session_settings_leaf(value)
            )
    return pairs


def _metadata_columns_to_record(metadata_df: pd.DataFrame | None) -> dict:
    '''
    Reduce a metadata DataFrame to a flat ``{column: value}`` dict suitable
    for a sessions-table row. If multiple metadata rows are present (rare),
    only the first is kept; subsequent rows are ignored on the assumption
    that they describe the same session.
    '''
    if not isinstance(metadata_df, pd.DataFrame) or metadata_df.empty:
        return {}
    first_row = metadata_df.iloc[0]
    return {col: first_row[col] for col in metadata_df.columns}


def collect_session_settings_directory(
        base_dir: str | Path,
        *,
        mouse_pattern: str = '*',
        day_pattern: str = 'Day*',
        session_pattern: str = '*',
        expected_columns: dict | None = None,
        strict: bool = False,
        verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    '''
    Walk a Bonsai experiment tree and combine SessionSettings into long-form tables.

    The function loads every session under ``base_dir`` using the existing
    :class:`~data_conduit.datasources.monosource.SessionSettings` reader and
    concatenates the results into two DataFrames:

    - ``sessions`` — one row per discovered ``SessionSettings`` JSONL file,
      tagged with mouse / day / session columns and a Unique Session
      Identifier (USID). All metadata columns from the JSONL record are
      preserved, plus a ``n_trials`` column computed from the trials frame.
    - ``trials`` — one row per trial across all sessions, tagged with USID,
      a Unique Trial Identifier (UTID), the parent mouse / day / session,
      and the 1-based ``trial_index`` within its session. All trial-setting
      columns produced by ``SessionSettings`` (with list expansion and
      schema reconciliation already applied) are retained.

    Expected layout::

        base_dir/
        └── {mouse}/
            └── {day}/
                └── {session}/
                    └── SessionSettings/
                        └── SessionSettings/
                            └── *.jsonl

    Sessions without a ``SessionSettings`` folder are skipped silently
    (or noted when ``verbose=True``).

    Parameters
    ----------
    base_dir : str or Path
        Top-level directory of the Bonsai tree (e.g. ``.../BonsaiFiles``).
    mouse_pattern, day_pattern, session_pattern : str
        Glob patterns used to filter the three directory levels. Defaults
        match the typical Bonsai layout (any mouse, any directory starting
        with ``'Day'``, any session).
    expected_columns : dict or None
        Forwarded to :class:`SessionSettings` for trial-schema reconciliation.
    strict : bool
        Forwarded to :class:`SessionSettings`. If True, only the expected
        columns are kept on each trials frame.
    verbose : bool
        If True, print a progress line for each session.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{'sessions': sessions_df, 'trials': trials_df}``. Either frame may
        be empty if no matching sessions or trials are found.
    '''
    base = Path(base_dir)
    if not base.is_dir():
        raise NotADirectoryError(f"base_dir does not exist or is not a directory: {base}")

    sessions_records: list[dict] = []
    trials_frames: list[pd.DataFrame] = []

    for mouse, day, session, session_path in _iter_session_directories(
        base, mouse_pattern, day_pattern, session_pattern,
    ):
        usid = make_usid(mouse, day, session)

        try:
            ss = SessionSettings(
                experiment_directory_path=str(session_path),
                expected_columns=expected_columns,
                strict=strict,
                verbose=False,
            )
        except Exception as error:
            if verbose:
                print(f"[skip] {usid}: SessionSettings load failed: {error}")
            continue

        pairs = _flatten_session_settings_leaf(ss.df)
        if not pairs:
            if verbose:
                print(f"[empty] {usid}: no SessionSettings JSONL found")
            continue

        for stem, metadata_df, trials_df in pairs:
            n_trials = int(len(trials_df)) if isinstance(trials_df, pd.DataFrame) else 0

            session_record = {
                'USID': usid,
                'mouse': mouse,
                'day': day,
                'session': session,
                'session_settings_file': stem,
                'n_trials': n_trials,
                **_metadata_columns_to_record(metadata_df),
            }
            sessions_records.append(session_record)

            if isinstance(trials_df, pd.DataFrame) and not trials_df.empty:
                tagged = trials_df.copy()
                trial_indices = list(range(1, n_trials + 1))
                tagged.insert(0, 'UTID', [make_utid(usid, i) for i in trial_indices])
                tagged.insert(1, 'USID', usid)
                tagged.insert(2, 'mouse', mouse)
                tagged.insert(3, 'day', day)
                tagged.insert(4, 'session', session)
                tagged.insert(5, 'trial_index', trial_indices)
                trials_frames.append(tagged)

        if verbose:
            print(f"[ok] {usid}: {len(pairs)} session-settings file(s)")

    sessions_df = pd.DataFrame(sessions_records)
    trials_df = (
        pd.concat(trials_frames, axis=0, ignore_index=True)
        if trials_frames else pd.DataFrame()
    )

    return {'sessions': sessions_df, 'trials': trials_df}


__all__ = [
    'collect_session_settings_directory',
    'make_usid',
    'make_utid',
]
