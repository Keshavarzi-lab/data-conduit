'''
Select session subfolders from a data directory, and label each one.
--------------------------------------------------------------------

Description:
    Before we can load and combine sessions, we need to choose WHICH
    session folders to use. ``select_sessions`` walks a data directory
    down to the level where the session folders live and returns a dict
    describing the chosen sessions (one entry per session), carrying its
    path plus any labels (Task 1 + Task 2 of the brief). It reads no data;
    it only inspects folder names.

    Filter modes:
      * ALL      -> pass neither include nor exclude_names (keep every folder).
      * INCLUDE  -> include=[...] keeps only the named session folders.
      * EXCLUDE  -> exclude_names=[...] keeps everything except those named.

    Configurable depth:
      Sessions are not always the immediate children of the directory you
      point at. ``depth`` says how many levels down the session folders
      are: ``depth=0`` means "the immediate subfolders are sessions";
      ``depth=3`` means "descend mouse / phase / day, and the folders at
      that fourth level are sessions" (the Q_C layout). The names of the
      intermediate levels are attached to each session as metadata via
      ``level_names``.

    Labelling:
      Each session is tagged using one or more ``extractors`` (see
      extractors.py). With no extractor the default is the folder NAME as
      a string (under ``'label'``).

Contents:
--------------------------------
- _meta_from_path:      Turn intermediate folder names into a metadata dict.
- _resolve_extractors:  Normalise the extractors argument to a list.
- _walk_to_depth:       Recurse into directories yielding (intermediate, session_path).
- _leaf_selector:       Build the session-level selector from include/exclude.
- select_sessions:      Walk root and return {session_key: {path, level_meta, labels}}.
'''





################################################################################
# Imports
################################################################################

from pathlib import Path

# Selector matching reused from the IO utilities:
#   _matches_selector decides whether a name passes a selector
#       (None=all, list=include, callable=custom);
#   _passes_selector is the same but keyed by depth;
#   exclude(...) builds the "everything except these" callable.
from data_conduit._refactor_template.datastructures.extractors import LabelExtractor, NameExtractor
from data_conduit._refactor_template.core.utils.utils_core import (
    _matches_selector,
    _parse_selectors,
    _passes_selector,
    exclude,
)

################################################################################




################################################################################
# Private Helpers
################################################################################



#===============================================================================
# 1| Turn Intermediate Folder Names Into Metadata
#===============================================================================
def _meta_from_path(
        path_keys: tuple,
        level_names,
) -> dict:
    '''
    Turn the intermediate folder names into a metadata dict.

    E.g. ``path_keys=('FbR', 'Testing', 'Day 1')`` with
    ``level_names=('mouseID', 'phase', 'day')`` becomes
    ``{'mouseID': 'FbR', 'phase': 'Testing', 'day': 'Day 1'}``.

    If ``level_names`` is None or too short, the unnamed positions fall
    back to ``level_0``, ``level_1``, ... so this never raises on a
    length mismatch.

    ----------
    Parameters:
        path_keys (tuple):
            The intermediate folder names, one per level above the session.
        level_names (sequence of str | None):
            The names to attach to each intermediate position. None falls
            back to ``level_{i}`` everywhere.
    Returns:
        dict:
            Mapping from level name to folder name.
    '''

    # 1| Resolve names: pad with level_{i} when missing, default to all
    #    level_{i} if no names supplied.
    if level_names is None:
        names = [f'level_{i}' for i in range(len(path_keys))]
    else:
        names = list(level_names)
        if len(names) < len(path_keys):
            names += [f'level_{i}' for i in range(len(names), len(path_keys))]

    # 2| Zip and return. strict=False: extra names (more names than levels)
    #    are simply unused.
    return dict(zip(names, path_keys, strict=False))

#===============================================================================



#===============================================================================
# 2| Normalise the Extractors Argument to a List
#===============================================================================
def _resolve_extractors(
        extractors,
) -> list[LabelExtractor]:
    '''
    Normalise the ``extractors`` argument into a list of extractors.

    Accepts None (-> the default ``NameExtractor``), a single extractor,
    or a list/tuple of them, so callers can pass whichever is convenient.

    ----------
    Parameters:
        extractors (None | LabelExtractor | iterable of LabelExtractor):
            What the caller passed.
    Returns:
        list[LabelExtractor]:
            A list of extractor instances, never empty.
    '''

    # None -> default to a single NameExtractor (folder name as 'label').
    if extractors is None:
        return [NameExtractor()]

    # Single extractor -> wrap in a list.
    if isinstance(extractors, LabelExtractor):
        return [extractors]

    # Iterable -> materialise as a list.
    return list(extractors)

#===============================================================================



#===============================================================================
# 3| Recursive Walk to the Session Depth
#===============================================================================
def _walk_to_depth(
        node: Path,
        current_depth: int,
        depth: int,
        leaf_selector,
        level_selectors,
        prefix,
):
    '''
    Recurse into directories, yielding ``(intermediate_names, session_path)``.

    ``current_depth`` is the depth of ``node``'s children. We descend
    until the children sit at ``depth`` (those are the sessions). Along
    the way:
      * at intermediate levels we apply any ``level_selectors`` for that depth;
      * at the session level we apply ``leaf_selector`` (the
        include/exclude/all filter).
    ``prefix`` accumulates the intermediate folder names (one per level
    above the session), which become the session's path metadata.

    ----------
    Parameters:
        node (Path):
            The directory currently being iterated.
        current_depth (int):
            Depth of ``node``'s children (0 at the top-level walk).
        depth (int):
            The depth at which sessions live.
        leaf_selector (None | list | callable):
            Selector applied at the session level.
        level_selectors (dict[int, selector]):
            Per-depth selectors applied at intermediate levels.
        prefix (tuple):
            Folder names accumulated from levels above ``node``.
    Yields:
        tuple[tuple, Path]:
            ``(intermediate_folder_names, session_path)``.
    '''

    # Sort for a deterministic, reproducible session order.
    for child in sorted(node.iterdir(), key=lambda p: p.name):

        # Skip non-directories: sessions (and the levels above them) are folders.
        if not child.is_dir():
            continue

        # Intermediate level (e.g. mouse / phase / day).
        if current_depth < depth:
            # Apply the optional selector for this depth.
            if not _passes_selector(child.name, current_depth, level_selectors):
                continue
            # Descend, remembering this folder name in the prefix.
            yield from _walk_to_depth(
                child, current_depth + 1, depth, leaf_selector, level_selectors,
                (*prefix, child.name),
            )

        # Session level: apply the include/exclude/all filter.
        else:
            if not _matches_selector(child.name, leaf_selector):
                continue
            yield prefix, child

#===============================================================================



#===============================================================================
# 4| Build the Session-Level Selector From Include / Exclude
#===============================================================================
def _leaf_selector(
        include,
        exclude_names,
):
    '''
    Build the session-level selector from the include/exclude/all choice.

    Returns one of:
      * a list             -> include only these (matched by _matches_selector);
      * an exclude(...) callable -> everything except those listed;
      * None               -> keep all.
    Include and exclude are mutually exclusive.

    ----------
    Parameters:
        include (sequence | None):
            Names to keep.
        exclude_names (sequence | None):
            Names to drop.
    Returns:
        None | list | callable:
            The selector consumed by ``_matches_selector``.
    '''

    # Reject ambiguous combinations early.
    if include is not None and exclude_names is not None:
        raise ValueError('pass either include or exclude_names, not both.')

    # INCLUDE: list -> _matches_selector keeps only these.
    if include is not None:
        return list(include)

    # EXCLUDE: callable -> keep everything not listed.
    if exclude_names is not None:
        return exclude(*exclude_names)

    # ALL: None -> _matches_selector keeps everything.
    return None

#===============================================================================



################################################################################




################################################################################
# Public API
################################################################################



#===============================================================================
# 1| select_sessions (Top-Level Walk + Labelling)
#===============================================================================
def select_sessions(
        root: str | Path,
        *,
        depth: int = 0,
        include=None,
        exclude_names=None,
        extractors=None,
        level_names=None,
        **level_selectors,
) -> dict[str, dict]:
    '''
    Select session folders under ``root`` and describe each with path + labels.

    ----------
    Parameters:
        root (str | Path):
            The data directory to search.
        depth (int):
            How many levels below ``root`` the session folders sit. ``0``
            (default) means the immediate subfolders are sessions; ``3``
            means descend three levels (e.g. mouse/phase/day) and treat
            the next level as sessions.
        include (sequence of str | None):
            INCLUDE mode: keep only session folders whose name is in this list.
        exclude_names (sequence of str | None):
            EXCLUDE mode: keep every session folder except those named here.
            Mutually exclusive with ``include``; passing neither keeps
            ALL folders.
        extractors (LabelExtractor | sequence of LabelExtractor | None):
            How to label each session. Default (None) uses
            ``NameExtractor`` -> the folder name as a string under
            ``'label'``. Pass several to attach several labels (each
            under its own key).
        level_names (sequence of str | None):
            Names for the intermediate levels (length up to ``depth``),
            e.g. ``('mouseID', 'phase', 'day')``. Attached to each
            session as metadata.
        **level_selectors:
            Optional ``l{n}_selector`` filters for the INTERMEDIATE
            levels (depths ``0 .. depth-1``), using the same
            string/list/callable form as ``collect_dfs``. For example,
            ``l1_selector='Testing'`` to descend only the Testing phase.
            (The session level itself is filtered by include / exclude.)
    Returns:
        dict[str, dict]:
            One entry per selected session, keyed by the session folder
            name (a relative path is used instead if two sessions share
            a name). Each value is ``{'path': Path, <level_names...>:
            ..., <label keys...>: ...}``.
    '''

    # 1| Validate root up front.
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f'root is not a directory: {root}')

    # 2| Resolve the session-level filter and the labelling extractors.
    leaf_selector = _leaf_selector(include, exclude_names)
    extractor_list = _resolve_extractors(extractors)

    # 3| The l{n}_selector kwargs describe the intermediate levels; parse
    #    them into a {depth: selector} mapping the walk understands.
    parsed_levels = _parse_selectors(level_selectors)

    # 4| Walk the tree to the session depth, building an entry per session.
    sessions: dict[str, dict] = {}
    for intermediate, session_path in _walk_to_depth(
        root, 0, depth, leaf_selector, parsed_levels, (),
    ):
        # 4a| Start with the session's path plus intermediate-level metadata
        #     (mouse / phase / day / ...).
        entry = {'path': session_path}
        entry.update(_meta_from_path(intermediate, level_names))

        # 4b| Attach each extractor's label (default: folder name under 'label').
        for extractor in extractor_list:
            label_key, label_value = extractor.label_for(session_path)
            entry[label_key] = label_value

        # 4c| Key by folder name; if two sessions share a name, fall back to
        #     the path relative to root so keys stay unique and informative.
        key = session_path.name
        if key in sessions:
            key = str(session_path.relative_to(root))
        sessions[key] = entry

    return sessions

#===============================================================================



################################################################################
