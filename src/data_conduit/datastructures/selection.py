"""
Subdirectory Selection submodule for DataStructures in data-conduit.
--------------------------------------------------------------------

Description:
    ``selection.py`` is the first stage of the datastructure pipeline. Its job
    is deliberately narrow: inspect the directory hierarchy, decide which
    session folders should be used, and return a lightweight ``SessionRef`` for
    each selected session. It does NOT read experimental data.

    A ``SessionRef`` keeps three kinds of information separate:
      * ``path``     -> the actual session directory readers should receive;
      * ``levels``   -> hierarchy-derived grouping / identity information such
                        as mouseID, phase, or day;
      * ``metadata`` -> additional information extracted from the session path,
                        such as a human-readable label or parsed datetime.

    Keeping ``levels`` separate from ``metadata`` is intentional. Levels are
    later attached to combined streams by ``attach_level`` so that downstream
    data can still be grouped by mouse / day / phase. Metadata remains attached
    to the ``SessionRef`` but is not automatically broadcast onto every row or
    time point.

    Selection supports three session-folder filtering modes:
      * ALL      -> pass neither include nor exclude_names;
      * INCLUDE  -> ``include=[...]`` keeps only matching session folders;
      * EXCLUDE  -> ``exclude_names=[...]`` keeps everything except those names.

    Intermediate levels can also be filtered with ``l{n}_selector`` keyword
    arguments. The selector interpretation itself is reused from ``utils_core``
    so selection and the rest of DataConduit share one matching convention.

Contents:
--------------------------------
- SessionRef:            Store one selected session path + levels + metadata.
- _meta_from_path:       Convert intermediate folder names into named levels.
- _resolve_extractors:   Normalise the extractor argument into a list.
- _walk_to_depth:        Traverse only as far as the configured session depth.
- _leaf_selector:        Build the session-level include/exclude selector.
- select_sessions:       Return ``{session_key: SessionRef}`` for selected folders.
"""


################################################################################
# Imports
################################################################################

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from operator import index
from pathlib import Path
from typing import Any

from ..core.utils import (
    _matches_selector,
    _parse_selectors,
    _passes_selector,
    exclude,
)
from .extractors import LabelExtractor, NameExtractor

################################################################################


################################################################################
# Types
################################################################################


# ===============================================================================
# 2| Selector Type Aliases
# ===============================================================================

# Selector type: a list, tuple, set, or frozenset of strings; a callable that takes
# a string and returns a bool; a single string; or None.
# Intentionally mirrors the type accepted by _matches_selector() and _passes_selector().

Selector = list[str] | tuple[str, ...] | set[str] | frozenset[str] | Callable[[str], bool] | str | None

# ===============================================================================

# ===============================================================================
# 1| SessionRef (Selected Session Path + Levels + Metadata)
# ===============================================================================


@dataclass
class SessionRef:
    """
    Store the information associated with one selected session directory.

    ``SessionRef`` is intentionally a small data object rather than a behavioural
    class. The outer selection dictionary supplies the session key; this object
    carries everything about that session that later pipeline stages need.

    Parameters
    ----------
    path : Path
        Path to the selected session directory. This is the directory passed to
        every reader in the ``StreamCatalog`` for this session.
    levels : dict[str, Any]
        Grouping / identity values derived from the intermediate directory
        hierarchy. For a layout such as ``mouse / phase / day / session`` this
        might be ``{'mouseID': 'FbR', 'phase': 'Testing', 'day': 'Day 1'}``.
        These values are explicitly treated as levels later and can therefore be
        attached to combined streams.
    metadata : dict[str, Any]
        Additional information describing the session but not automatically
        treated as a grouping level. Values normally come from ``LabelExtractor``
        objects, for example ``{'label': 'session_001', 'datetime': ...}``.
    """

    path: Path  # Filesystem directory represented by this session reference.
    levels: dict[str, Any]  # Hierarchy-derived values that should behave as data levels.
    metadata: dict[str, Any]  # Other session information produced by extractors, not automatically treated as levels.


# ===============================================================================

################################################################################


################################################################################
# Private Helper Functions
################################################################################


# ===============================================================================
# 1| Convert Intermediate Folder/Subdirectory Names into Metadata
# ===============================================================================


def _meta_from_path(
    path_keys: tuple[str, ...],  # Intermediate folder names, one per level above the session.
    level_names: Sequence[str] | None = None,  # The names of the metadata levels/intermediate positions.
) -> dict[str, Any]:  # Returns a dictionary mapping level names to path keys.
    """
    Convert a tuple of intermediate folder names into a named levels dictionary.

    This helper turns the positional information accumulated while walking the
    directory tree into explicit metadata names. For example,
    ``('FbR', 'Testing', 'Day 1')`` together with
    ``('mouseID', 'phase', 'day')`` becomes
    ``{'mouseID': 'FbR', 'phase': 'Testing', 'day': 'Day 1'}``.

    Parameters
    ----------
    path_keys : tuple[str, ...]
            Intermediate folder names, one per level above the session.
            Type is tuple[str,...], i.e. a tuple of strings, where each
            string is the name of a folder in the path leading to the
            session directory.
    level_names : Sequence[str] | None
        The names of the metadata levels/intermediate positions.
        If None, default names are used, e.g. level_0, level_1, etc.
        Type is Sequence[str] | None, i.e. either a sequence of strings
        (e.g. list or tuple) or None. A ``Sequence[str]`` may
        be a list, tuple, or other ordered string sequence.

    Returns
    -------
    dict[str, Any]
        Returns a dictionary mapping level names to path keys.
        The type is dict[str, Any], i.e. a dictionary with string keys
        and values of any type.  Values are currently strings because
        they originate from ``Path.name``, but the wider ``Any`` type
        matches the ``SessionRef.levels`` contract and leaves room for
        future level normalisation.

    Example
    -------
    ``path_keys=('M1', 'Phase A', 'Day 1')`` with
    ``level_names=('mouseID', 'phase', 'day')`` becomes
    ``{'mouseID': 'M1', 'phase': 'Phase A', 'day': 'Day 1'}``.

    """

    # === 1| Assign Every Intermediate Position a Name ==========================
    # If the caller did not supply names, use level_{i} for every position. If
    # they supplied only some names, retain those names and generate defaults
    # only for the remaining positions. This lets callers name the hierarchy as
    # precisely as they know it without requiring an exact-length sequence.

    if level_names is None:  # When no names are supplied, generate default names for every level/path position.
        names = [f"level_{i}" for i in range(len(path_keys))]
    else:
        names = list(level_names)  # Take the supplied names.
        if len(names) < len(path_keys):  #   Only pads with default names when path contains more levels than supplied names.
            names += [f"level_{i}" for i in range(len(names), len(path_keys))]

    # === 2| Pair Each Name with its Folder Name =================================
    # strict = False => where there are more names than levels,
    # then the extra names are simply unused.
    # Should always be strict=False, as step 1 ensures that names is at
    # least as long as path_keys. When this function is called in later
    # stages of the pipeline, it is possible that there may be more names
    # than levels, potentially when there are variable depths/numbers of
    # levels in the directory structure. In such cases, we want to avoid
    return dict(zip(names, path_keys, strict=False))  # raising an error and simply ignore the extra names, hence strict=False.


# ===============================================================================


# ===============================================================================
# 2| Normalise the Extractors Argument to a List
# ===============================================================================
def _resolve_extractors(
    extractors: LabelExtractor | Sequence[LabelExtractor] | None,  # Extractors passed by caller. Can be None, one extractor, or ordered sequence of extractors.
) -> list[LabelExtractor]:  # Returns a non-empty list of LabelExtractor instances.
    """
    Normalises the ``extractors`` argument into a list of extractors
    in ``select_sessions()``.

    As ``select_sessions()`` accepts None (-> the default ``NameExtractor``),
    a single extractor, or a list/tuple of extractors, this helper converts
    every accepted form into a list, so the selection code can iterate
    uniformly regardless of which form was passed to ``select_sessions()``.


    Parameters
    ----------
    extractors : LabelExtractor | Sequence[LabelExtractor] | None
        The extractors passed by the caller. Can be None, a single extractor,
        or an ordered sequence of extractors.
        If ``None``, will use default ``NameExtractor``, which extracts the
        folder name as a string under ``metadata['label']``.
        If a single extractor is passed, it will be wrapped in a list.
        If a sequence of extractors is passed, it will be converted to a list.

    Returns
    -------
    list[LabelExtractor]
        List of extractor instances to apply to every selected session.
        The list is never empty when ``extractors`` is None because the default
        ``NameExtractor`` is inserted.
        If several extractors are passed, each is applied to every session and
        its result stored under that extractor's own ``key`` in
        ``SessionRef.metadata``. Duplicate keys are rejected, so there is always
        one metadata entry per extractor.
    """

    # === 1| Supply the Default Session Label ====================================
    # If the caller did not supply any extractors, use the default ``NameExtractor``,
    # which extracts the folder name as a string under ``metadata['label']``.
    # This ensures that every session has at least one label, even if the caller did
    # not specify any extractors.
    if extractors is None:  # None -> default to a single NameExtractor (folder name as 'label')
        return [NameExtractor()]

    # === 2| Wrap a Single Extractor in a List ===================================
    # If a single extractor is passed, wrap it in a list so that the selection
    # code can iterate uniformly regardless of which form was passed.
    if isinstance(extractors, LabelExtractor):  # Single extractor -> one-element list.
        return [extractors]

    # === 3| Materialise an Extractor Sequence ===================================
    # Convert tuples and other sequences to a list once. This gives the main
    # selection loop a simple, stable representation while preserving order.
    resolved = list(extractors)  # Materialise once while preserving caller order.

    # === 4| Reject Duplicate Metadata Keys ======================================
    # Each extractor writes one entry into SessionRef.metadata under its own
    # key, so two extractors sharing a key would silently overwrite each other.
    # Catch it here, once, before any walking happens.
    keys = [extractor.key for extractor in resolved]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"extractors must use unique metadata keys; duplicated: {duplicates}")
    return resolved


# ===============================================================================


# ===============================================================================
# 3| Recursively Walk the Directory Tree to the Session Depth
# ===============================================================================
def _walk_to_depth(
    node: Path,  # Directory whose immediate children should be inspected now.
    current_depth: int,  # Depth represented by the children of ``node``.
    depth: int,  # Number of intermediate levels before session directories.
    leaf_selector: Selector,  # Selector applied only to session-folder names.
    level_selectors: dict[int, Selector],  # Mapping of intermediate depth -> selector.
    prefix: tuple[str, ...],  # Folder names accumulated above the current node.
) -> Iterator[tuple[tuple[str, ...], Path]]:  # Yields hierarchy labels + selected session path.
    """
    Traverse directories until the session depth is reached, yielding each
    selected session.

    The helper recursively descends the directory tree, applying selectors at each
    level. When it reaches the session depth, it yields the accumulated intermediate
    folder names and the session path as a tuple ``(intermediate_names, session_path)``.

    Unlike a general recursive directory walker, this function does not descend into any
    files or subdirectories inside the session itself. Once a child folder is identified
    as a session (i.e., it is at the specified depth and passes the leaf selector), it is
    yielded and the recursion stops for that branch.

    Parameters
    ----------
    node : Path
        The directory currently being iterated. This is the parent directory whose immediate
        child directories are being inspected. Recursion continues into child directories
        that pass the intermediate level selectors until the session depth is reached.
    current_depth : int
        Depth of ``node``'s children (0 at the top-level walk). This indicates how many levels
        deep the recursion currently is. Value is compared against the target ``depth`` to
        determine whether the child directories are intermediate levels or session directories.
    depth : int
        Number of intermediate levels between ``root`` and the session directories. This is the
        target depth at which session directories are expected to be found.
        ``depth=0`` therefore means immediate children are sessions; ``depth=3`` means descend
        through three intermediate levels before treating the next child directories as sessions.
    leaf_selector : Selector
        Selector to be applied to session-folder names only. Produced by ``_leaf_selector()``
        from the include/exclude/all choice. Determines which session directories are yielded.
        If None, all session directories are yielded; if a list, only those named are yielded;
        if a callable, only those for which the callable returns True are yielded (e.g. exclude(...),
        include(...)).
    level_selectors : dict[int, Selector]
        Mapping from intermediate depth (0 to depth-1) to selectors for those levels.
        Each selector is applied to the names of child directories at that depth, determining which
        intermediate directories are descended into.
        A depth absent from the dictionary accepts every folder at that level.
        Selector matching itself is delegated to ``_passes_selector`` / ``_matches_selector``.
    prefix : tuple[str, ...]
        Folder names accumulated from levels above ``node``. This tuple grows as the recursion descends,
        and is used to construct the ``levels`` dictionary for each session. Each element corresponds
        to an intermediate level above the session, in order from the root down to the session.
        When a session is reached this tuple is returned so those positional names can
        become ``SessionRef.levels``.

    Yields
    ------
    tuple[tuple[str, ...], Path]
        Each yielded value is a tuple containing:
        - A tuple of intermediate folder names (one per level above the session), which will become
          the ``SessionRef.levels`` for that session.
        - The Path to the selected session directory.
        ``(intermediate_folder_names, session_path)`` for every selected session directory.
    """

    # === 1| Iterate over Children in Deterministic Order ========================
    # Sorts by name to ensure that order of selected sessions does not depend on
    # filesystem enumeration order, which can vary across platforms and runs.
    # This ensures reproducibility, and is important as this insertion order may
    # be used in subsequent stages when combining streams across sessions.

    for child in sorted(node.iterdir(), key=lambda p: p.name):  # For each child directory of the current node,
        # sort by name to ensure deterministic order of selected sessions.

        # === 1.1| Ignore Non-Directories at Hierarchy/Session Selection Levels
        # Selection is only concerned with choosing directories. Files within these
        # levels are therefore ignored.
        # This is important because session directories may contain files, but we only
        # want to select the session directories themselves, not their contents.

        if not child.is_dir():  # Skips any child that is not a directory
            continue

        # === 1.2| Handle Intermediate Levels
        # If ``current_depth`` is less than ``depth``, then the child dir is another
        # intermediate level, not a session.
        # Apply any selector(s) that are defined for this depth before next step of
        # recursion. Rejected branches need not be descended into, which saves time
        # and avoids unnecessary processing.
        if current_depth < depth:  # If the current depth < target depth, this child is an intermediate level.
            if not _passes_selector(child.name, current_depth, level_selectors):  # Apply optional selector for this depth. Reject child if it does not pass.
                continue

            # Recurse into the child directory, increasing current_depth by 1, and adding the child name to the prefix.
            yield from _walk_to_depth(
                child,  # Descend into intermediate child directory that passed the selector.
                current_depth + 1,  # Increase current depth by 1 for the next level of recursion.
                depth,  # Target depth for session directories remains fixed throughout recursion.
                leaf_selector,  # Session-level selector remains the same for all branches.
                level_selectors,  # Same intermediate level selector mapping applies recursively.
                (*prefix, child.name),  # Add current accepted hierarchy folder name to the prefix for the next
            )  #   recursion level, so it can become part of SessionRef.levels.

        # === 1.3| Handle Session Level
        # If ``current_depth`` equals ``depth``, then the child dir is a session candidate.
        # Apply the include/exclude/all selector (``leaf_selector``) to determine whether to
        # yield this session or skip it.
        # Does not recurse further into session directories, as we only want to select the
        # session itself, not its contents.

        else:  # If current depth == target depth, this child is a session candidate.
            if not _matches_selector(child.name, leaf_selector):  # Session folder fails required include/exclude/all filter, so skip it.
                continue
                # Yield the accumulated intermediate folder names (prefix) and selected session
            yield prefix, child  # path (child) as a tuple. This represents one selected session.


# ===============================================================================


# ===============================================================================
# 4| Build the Session-Level Selector From Include / Exclude
# ===============================================================================
def _leaf_selector(
    include: Sequence[str] | None,  # Optional allowlist of session names to include. If None, all sessions are included.
    exclude_names: Sequence[str] | None,  # Optional blocklist of session names to exclude. If None, no sessions are excluded.
) -> Selector:  # Returns selector consumed by ``matches_selector``.
    """
    Convert the mutually exlusive include/exclude options into one session-level selector.

    Traversal code requires a single session-level selector to determine which session directories
    to yield. This helper converts the three mutually exclusive options (include, exclude_names, or
    neither) into a single selector that can be passed to ``_matches_selector()``.

    Returns one of:
      * a list                  -> include only these;
      * an exclude(...) callable -> everything except those listed;
      * None                    -> keep all.

    Include and exclude are mutually exclusive.

    Parameters
    ----------
    include : Sequence[str] | None
        Optional allowlist of session folder names to include/keep.
        Mutually exclusive with ``exclude_names``. If None, all sessions are included.
    exclude_names : Sequence[str] | None
        Optional blocklist of session folder names to exclude/drop. Converted to a
        callable selector using ``utils_core.exclude()``. Mutually exclusive with ``include``.
        If None, no sessions are excluded.

    Returns
    -------
    Selector (list[str] | Callable[[str], bool] | None)
        Returns one of:
        - a list of strings -> include only these session names;
        - an exclude(...) callable -> everything except those listed;
        - None -> keep all sessions.
        Selector is consumed by ``_matches_selector()``.

    """

    # === 1| Reject Ambiguous Filter Configuration ===============================
    # Applying both allowlist and blocklist simultaneously is ambiguous and raises
    # questions about ordering/precedence. Reject this configuration early to avoid
    # confusion

    if include is not None and exclude_names is not None:
        raise ValueError("pass either include or exclude_names, not both.")

    # === 2| Build INCLUDE Selector ==============================================
    if include is not None:
        return list(include)  # INCLUDE: list -> _matches_selector keeps only members of this list.

    # === 3| Build EXCLUDE Selector ==============================================
    if exclude_names is not None:
        return exclude(*exclude_names)  # ``exclude(...)`` returns callable that keeps everything not listed in ``exclude_names``.

    # === 4| Default to ALL Selector ==============================================
    return None  # ``_matches_selector(..., None)`` intentionally accepts everything.


# ===============================================================================


################################################################################


################################################################################
# Public API
################################################################################


# ===============================================================================
# 1| select_sessions (Top-Level Walk + SessionRef Construction)
# ===============================================================================
def select_sessions(
    root: str | Path,  # The data directory to search for session folders.
    *,  # Force all following arguments to be keyword-only to prevent positional misplacement.
    depth: int = 0,  # How many levels below ``root`` the session folders sit.
    include: Sequence[str] | None = None,  # Optional allowlist of session folder names. Mutually exclusive with ``exclude_names``.
    exclude_names: Sequence[str] | None = None,  # Optional blocklist of session folder names. Mutually exclusive with ``include``.
    extractors: LabelExtractor | Sequence[LabelExtractor] | None = None,  # Metadata extractors to apply to each selected session.
    level_names: Sequence[str] | None = None,  # Names assigned to the intermediate levels.  Stored under ``SessionRef.levels``.
    **level_selectors: Any,  # Optional ``l{n}_selector`` filters for intermediate levels.
) -> dict[str, SessionRef]:  # Returns dictionary mapping session names to SessionRef objects, one per selected session.
    """
    Select session folders under ``root`` and describe each with a ``SessionRef``.

    Function provides complete selection stage for datastructure pipelines, including:
        * Validate the root directory exists and is a directory.
        * Resolve the session-level filter and the labelling extractors.
        * Parse the intermediate level selectors into a mapping the walk understands.
        * Walk the tree to the session depth, building a SessionRef per session.

    Directory-derived grouping / identity information is stored under
    ``SessionRef.levels``. Extractor-produced information is stored under
    ``SessionRef.metadata``.


    Parameters
    ----------
    root : str | Path
        The data directory to search for session folders. Can be a string or a Path object.
    depth : int, optional
        How many levels below ``root`` the session folders sit. ``0``
        (default) means the immediate subfolders are sessions; ``3``
        means descend three levels (e.g. mouse/phase/day) and treat
        the next level as sessions.
    include : Sequence[str] | None, optional
        INCLUDE mode: keep only session folders whose name is in this list.
        If None, all sessions are included. Mutually exclusive with ``exclude_names``,
        must be None if ``exclude_names`` is not None.
    exclude_names : Sequence[str] | None, optional
        EXCLUDE mode: keep every session folder except those named here.
        If None, no sessions are excluded. Mutually exclusive with ``include``,
        must be None if ``include`` is not None.
    extractors : LabelExtractor | Sequence[LabelExtractor] | None
        Metadata extraction rule(s) applied to each selected session path. Each
        extractor returns one ``(key, value)`` pair stored in
        ``SessionRef.metadata``. None preserves the historical default of using a
        ``NameExtractor`` so the folder name is stored under ``'label'``.
    level_names : Sequence[str] | None
        Names assigned positionally to the intermediate hierarchy folders. For
        example ``('mouseID', 'phase', 'day')`` converts the three path positions
        into explicit ``SessionRef.levels`` keys. Missing names are filled with
        ``level_{i}``; extra names are ignored for shorter observed paths.
    **level_selectors : Any
        Optional intermediate level filters, passed as ``l{n}_selector`` keyword arguments.
        e.g. ``l0_selector=['MouseA', 'MouseB']`` or ``l1_selector='Testing'``.
        Each selector is applied to the names of child directories at that depth, determining
        which intermediate directories are descended into.
        A depth absent from the dictionary accepts every folder at that level.
        Selector matching itself is delegated to ``_parse_selectors()``.

    Returns
    -------
    dict[str, SessionRef]
        One entry per selected session, keyed by the session folder name (a relative path is used instead if two sessions share
        a name). Each ``SessionRef`` stores the session path, its hierarchy-derived levels, and its extractor-derived metadata.

    """

    # === 1| Validate and Normalise the Root Directory ===========================
    # Convert the root to a Path object and check that it is a directory.
    # Raise an error if not.

    root = Path(root)  # Convert root to a Path object for consistent handling of filesystem paths.
    if not root.is_dir():  # Check that root is a directory. Raise error to prevent further processing on invalid path.
        raise NotADirectoryError(f"root is not a directory: {root}")

    try:
        depth = index(depth)
    except TypeError as error:
        raise TypeError("depth must be a non-negative integer.") from error
    if depth < 0:
        raise ValueError("depth must be a non-negative integer.")

    if level_names is not None:
        if not all(isinstance(name, str) and name for name in level_names):
            raise TypeError("level_names must contain non-empty strings.")
        duplicates = sorted(name for name, count in Counter(level_names).items() if count > 1)
        if duplicates:
            raise ValueError(f"level_names must be unique; duplicated: {duplicates}.")

    # === 2| Resolve Session-Level Filtering and Metadata Extraction =============
    leaf_selector = _leaf_selector(include, exclude_names)  # Resolve the session-level filter from include/exclude options into a single selector.
    extractor_list = _resolve_extractors(extractors)  # One concrete list now represents every accepted extractor input form.

    # === 3| Parse Intermediate-Level Selector Keywords ===========================
    # ``_parse_selectors`` converts names such as l1_selector into integer-depth
    # keys. The recursive walker can then look up selectors directly by depth.
    parsed_levels = _parse_selectors(level_selectors)  # Result shape: {0: selector, 1: selector, ...}.
    out_of_range = sorted(level for level in parsed_levels if level >= depth)
    if out_of_range:
        raise ValueError(f"level selectors {out_of_range} are outside depth={depth}; selectors may target intermediate levels 0 through depth - 1.")

    # === 4| Walk to the Session Depth and Build SessionRefs ======================
    # The dictionary preserves walk insertion order, which later becomes the
    # natural order used when corresponding streams are combined

    sessions: dict[str, SessionRef] = {}  # Init dict to hold selected sessions, keyed by session name (relative path if duplicate).

    # Determine basename collisions from the complete hierarchy, not merely the
    # currently selected subset. This keeps a session's identifier stable when a
    # level selector includes or excludes a sibling with the same folder name.
    all_session_names = Counter(
        session_path.name
        for _, session_path in _walk_to_depth(
            root,
            0,
            depth,
            None,
            {},
            (),
        )
    )

    for intermediate, session_path in _walk_to_depth(
        root,  # Start the walk at the root directory.
        0,  # Current depth is 0 at the root level.
        depth,  # Target depth for session directories, as specified by the caller.
        leaf_selector,  # Apply include/exclude/all only at the session level (leaf).
        parsed_levels,  # Apply any hierarchy selectors at their matching depths.
        (),  # Prefix starts empty; grows as recursion descends into intermediate levels.
    ):
        # === 4.1| Convert Hierarchy Position into Explicit Session Levels =========
        # These levels are stored in ``SessionRef.levels`` rather than in
        # `SessionRef.metadata```, as the latter stores arbitrary metadata, whilst
        # StreamContainer later broadcasts levels onto combined streams via ``attach_levels()``.
        # This distinction is important for later stages of the pipeline.

        levels = _meta_from_path(intermediate, level_names)  # Convert the tuple of intermediate folder names into a dictionary of named levels.

        # === 4.2| Extract Additional Session Metadata ============================
        # Each extractor produces one key/value pair stored in ``SessionRef.metadata``.
        # This allows arbitrary metadata to be attached to each session, beyond the
        # hierarchy-derived levels.
        # Later extractors with the same key will raise an error, ensuring that each
        # metadata key is unique per session.

        metadata: dict[str, Any] = {}  # Init dict to hold metadata extracted from the session path.
        for extractor in extractor_list:  # For each extractor, apply it to the session path.
            metadata_key, metadata_value = extractor.label_for(session_path)  # Extractor decides both metadata key and value type.
            metadata[metadata_key] = metadata_value  # Store arbitrary metadata without treating it as a level.

        # === 4.3| Construct a Stable, Human-Readable Session Dictionary Key ======
        # Folder names are the normal identifiers because they are concise and
        # familiar. Only when a collision is actually encountered do we fall back
        # to a relative path for the later session.

        key = str(session_path.relative_to(root)) if all_session_names[session_path.name] > 1 else session_path.name

        # === 4.4| Create the SessionRef and Store it in the Dictionary =============
        sessions[key] = SessionRef(
            path=session_path,  # Concrete directory readers will receive.
            levels=levels,  # Hierarchy values that later become attachable levels.
            metadata=metadata,  # Other extracted session information retained with the reference.
        )

    # === 5| Return the Ordered Selection Mapping =================================

    return sessions  # Empty dict is valid here; DataStructure decides whether an empty load is an error.


# ===============================================================================


################################################################################
