'''
A basic Catalog / DataStructure model: readers + configurators over selected sessions.
=====================================================================================

Description:
    A small, deliberately simple front end for turning a directory of sessions into
    one concatenated, multi-session dataset. Two user-facing pieces:

      * Catalog       : holds the per-session loading recipe. It has READERS (named
                        ``path -> data object`` callables, the ordinary sources such
                        as VideoData / DLCPose / Nosepoke) and CONFIGURATORS (named
                        ``objects -> objects`` callables that run AFTER all readers
                        in a session, see EVERY loaded object at once, and do
                        cross-object work such as "stamp the VideoData times onto
                        the DLC frames"). Configurators are the general replacement
                        for the older clock-specific ``sync`` mechanism.

      * DataStructure : applies a Catalog to a root directory under level selectors
                        (include / exclude / all + per-level filters). Its single
                        job, in one call, is: find every selected session, apply the
                        catalog to each, then CONCATENATE each stream across all of
                        those sessions, attaching the directory levels (session id,
                        mouse, day, ...) to every entry so the concatenated data can
                        be sorted, grouped, or split by session.

    What this module deliberately does NOT do: it does not touch the sessiongroups
    ``sync`` / TTL cross-clock machinery, ``normalise_to_zero``, global clocks, or
    the ``Session`` wrapper. It reuses the clean lower-level primitives
    (``select_sessions`` for the directory walk, ``_object_to_bundle`` for coercing
    a source object into named members, ``combine_bundles`` for the ordered
    cross-session concatenation) and adds the reader/configurator front end on top.

    Ordering note: ``select_sessions`` walks the tree in sorted order at every
    level, so sessions come back hierarchically sorted (mouse, then the
    intermediate levels, then the session folder). Concatenation preserves exactly
    that order, so for one mouse/day, session 1 trial 50 precedes session 2 trial 1,
    and session 1 time 10 precedes session 2 time 1.

Contents:
--------------------------------
- Reader:           Type alias for a ``path -> object`` stream loader.
- Configurator:     Type alias for an ``objects -> objects`` cross-object transform.
- Catalog:          Ordered readers + configurators; reads one session.
- DataStructure:    Applies a Catalog across selected sessions and concatenates them.
'''





################################################################################
# Imports
################################################################################

import warnings
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Reused clean primitives from sessiongroups (NOT the sync / Session machinery):
#   * select_sessions   : directory walk with include / exclude / all + level filters.
#   * _object_to_bundle : coerce one source object into named members.
#   * combine_bundles   : concatenate matching members across sessions, in order.
from data_conduit.sessiongroups.builders import _object_to_bundle
from data_conduit.sessiongroups.bundle import Bundle, combine_bundles
from data_conduit.sessiongroups.selection import select_sessions

################################################################################




# A reader loads ONE stream from ONE session directory: it takes the session
# path and returns a data object (a source instance such as VideoData / DLCPose,
# a bare DataArray / Dataset / DataFrame, or a mapping of those).
Reader = Callable[[Path], object]

# A configurator runs AFTER all readers in a session. It receives the mapping of
# {reader_name: object} for that session and returns a (possibly modified)
# mapping. It is free to read several objects and replace one (or add new ones).
Configurator = Callable[[dict], dict]




################################################################################
# Catalog
################################################################################



#===============================================================================
# 1| Catalog (Ordered Readers + Configurators)
#===============================================================================
class Catalog:
    '''
    An ordered collection of readers and configurators for one experiment layout.

    A catalog answers two questions about a session: WHICH streams to load (the
    readers) and HOW to post-process them once they are all loaded (the
    configurators). It is a small, mutable configuration object: you build it
    with chained ``add_reader`` / ``add_configurator`` calls and then hand it to a
    ``DataStructure``. Readers run first, in the order added; configurators run
    afterwards, in the order added, each receiving the full set of objects.

    ----------
    Parameters:
        readers (Mapping[str, Reader] | None):
            Optional initial readers as ``{name: path -> object}``. Default: none.
        configurators (Mapping[str, Configurator] | None):
            Optional initial configurators as ``{name: objects -> objects}``.
            Default: none.
    '''

    #---------------------------------------------------------------------------
    # 1.1| Construction
    #---------------------------------------------------------------------------
    def __init__(
            self,
            readers: Mapping[str, Reader] | None = None,
            configurators: Mapping[str, Configurator] | None = None,
    ) -> None:
        '''
        Seed the catalog with optional initial readers and configurators.

        ----------
        Parameters:
            readers (Mapping[str, Reader] | None):
                Initial ``{name: reader}`` entries, kept in iteration order.
            configurators (Mapping[str, Configurator] | None):
                Initial ``{name: configurator}`` entries, kept in iteration order.
        Returns:
            None.
        '''
        # Order-preserving name -> callable maps. Reuse the add_* methods so the
        # duplicate-name checks apply to seeded entries too.
        self._readers: dict[str, Reader] = {}
        self._configurators: dict[str, Configurator] = {}
        for name, reader in (readers or {}).items():
            self.add_reader(name, reader)
        for name, configurator in (configurators or {}).items():
            self.add_configurator(name, configurator)


    #---------------------------------------------------------------------------
    # 1.2| Editing (chainable)
    #---------------------------------------------------------------------------
    def add_reader(
            self,
            name: str,
            reader: Reader,
    ) -> 'Catalog':
        '''
        Register a reader under ``name`` (returns self, so calls chain).

        ----------
        Parameters:
            name (str):
                Stream name, e.g. ``'video'`` or ``'dlc'``. Becomes the member
                name (or member prefix) once the object is coerced into members.
                Must be unique among readers.
            reader (Reader):
                A ``path -> object`` callable loading this stream from a session
                directory.
        Returns:
            Catalog:
                The catalog itself (for chaining).
        '''
        # A duplicate name would silently shadow the earlier reader, so guard it.
        if name in self._readers:
            raise ValueError(f'a reader named {name!r} is already in the catalog.')
        self._readers[name] = reader
        return self

    def add_configurator(
            self,
            name: str,
            configurator: Configurator,
    ) -> 'Catalog':
        '''
        Register a configurator under ``name`` (returns self, so calls chain).

        ----------
        Parameters:
            name (str):
                Identifier for this configurator (for ordering / introspection
                only; it does not name a member). Must be unique among
                configurators.
            configurator (Configurator):
                An ``objects -> objects`` callable run after all readers, given
                the session's ``{reader_name: object}`` mapping.
        Returns:
            Catalog:
                The catalog itself (for chaining).
        '''
        if name in self._configurators:
            raise ValueError(f'a configurator named {name!r} is already in the catalog.')
        self._configurators[name] = configurator
        return self


    #---------------------------------------------------------------------------
    # 1.3| Reading one session
    #---------------------------------------------------------------------------
    def read_session(
            self,
            session_path: str | Path,
    ) -> dict:
        '''
        Apply every reader to one session, then run every configurator.

        ----------
        Parameters:
            session_path (str | Path):
                The session directory the readers are pointed at.
        Returns:
            dict:
                ``{reader_name: object}`` after all configurators have run. Keys
                may differ from the reader names if a configurator added or
                removed entries.
        '''
        # 1| Run the readers, in order, each on the session path. A reader that
        #    fails (e.g. its files are absent for this session) is SKIPPED with a
        #    warning rather than aborting the whole load: that stream is simply
        #    missing for this session, and the combine keeps it for the sessions
        #    that do have it.
        path = Path(session_path)
        objects = {}
        for name, reader in self._readers.items():
            try:
                objects[name] = reader(path)
            except Exception as error:
                warnings.warn(
                    f'reader {name!r} skipped for session {path.name!r}: '
                    f'{type(error).__name__}: {error}',
                    stacklevel=2,
                )

        # 2| Run the configurators, in order, threading the (possibly modified)
        #    object mapping through each. A configurator sees every object, so it
        #    can do cross-stream work (e.g. stamp video times onto DLC frames).
        #    A configurator that fails for this session (e.g. its events are too
        #    truncated to build a trial table) is SKIPPED with a warning rather
        #    than aborting the whole load, exactly like a failed reader above: the
        #    derived stream is simply missing for this session. Configurators are
        #    expected to leave the object mapping consistent even when they raise
        #    (see the trials configurator, which drops its raw events up front).
        for name, configurator in self._configurators.items():
            try:
                objects = configurator(objects)
            except Exception as error:
                warnings.warn(
                    f'configurator {name!r} skipped for session {path.name!r}: '
                    f'{type(error).__name__}: {error}',
                    stacklevel=2,
                )
        return objects


    #---------------------------------------------------------------------------
    # 1.4| Introspection
    #---------------------------------------------------------------------------
    @property
    def reader_names(self) -> list[str]:
        '''Return the reader names in insertion order.'''
        return list(self._readers)

    @property
    def configurator_names(self) -> list[str]:
        '''Return the configurator names in insertion order.'''
        return list(self._configurators)

    def __repr__(self) -> str:
        '''Show the reader names and configurator names.'''
        return (
            f'<Catalog readers={self.reader_names} '
            f'configurators={self.configurator_names}>'
        )

#===============================================================================



################################################################################




################################################################################
# Private Helper
################################################################################



#===============================================================================
# 1| Attach a Per-Session Level As a Coordinate / Column
#===============================================================================
def _attach_level(
        combined: Bundle,
        level_name: str,
        session_to_value: dict,
        *,
        dim: str, 
        base: str,
) -> Bundle:
    '''
    Add a provenance level (e.g. mouse / day) to every stream of a combined result.

    ``combine_bundles`` already tags each entry with its source session under
    ``base`` (a coordinate along ``dim`` for xarray streams, a column for table
    streams). The other levels (mouse, day, ...) are constant per session, so this
    maps that per-entry session tag through ``session_to_value`` and attaches the
    result as a new coordinate / column, entry-aligned to the existing session tag.

    ----------
    Parameters:
        combined (Bundle):
            A combined result carrying the ``base`` session tag on every stream.
        level_name (str):
            Name of the level to attach (e.g. ``'mouse'``).
        session_to_value (dict):
            Mapping from session id to that session's value for this level.
        dim (str):
            The alignment axis the provenance rides along (for xarray streams).
        base (str):
            Name of the existing per-entry session tag to map from.
    Returns:
        Bundle:
            A new combined result with ``level_name`` attached to every stream.
    '''

    out: Bundle = {}
    for name, obj in combined.items():
        # xarray stream: read the per-entry session coord, map it to this level's
        # value, and attach the mapped values as a coord along dim.
        if isinstance(obj, xr.DataArray | xr.Dataset):
            sessions = obj.coords[base].values
            values = np.array([session_to_value.get(s) for s in sessions], dtype=object)
            out[name] = obj.assign_coords({level_name: (dim, values)})
        # table stream: same idea, but the session tag is a column and the mapped
        # level becomes a new column.
        elif isinstance(obj, pd.DataFrame):
            sessions = obj[base].to_numpy()
            obj = obj.copy()
            obj[level_name] = [session_to_value.get(s) for s in sessions]
            out[name] = obj
        # Anything else is passed through untouched.
        else:
            out[name] = obj
    return out

#===============================================================================



################################################################################




################################################################################
# DataStructure
################################################################################



#===============================================================================
# 1| DataStructure (Apply a Catalog Across Selected Sessions and Concatenate)
#===============================================================================
class DataStructure:
    '''
    Apply a Catalog across the selected sessions and concatenate them.

    A DataStructure is the single operation: given a root directory, a catalog,
    and level selectors, ``load`` finds every selected session, applies the
    catalog (readers then configurators) to each, and concatenates each stream
    across all of those sessions, attaching the directory levels (session id,
    mouse, day, ...) to every entry so the result can be sorted / grouped / split
    by session. There is no separate per-session result and no separate combine
    step: ``load`` returns the concatenated data directly.

    The selection arguments are passed straight through to ``select_sessions``, so
    the same conventions apply: ``depth`` says how far below ``root`` the session
    folders sit, ``level_names`` names the intermediate directory levels (e.g.
    ``('mouse', 'phase', 'day')``), ``l{n}_selector`` keyword filters restrict the
    intermediate levels, and the session level itself is filtered by ``include`` /
    ``exclude`` (pass neither to keep all).

    ----------
    Parameters:
        root (str | Path):
            The data directory to search.
        catalog (Catalog):
            The readers + configurators to apply to each selected session.
        depth (int):
            How many levels below ``root`` the session folders sit (``0`` =
            immediate subfolders). Default 0.
        level_names (Sequence[str] | None):
            Names for the intermediate directory levels, attached to every entry
            of the concatenated result as provenance. Default: none.
        include (Sequence[str] | None):
            INCLUDE mode: keep only session folders whose name is in this list.
        exclude (Sequence[str] | None):
            EXCLUDE mode: keep every session folder except those named here.
            Mutually exclusive with ``include``; passing neither keeps ALL.
        extractors (LabelExtractor | Sequence | None):
            Optional label extractors forwarded to ``select_sessions``. Default:
            folder name.
        **level_selectors:
            Optional ``l{n}_selector`` filters for the intermediate levels,
            forwarded to ``select_sessions`` (e.g. ``l1_selector='Testing'``).
    '''

    #---------------------------------------------------------------------------
    # 1.1| Construction
    #---------------------------------------------------------------------------
    def __init__(
            self,
            root: str | Path,
            catalog: Catalog,
            *,
            depth: int = 0,
            level_names=None,
            include=None,
            exclude=None,
            extractors=None,
            **level_selectors,
    ) -> None:
        '''
        Store the root, catalog, and selection configuration (no I/O yet).

        ----------
        Parameters:
            root (str | Path):
                The data directory to search.
            catalog (Catalog):
                The readers + configurators to apply.
            depth (int):
                Levels below ``root`` where sessions sit.
            level_names (Sequence[str] | None):
                Names for the intermediate directory levels.
            include (Sequence[str] | None):
                INCLUDE-mode session-name allowlist.
            exclude (Sequence[str] | None):
                EXCLUDE-mode session-name denylist.
            extractors (LabelExtractor | Sequence | None):
                Label extractors forwarded to ``select_sessions``.
            **level_selectors:
                ``l{n}_selector`` intermediate-level filters.
        Returns:
            None.
        '''
        # Pure configuration: nothing is read until load() is called.
        self.root = Path(root)
        self.catalog = catalog
        self.depth = depth
        self.level_names = tuple(level_names) if level_names is not None else ()
        self.include = include
        self.exclude = exclude
        self.extractors = extractors
        self.level_selectors = level_selectors
        # Populated by load(); None until then.
        self.data: Bundle | None = None


    #---------------------------------------------------------------------------
    # 1.2| Selection (reuse select_sessions)
    #---------------------------------------------------------------------------
    def select(self) -> dict[str, dict]:
        '''
        Find the selected session folders and their per-level labels.

        Thin wrapper over ``select_sessions`` using this DataStructure's stored
        configuration. The result is ordered (hierarchically sorted by the
        directory walk), which is the order concatenation preserves. Mostly useful
        for inspecting which sessions matched before loading them.

        Returns:
            dict[str, dict]:
                ``{session_id: {'path': Path, <level_names...>: ..., ...}}`` in
                walk order.
        '''
        return select_sessions(
            self.root,
            depth=self.depth,
            include=self.include,
            exclude_names=self.exclude,
            extractors=self.extractors,
            level_names=self.level_names or None,
            **self.level_selectors,
        )


    #---------------------------------------------------------------------------
    # 1.3| Load (apply the catalog to every session AND concatenate across them)
    #---------------------------------------------------------------------------
    def load(
            self,
            *,
            dim: str = 'Time',
            session_coord: str = 'session',
    ) -> Bundle:
        '''
        Apply the catalog to every selected session and concatenate across sessions.

        This is the DataStructure's whole job, in one call: select the session
        folders, run the catalog (readers then configurators) on each, then
        concatenate each stream across the selected sessions, in walk order. Every
        entry of every concatenated stream is tagged with its session id plus each
        directory level (mouse, day, ...) as a coordinate (xarray streams) or a
        column (table streams), so the result can be sorted, grouped, or split by
        session.

        Streams are reconciled by UNION: a stream that only some sessions provide
        (e.g. DLC pose, present on some days only) is still kept, concatenated over
        just the sessions that have it. Combined with the warn-and-skip behaviour
        of ``read_session``, this means a mix of sessions with and without a given
        stream loads cleanly rather than erroring.

        ----------
        Parameters:
            dim (str):
                Axis to concatenate xarray streams along. Table (DataFrame)
                streams always concatenate by row. Default ``'Time'``.
            session_coord (str):
                Name of the per-entry session tag. Default ``'session'``.
        Returns:
            dict[str, xr.DataArray | xr.Dataset | pd.DataFrame]:
                One concatenated object per stream, each carrying the session id
                and every directory level as a coordinate / column on its entries.
        '''
        # 1| Select the sessions. They come back hierarchically sorted by the
        #    directory walk, which is the order concatenation preserves.
        selection = self.select()
        if not selection:
            raise ValueError(
                f'no sessions selected under {self.root} (check depth / include / '
                f'exclude / level selectors).'
            )

        # 2| Apply the catalog to each session and reduce its objects to
        #    concatenable members (so a DLCPose becomes 'dlc:position' /
        #    'dlc:confidence', a VideoData becomes 'video', and so on).
        per_session: list[Bundle] = []
        session_ids: list[str] = []
        metadata: dict[str, dict] = {}
        for session_id, info in selection.items():
            objects = self.catalog.read_session(info['path'])
            members: Bundle = {}
            for name, obj in objects.items():
                members.update(_object_to_bundle(obj, name, prefix_data_arrays=True))
            per_session.append(members)
            session_ids.append(session_id)
            metadata[session_id] = info

        # 3| Union of stream names across sessions, in first-appearance order. A
        #    stream missing from some sessions is still kept; it is concatenated
        #    over only the sessions that provide it (tagged with their session id).
        member_names: list[str] = []
        for members in per_session:
            for name in members:
                if name not in member_names:
                    member_names.append(name)

        result: Bundle = {}
        for name in member_names:
            sub_bundles = [{name: members[name]} for members in per_session if name in members]
            sub_keys = [sid for sid, members in zip(session_ids, per_session) if name in members]
            combined = combine_bundles(
                sub_bundles, dim=dim, keys=sub_keys, label_coord=session_coord, on='strict',
            )
            result[name] = combined[name]

        # 4| Add the remaining directory levels (mouse, day, ...) as further
        #    per-entry tags, mapped from the session tag attached in step 3. Each
        #    stream maps its own session tag, so a stream covering only some
        #    sessions still gets the right level values.
        for level in self.level_names:
            session_to_value = {sid: info.get(level) for sid, info in metadata.items()}
            result = _attach_level(result, level, session_to_value, dim=dim, base=session_coord)

        self.data = result
        return result


    #---------------------------------------------------------------------------
    # 1.4| Dict-like access (delegates to self.data after load)
    #---------------------------------------------------------------------------
    def _require_loaded(self):
        # Internal guard: raises if load() has not been called yet.
        if self.data is None:
            raise RuntimeError('call .load() before accessing data.')

    def keys(self):
        '''
        Return the stream names in the loaded bundle.

        Equivalent to ``datastructure.data.keys()``. Raises ``RuntimeError``
        if ``.load()`` has not been called yet.

        Returns:
            KeysView: the stream names (e.g. ``'events'``, ``'dlc:position'``).
        '''
        self._require_loaded()
        return self.data.keys()

    def __getitem__(self, key: str):
        '''
        Return a single stream from the loaded bundle by name.

        Raises ``RuntimeError`` if ``.load()`` has not been called yet.

        Parameters:
            key (str): stream name (e.g. ``'events'``, ``'dlc:position'``).
        Returns:
            xr.DataArray | xr.Dataset | pd.DataFrame: the concatenated stream.
        '''
        self._require_loaded()
        return self.data[key]

    def __iter__(self):
        '''Iterate over stream names. Raises ``RuntimeError`` if not loaded.'''
        self._require_loaded()
        return iter(self.data)

    def __len__(self) -> int:
        '''Return the number of streams in the loaded bundle. Raises ``RuntimeError`` if not loaded.'''
        self._require_loaded()
        return len(self.data)


#===============================================================================



################################################################################
